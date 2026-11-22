"""Compare two frozen annotation files. Never reads dataset gold."""
import argparse,copy,hashlib,json,re,unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    # Accept JSONC without modifying the supplied annotation.
    text=re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*[\s\S]*?\*/',lambda m:m.group(1) if m.group(1) else '',text)
    text=re.sub(r'("(?:\\.|[^"\\])*")|,(\s*[}\]])',lambda m:m.group(1) if m.group(1) else m.group(2),text)
    return json.loads(text)

def norm(value):
    if isinstance(value,bool) or value is None:return value
    if isinstance(value,(int,float,Decimal)):return str(Decimal(str(value)).normalize())
    if isinstance(value,str):
        value=unicodedata.normalize('NFC',value).strip()
        if re.fullmatch(r'-?\d+(?:\.\d+)?',value):return str(Decimal(value).normalize())
        return re.sub(r'\s+',' ',value).casefold().replace('ё','е')
    if isinstance(value,tuple):return [norm(v) for v in value]
    if isinstance(value,list):return sorted([norm(v) for v in value],key=lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True))
    return {k:norm(v) for k,v in sorted(value.items())}

def dump(value):return json.dumps(norm(value),ensure_ascii=False,sort_keys=True,separators=(',',':'))

def role(task):
    text=norm(task['title'])
    obj=next((v for k,v in [('спецификац','specification'),('маршрут','route'),('подтверждение отгруз','shipment_confirmation'),('паспорт','passport'),('сертификат','certificate'),('счет','invoice'),('пропуск','gate_pass'),('прием','receipt'),('принять','receipt')] if k in text),text)
    intent='check' if any(v in text for v in ['провер','сверить']) else 'prepare' if 'маршрут' in text else 'receive' if obj=='receipt' else 'obtain'
    return (task['purchase_id'],intent,obj)

class Canonical:
    def __init__(self,item,source):
        self.item=item;self.source=source;self.tasks={k:role(v) for k,v in source['input']['initial_state']['tasks'].items()}
        self.phase_map={};self.order={};index=0
        for step in source['input'].get('timeline',[]):
            sid=step['episode_id'];self.phase_map[sid]=sid;self.order[sid]=index;index+=1
            for event in step.get('events',[]):self.phase_map[event['id']]=sid
            for action in step.get('user_actions',[]):self.phase_map[action['id']]='action:'+action['id']
        for p in item.get('proposals',[]):
            for op in p['operations']:
                if op['op']=='create_task':self.tasks[op['task']['id']]=role(op['task'])
    def phase(self,p,default=None):
        if not self.source['input'].get('timeline'):return 'ingest'
        point=p.get('available_after',p.get('after',default))
        if point is None:
            ids=([p['event_id']] if 'event_id' in p else p.get('event_ids',[]))
            phases=[self.phase_map[v] for v in ids if v in self.phase_map]
            point=max(phases,key=lambda s:self.order.get(s,-1)) if phases else 'unspecified'
        if isinstance(point,str) and point.startswith('ingest:'):point=point[7:]
        return self.phase_map.get(point,point)
    def task_ref(self,id):return self.tasks.get(id,('unresolved_task',id))
    def facts(self):
        values=[]
        for p in self.item.get('proposals',[]):
            phase=self.phase(p)
            for op in p['operations']:
                if op['op']=='set_field':values.append((phase,op['purchase_id'],op['field'],op['value']))
                elif op['op']=='create_task':
                    t=op['task'];ref=role(t)
                    for key,value in [('exists',True),('owner',t.get('owner')),('due_date',t.get('due_date')),('status',t.get('status','open'))]:values.append((phase,ref,'task.'+key,value))
                elif op['op']=='update_task':
                    for k,v in op['fields'].items():values.append((phase,self.task_ref(op['task_id']),'task.'+k,v))
        return Counter(dump(v) for v in values)
    def issues(self):
        return Counter(dump((self.phase(i),i['code'],i.get('purchase_id'))) for i in self.item.get('expected_issues',[]))
    def value(self,x,key=None):
        if key in ['reason','explanation']:return None
        if isinstance(x,dict):
            if 'title' in x and 'purchase_id' in x:
                return {'task':role(x),**{k:self.value(v,k) for k,v in x.items() if k not in ['id','title','purchase_id']}}
            out={}
            for k,v in x.items():
                if k in ['reason','explanation']:continue
                if k=='fields':out.update(self.value(v));continue
                out['status' if k=='answer_status' else k]=self.value(v,k)
            return out
        if isinstance(x,list):return norm([self.value(v,key) for v in x])
        if isinstance(x,str) and x in self.tasks:return self.task_ref(x)
        if key=='status' and x in ['unknown','insufficient_evidence']:return 'unknown'
        return x
    def answers(self):
        values=[]
        def leaves(x,path):
            if isinstance(x,dict):
                for k,v in x.items():leaves(v,path+(k,))
            elif isinstance(x,list):
                # Lists are compared as complete collections; no arbitrary positional pairing.
                values.append((path,x))
            else:values.append((path,x))
        for a in self.item.get('answers',[]):leaves(self.value(a['answer']),(a['query_id'],))
        return Counter(dump(v) for v in values)
    def state(self,state):
        return dump({'purchases':state['purchases'],'tasks':[self.value(t) for t in state['tasks'].values()]})

def metric(a,b):
    common=sum((a&b).values());na=sum(a.values());nb=sum(b.values())
    return {'a':na,'b':nb,'matched':common,'f1':2*common/(na+nb) if na+nb else None}

def run(a_path,b_path):
    a=load(a_path);b=load(b_path)
    assert a['dataset_version']==b['dataset_version']
    if a.get('source_sha256') and b.get('source_sha256'):assert a['source_sha256']==b['source_sha256']
    recorded_hashes=a.get('source_sha256') or b.get('source_sha256') or {}
    by_a={i['case_id']:i for i in a['items']};by_b={i['case_id']:i for i in b['items']}
    sources={}
    for split in ('dev','test'):
        path=ROOT.parent/f'{split}.jsonl'
        if split in recorded_hashes:assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==recorded_hashes[split], ('Dataset changed since annotation',split)
        for line in path.read_text(encoding='utf-8').splitlines():
            c=json.loads(line)
            if c['id'] in by_a:sources[c['id']]={'case_id':c['id'],'split':split,'input':c['input']}
    assert len(by_a)==len(a['items']) and len(by_b)==len(b['items'])
    assert set(by_a)==set(by_b)==set(sources)
    totals={k:[Counter(),Counter()] for k in ['facts','issues','answers']};rows=[];state_equal=state_total=0
    for cid in by_a:
        ca=Canonical(by_a[cid],sources[cid]);cb=Canonical(by_b[cid],sources[cid]);row={'case_id':cid}
        for label in totals:
            va=getattr(ca,label)();vb=getattr(cb,label)();totals[label][0]+=va;totals[label][1]+=vb
            # Namespace case IDs before accumulating so equal facts in different cases cannot match.
            if label in ['issues','facts']:
                totals[label][0]-=va;totals[label][1]-=vb
                totals[label][0]+=Counter({cid+'|'+k:v for k,v in va.items()});totals[label][1]+=Counter({cid+'|'+k:v for k,v in vb.items()})
            row[label]={**metric(va,vb),'only_a':[json.loads(v) for v in (va-vb).elements()],'only_b':[json.loads(v) for v in (vb-va).elements()]}
        checks_a={c['after']:c['expected_state'] for c in by_a[cid].get('checks',[])};checks_b={c['after']:c['expected_state'] for c in by_b[cid].get('checks',[])}
        overlap=sorted(set(checks_a)&set(checks_b));same=[k for k in overlap if ca.state(checks_a[k])==cb.state(checks_b[k])]
        state_equal+=len(same);state_total+=len(overlap)
        row['states']={'shared':len(overlap),'equal':len(same),'different':[k for k in overlap if k not in same],'only_a':sorted(set(checks_a)-set(checks_b)),'only_b':sorted(set(checks_b)-set(checks_a))}
        final_a=by_a[cid].get('final_state',next(reversed(checks_a.values())) if checks_a else None)
        final_b=by_b[cid].get('final_state',next(reversed(checks_b.values())) if checks_b else None)
        row['final_state_equal']=ca.state(final_a)==cb.state(final_b) if final_a is not None and final_b is not None else None
        rows.append(row)
    rel=lambda p:Path(p).resolve().relative_to(ROOT.parent.parent).as_posix() if Path(p).resolve().is_relative_to(ROOT.parent.parent) else str(p)
    report={'dataset_version':a['dataset_version'],'files':{'a':{'path':rel(a_path),'sha256':hashlib.sha256(Path(a_path).read_bytes()).hexdigest()},'b':{'path':rel(b_path),'sha256':hashlib.sha256(Path(b_path).read_bytes()).hexdigest()}},
      'cases':len(rows),'source_hashes_recorded':{'a':bool(a.get('source_sha256')),'b':bool(b.get('source_sha256'))},'metrics':{label:metric(*pair) for label,pair in totals.items()},
      'macro_nonempty_facts_f1':sum(r['facts']['f1'] for r in rows if r['facts']['f1'] is not None)/sum(r['facts']['f1'] is not None for r in rows),
      'both_empty_fact_cases':sum(r['facts']['a']+r['facts']['b']==0 for r in rows),'shared_state_checkpoints':state_total,'equal_state_checkpoints':state_equal,'equal_final_states':sum(r['final_state_equal'] for r in rows),
      'limitations':['LLM2 участвовал в подготовке датасета; это не полностью независимая слепая оценка.','Имя файла human1 не подтверждает личность аннотатора. Его annotation_policy описывает модельную разметку.','Состояния выведены из операций и не являются независимыми оценками.','Смысл названий задач сопоставлен фиксированными правилами; остальные свободные формулировки требуют арбитража.','Пустые множества дают F1=null и не добавляют 100% к macro F1.'],
      'normalization':'Фиксированная до чтения human1: локальные task ID заменяются на purchase+intent+document, нормализуются регистр, ё/е, пробелы и десятичные числа; evidence, reason, explanation не входят в сравнение содержания. Названия задач сохраняются в исходных файлах.',
      'per_case':rows}
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--a',default=str(ROOT/'human1.json'));parser.add_argument('--b',default=str(ROOT/'human2.json'));parser.add_argument('--out',default=str(ROOT/'agreement.json'));args=parser.parse_args()
    report=run(args.a,args.b);Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ['cases','metrics','macro_nonempty_facts_f1','both_empty_fact_cases','shared_state_checkpoints','equal_state_checkpoints','equal_final_states']},ensure_ascii=False,indent=2))

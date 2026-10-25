import copy
import json
import sys
import tempfile
import unittest
import importlib.util
import os
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score


def options(**kwargs):
    defaults = dict(judge="off", model="test", base_url=None, api_key_env="OPENAI_API_KEY",
                    timeout=1, retries=1, max_judge_chars=120000)
    defaults.update(kwargs)
    return Namespace(**defaults)


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.cases, _ = score.load_jsonl(cls.root / "dev.jsonl")

    def case(self, cid="E01"):
        return copy.deepcopy(self.cases[cid])

    def run_case(self, ref, pred):
        counts, result = score.evaluate_case(ref, pred, score.Judge(options()))
        return counts.report(), result

    def test_identical_dev_and_test(self):
        for filename in ("dev.jsonl", "test.jsonl"):
            with self.subTest(filename=filename):
                judge = score.Judge(options())
                report = score.evaluate(self.root/filename, self.root/filename, judge)
                self.assertTrue(report["all_structured_cases_match"])
                self.assertEqual(judge.calls, [])
                for name, metric in report["metrics"].items():
                    if isinstance(metric,dict) and metric.get("denominator",0):
                        self.assertEqual(metric["value"],1, name)

    def test_wrong_value_is_fp_and_fn(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["proposals"][0]["operations"][0]["value"] = 19
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["counts"]["diff_fp"],1)
        self.assertEqual(report["counts"]["diff_fn"],1)

    def test_duplicate_is_not_lost(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        ops = pred["gold"]["proposals"][0]["operations"]
        ops.append(copy.deepcopy(ops[0]))
        report, result = self.run_case(ref,pred)
        self.assertEqual(report["counts"]["diff_fp"],1)
        self.assertEqual(report["diff_recall"]["value"],1)
        self.assertFalse(result["fully_matches_structured_gold"])

    def test_missing_case_not_perfect(self):
        ref = self.case()
        report, result = self.run_case(ref,None)
        self.assertEqual(report["diff_recall"]["value"],0)
        self.assertIsNone(report["diff_precision"]["value"])
        self.assertFalse(result["fully_matches_structured_gold"])

    def test_numeric_normalization_and_boolean(self):
        self.assertTrue(score.equal("12500.00",12500,"unit_price"))
        self.assertFalse(score.equal(None,0,"unit_price"))
        self.assertFalse(score.equal(False,0,"quantity"))
        self.assertFalse(score.equal("P-17","p-17","purchase_id"))

    def test_pydantic_rejects_boolean_numeric_and_invalid_date(self):
        for changes in ({"quantity":True},{"unit_price":"NaN"},{"delivery_date":"2026-02-30"}):
            with self.subTest(changes=changes):
                errors = score.validation_errors({"revision":1,"fields":changes},score.PurchaseCard)
                self.assertTrue(errors)
                json.dumps(errors)  # Diagnostics must not contain unserializable ValueError objects.

    def test_pydantic_rejects_unknown_operation_and_missing_evidence_source(self):
        for change in ("operation","evidence"):
            pred = self.case()
            op = pred["gold"]["proposals"][0]["operations"][0]
            if change == "operation":
                op["op"] = "delete_database"
            else:
                op["evidence"] = [{"quote":"20 насосов"}]
            with self.subTest(change=change):
                self.assertTrue(score.validation_errors(pred,score.DatasetCase))

    def test_pydantic_judge_verdict_does_not_coerce_strings(self):
        errors = score.validation_errors({"accepted":"true","reason":"x","source_ids":[]},score.JudgeVerdict)
        self.assertTrue(errors)

    def test_order_and_ids_do_not_change_diff_score(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["proposals"][0]["id"] = "arbitrary-proposal"
        pred["gold"]["proposals"][0]["operations"].reverse()
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["diff_precision"]["value"],1)

    def test_task_ids_mapped_in_snapshots(self):
        ref = self.case()
        pred = score.substitute_ids(copy.deepcopy(ref), {"T-1":"model-task"})
        # The input in this case contains no T-1; generated IDs may differ.
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["diff_recall"]["value"],1)
        self.assertEqual(report["crm_chains"]["value"],1)

    def test_alternative_task_title_pending_without_judge(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["proposals"][0]["operations"][-1]["task"]["title"] = "Попросить Дельту выставить счёт"
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["diff_recall"]["status"],"incomplete")
        self.assertIsNone(report["diff_recall"]["value"])

    def test_future_phase_not_matched(self):
        ref = self.case("D-MEM-09")
        pred = copy.deepcopy(ref)
        pred["gold"]["proposals"][0]["available_after"] = "D-MEM-09-step1"
        report, _ = self.run_case(ref,pred)
        self.assertGreater(report["counts"]["diff_fp"],0)
        self.assertLess(report["literal_citations"]["value"],1)

    def test_context_stops_before_future_and_actions(self):
        ref = self.case("D-MEM-09")
        ctx = score.context_at(ref,"D-MEM-09-step4")
        self.assertEqual(len(ctx["timeline"]),4)
        self.assertEqual(ctx["timeline"][-1]["user_actions"],[])

    def test_wrong_citation_does_not_change_fact_value_score(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["proposals"][0]["operations"][0]["evidence"][0]["quote"] = "несуществующая цитата"
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["diff_precision"]["value"],1)
        self.assertLess(report["literal_citations"]["value"],1)
        self.assertLess(report["supported_citations"]["value"],1)

    def test_extended_gold_citation_does_not_call_judge(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["proposals"][0]["operations"][0]["evidence"][0]["quote"] = ref["input"]["events"][0]["text"]
        class NeverJudge:
            def decide(self,*args):
                raise AssertionError("Extended annotated citation must be checked without LLM")
        counts,result = score.evaluate_case(ref,pred,NeverJudge())
        self.assertEqual(counts.report()["supported_citations"]["value"],1)

    def test_numeric_fact_mismatch_does_not_call_judge(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["proposals"][0]["operations"][0]["value"] = 19
        class NeverJudge:
            def decide(self,*args):
                raise AssertionError("Known numeric mismatch must be checked without LLM")
        counts,result = score.evaluate_case(ref,pred,NeverJudge())
        self.assertEqual(counts.report()["counts"]["diff_fp"],1)

    def test_crm_revision_is_checked(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["checks"][-1]["expected_state"]["purchases"]["P-17"]["revision"] = 9
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["crm_chains"]["value"],0)
        self.assertEqual(report["diff_precision"]["value"],1)

    def test_malformed_snapshot_is_schema_failure(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["checks"][0]["expected_state"]["purchases"] = []
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["case_schema_validity"]["value"],0)
        self.assertLess(report["format_validity"]["value"],1)

    def test_summary_extra_and_missing_items(self):
        ref = self.case("D-SUM-05")
        pred = copy.deepcopy(ref)
        a = pred["gold"]["answers"][0]["answer"]
        a["overdue_task_ids"] = ["invented-task"]
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["counts"]["summary_fp"],1)
        self.assertEqual(report["counts"]["summary_fn"],1)

    def test_vat_and_remainder_calculations(self):
        state = {"purchases":{"p":{"fields":{"quantity":10,"unit_price":"500.00",
            "total_amount":"6000.00","price_includes_vat":False,"vat_rate":20,
            "delivered_quantity":3,"remaining_quantity":7}}}}
        self.assertTrue(all(c["correct"] for c in score.calculations(state)))
        state["purchases"]["p"]["fields"]["remaining_quantity"] = 8
        self.assertFalse(all(c["correct"] for c in score.calculations(state)))

    def test_missing_conflict_and_extra_issue(self):
        ref = self.case("E15")
        pred = copy.deepcopy(ref)
        pred["gold"]["expected_issues"] = [{"code":"invented","purchase_id":"other"}]
        report, _ = self.run_case(ref,pred)
        self.assertEqual(report["conflict_recall"]["value"],0)
        self.assertEqual(report["counts"]["extra_issues"],1)

    def test_wrong_procurement(self):
        ref = self.case()
        pred = copy.deepcopy(ref)
        pred["gold"]["proposals"][0]["operations"][0]["purchase_id"] = "P-18"
        report, _ = self.run_case(ref,pred)
        self.assertLess(report["purchase_link_accuracy"]["value"],1)

    def test_duplicate_ids_and_nan_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.jsonl"
            path.write_text('{"id":"a"}\n{"id":"a"}\n{"id":"b","x":NaN}\n',encoding="utf-8")
            rows,errors = score.load_jsonl(path)
            self.assertEqual(len(rows),1)
            self.assertEqual(len(errors),2)

    def test_empty_predictions_reduce_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"empty.jsonl"
            path.write_text("",encoding="utf-8")
            report = score.evaluate(self.root/"dev.jsonl",path,score.Judge(options()))
            self.assertEqual(len(report["coverage"]["missing_cases"]),100)
            self.assertEqual(report["metrics"]["diff_recall"]["value"],0)

    def test_limit_selects_first_gold_cases_and_ignores_other_known_predictions(self):
        report = score.evaluate(self.root/"dev.jsonl",self.root/"dev.jsonl",score.Judge(options()),limit=10)
        self.assertTrue(report["all_structured_cases_match"])
        self.assertEqual(report["coverage"]["reference_cases"],10)
        self.assertEqual(report["coverage"]["provided_cases"],10)
        self.assertEqual(report["coverage"]["unknown_cases"],[])
        self.assertEqual(report["selection"]["ignored_prediction_cases"],90)
        self.assertEqual(report["selection"]["selected_ids"],[f"E{i:02}" for i in range(1,11)])

    def test_limit_rejects_zero_and_negative_values(self):
        for limit in (0,-1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                score.evaluate(self.root/"dev.jsonl",self.root/"dev.jsonl",score.Judge(options()),limit=limit)

    def test_limit_missing_selected_case_stays_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"partial.jsonl"
            path.write_text(json.dumps(self.case(),ensure_ascii=False)+"\n",encoding="utf-8")
            report = score.evaluate(self.root/"dev.jsonl",path,score.Judge(options()),limit=2)
            self.assertEqual(report["coverage"]["missing_cases"],["E02"])
            self.assertFalse(report["all_structured_cases_match"])

    def test_judge_validates_verdict_and_caches(self):
        judge = score.Judge(options(judge="auto"))
        calls = []
        class Model:
            def invoke(self,messages):
                calls.append(messages)
                return SimpleNamespace(content=json.dumps({"accepted":True,"reason":"same action","source_ids":["e1"]}),usage_metadata={"total_tokens":10})
        judge.model = Model()
        for _ in range(2):
            self.assertTrue(judge.decide("equivalent_task",self.case(),"ingest","a","b"))
        self.assertEqual(len(calls),1)
        self.assertEqual(len(judge.calls),1)

    def test_judge_failure_is_pending_not_success(self):
        judge = score.Judge(options(judge="auto"))
        class Model:
            def invoke(self,messages):
                return SimpleNamespace(content='{"accepted":true,"reason":"x","source_ids":["future"]}')
        judge.model = Model()
        self.assertIsNone(judge.decide("equivalent_task",self.case(),"ingest","a","b"))
        self.assertEqual(judge.calls[-1]["status"],"judge_error")

    @unittest.skipUnless(importlib.util.find_spec("langchain_openai"), "LangChain not installed")
    def test_real_langchain_with_local_openai_compatible_endpoint(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path,body))
                # First response exercises a retry of malformed judge JSON.
                content = "not-json" if len(requests)==1 else json.dumps({
                    "accepted":True,"reason":"same task","source_ids":["e1"]})
                response = json.dumps({"id":"local-test", "object":"chat.completion", "created":1,
                    "model":"judge-test", "choices":[{"index":0,"message":{"role":"assistant","content":content},
                        "finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ,{"NO_PROXY":"127.0.0.1,localhost","no_proxy":"127.0.0.1,localhost"}):
                judge = score.Judge(options(judge="auto",model="judge-test",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1"))
                result = judge.decide("equivalent_task",self.case(),"ingest", "Запросить счёт", "Попросить счёт")
            self.assertTrue(result, {"judge_calls":judge.calls,"requests":len(requests)})
            self.assertEqual(len(requests),2)
            self.assertEqual(requests[0][0],"/v1/chat/completions")
            self.assertEqual(requests[0][1]["model"],"judge-test")
            self.assertEqual(judge.calls[-1]["usage"]["total_tokens"],30)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

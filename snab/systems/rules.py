"""B0: правила на регулярных выражениях без LLM.

Достает номер закупки, деньги и валюту, количество, НДС, предоплату, даты и задачи
"Имя, сделай ... до <дата>". Условные фразы ("возможно", "если") изменений не дают.
Цитата - дословный найденный фрагмент. Вопросов закупщику B0 не задает.
"""

import re
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from decimal import Decimal

from snab.contracts import SCORER, EpisodeContext, StopReason, SystemOutput
from snab.crm import morning_summary
from snab.tools import PURCHASE_ID, ModelTools

MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}
RELATIVE_DAYS = {"завтра": 1, "послезавтра": 2}
CURRENCIES = (
    ("рубл", "RUB"),
    ("руб", "RUB"),
    ("₽", "RUB"),
    ("евро", "EUR"),
    ("eur", "EUR"),
    ("€", "EUR"),
    ("долл", "USD"),
    ("usd", "USD"),
    ("$", "USD"),
)
SCALES = (("млн", 1_000_000), ("миллион", 1_000_000), ("тыс", 1_000))
IMPERATIVES = {
    "запроси": "запросить",
    "пересобери": "пересобрать",
    "подтверди": "подтвердить",
    "проверь": "проверить",
    "получи": "получить",
    "отправь": "отправить",
    "уточни": "уточнить",
    "закажи": "заказать",
    "оформи": "оформить",
    "подготовь": "подготовить",
    "прими": "принять",
    "сверь": "сверить",
    "найди": "найти",
    "согласуй": "согласовать",
    "пришли": "прислать",
    "передай": "передать",
    "собери": "собрать",
    "напомни": "напомнить",
    "оплати": "оплатить",
    "забронируй": "забронировать",
}

DATE = (
    r"(?:(?P<iso>\d{4}-\d{2}-\d{2})|(?P<dmy>\d{1,2}\.\d{1,2}\.\d{4})"
    r"|(?P<day>\d{1,2})\s+(?P<month>" + "|".join(MONTHS) + r")(?:\s+(?P<year>\d{4})(?:\s+года)?)?"
    r"|\b(?P<relative>послезавтра|завтра)\b)"
)
AMOUNT = r"(?P<amount>\d{1,3}(?:[  ]\d{3})+(?!\d)|\d+(?:[.,]\d+)?)"
SCALE = r"(?:\s*(?P<scale>млн|миллион\w*|тыс\.?|тысяч\w*))?"
CURRENCY = r"\s*(?P<currency>рубл\w*|руб\.?|₽|евро|EUR|€|долл\w*|USD|\$)"
MONEY = AMOUNT + SCALE + CURRENCY

DATE_RE = re.compile(DATE)
CLAUSE_RE = re.compile(r"(?:[^.!?;\n]|(?<=\d)\.(?=\d))+")
CONDITIONAL_RE = re.compile(
    r"\b(?:возможно|если|может|можем|могу|могут|предлага\w*|либо|наверное|вероятно|пока не|не берём)\b", re.IGNORECASE
)
TASK_RE = re.compile(r"(?:^|(?<=[.;:!?]\s))(?P<owner>[А-ЯЁ][а-яё]+),\s+(?P<body>[^.;!?]+)")
DUE_RE = re.compile(r"\s*(?:\b(?:до|к)\s+)?" + DATE)
PER_UNIT = r"(?P<per>\s+за\s+(?:штуку|шт\.?|единицу|одну))"
UNIT_PRICE_PATTERNS = (
    re.compile(r"(?:\bпо|\bцен[аеуы])\s+" + AMOUNT + SCALE + r"(?:" + CURRENCY + r")?" + PER_UNIT + "?", re.IGNORECASE),
    re.compile(MONEY + PER_UNIT, re.IGNORECASE),
)
QUANTITY_RE = re.compile(r"\b(?P<quantity>\d+)\s+(?:штук\w*|шт\b\.?)", re.IGNORECASE)
RECEIPT_RE = re.compile(r"\b(?:прин[яи]\w*|получ\w*|приех\w*|пришл\w*)", re.IGNORECASE)
TOTAL_RE = re.compile(
    r"(?:итого|всего|общ\w+ (?:цена|сумма|стоимость)|сумма заказа|стоимость\s*[—:-])[^.;]{0,40}?" + MONEY, re.IGNORECASE
)
BUDGET_RE = re.compile(r"(?:бюджет|лимит)\w*[^.;]{0,50}?" + MONEY, re.IGNORECASE)
DELIVERY_COST_RE = re.compile(r"доставк\w*[^.;]{0,30}?" + MONEY, re.IGNORECASE)
PREPAYMENT_RE = re.compile(
    r"\b(?P<percent>\d+)\s*(?:процент\w*|%)\s*(?:до отгрузки|аванс\w*|предоплат\w*|вперёд)"
    r"|(?P<full>вс[юё]\s+сумм\w*\s+(?:платим\s+)?(?:вперёд|до отгрузки)"
    r"|платим вс[её] до отгрузки|полн\w+\s+предоплат\w*)",
    re.IGNORECASE,
)
VAT_RE = re.compile(r"(?:(?P<prefix>\b[сС]|\bбез)\s+)?НДС\s*(?P<rate>\d+)\s*%(?P<included>\s*включ\w*)?")
DELIVERY_CUE_RE = re.compile(
    r"постав\w*|привез\w*|привоз\w*|везём|везем|доставк\w*|срок\w*|ждёт|ждём|ждем|приедет|машин\w*|"
    r"перенос\w*|перенесли|сдвинули|согласова\w*|договорил\w*|подтвер\w*|фиксируем|(?P<shipment>отгруз\w*)",
    re.IGNORECASE,
)
REPLACED_DATE_RE = re.compile(r"вместо\s*$", re.IGNORECASE)
SUMMARY_QUERY_RE = re.compile(r"сегодня.*ближайш", re.IGNORECASE)
CONDITIONS_QUERY_RE = re.compile(r"действующие условия.*задач", re.IGNORECASE)
QUERY_FIELDS = (
    (re.compile(r"цен", re.IGNORECASE), ("unit_price", "currency")),
    (re.compile(r"количеств|сколько заказали", re.IGNORECASE), ("quantity",)),
    (re.compile(r"поступлен|сколько приняли", re.IGNORECASE), ("delivered_quantity",)),
    (re.compile(r"ещё ждём|осталось", re.IGNORECASE), ("remaining_quantity",)),
    (re.compile(r"\bсрок", re.IGNORECASE), ("delivery_date",)),
    (re.compile(r"бюджет|лимит", re.IGNORECASE), ("budget", "currency")),
)
UNKNOWN_QUERY_FIELDS = (
    (re.compile(r"\bИНН\b"), "inn"),
    (re.compile(r"банковск\w* сч[её]т", re.IGNORECASE), "bank_account"),
    (re.compile(r"оплат", re.IGNORECASE), "paid"),
)
TOOL_ISSUES = {"timeout": "tool_timeout", "budget_exhausted": "step_budget_exhausted"}
TOOL_STOPS = {"timeout": StopReason.TIMEOUT, "budget_exhausted": StopReason.BUDGET_EXHAUSTED}


def parse_amount(match: re.Match) -> Decimal:
    """'4 млн' -> 4000000, '127,50' -> 127.50."""
    amount = Decimal(re.sub(r"\s", "", match["amount"]).replace(",", "."))
    scale = (match["scale"] or "").lower()
    return amount * next((factor for prefix, factor in SCALES if scale.startswith(prefix)), 1)


def money_text(amount: Decimal) -> str:
    """Деньги как десятичная строка с копейками."""
    return str(amount.quantize(Decimal("0.01")))


def currency_code(word: str | None) -> str | None:
    """'рублей' -> 'RUB'."""
    word = (word or "").lower()
    return next((code for prefix, code in CURRENCIES if word.startswith(prefix)), None) if word else None


def resolve_date(match: re.Match, occurred_at: str | None) -> date | None:
    """Абсолютная или относительная дата; без времени сообщения год и 'завтра' неизвестны."""
    if match["iso"]:
        return date.fromisoformat(match["iso"])
    if match["dmy"]:
        day, month, year = map(int, match["dmy"].split("."))
        return date(year, month, day)
    if occurred_at is None and (match["relative"] or not match["year"]):
        return None
    if match["relative"]:
        return datetime.fromisoformat(occurred_at).date() + timedelta(days=RELATIVE_DAYS[match["relative"]])
    year = int(match["year"]) if match["year"] else datetime.fromisoformat(occurred_at).year
    return date(year, MONTHS[match["month"]], int(match["day"]))


def find_tasks(text: str) -> list[dict]:
    """'Анна, запроси счёт до 6 октября' -> владелец, действие, срок и точный фрагмент."""
    tasks = []
    for match in TASK_RE.finditer(text):
        body = match["body"]
        verb = next((word for word in re.findall(r"[а-яё]+", body) if word in IMPERATIVES), None)
        if verb is None:
            continue
        due = None
        for candidate in DUE_RE.finditer(body):
            if candidate["relative"] or candidate.group(0).strip().startswith(("до", "к ")):
                due = candidate
        action = body[: due.start()] + body[due.end() :] if due else body
        rest = " ".join(re.sub(rf"\b{verb}\b", "", action).split())
        title = IMPERATIVES[verb].capitalize() + (" " + rest if rest else "")
        end = match.start("body") + (due.end() if due else len(body.rstrip()))
        tasks.append({"owner": match["owner"], "title": title, "due": due, "start": match.start(), "end": end})
    return tasks


def iter_clauses(text: str) -> Iterator[re.Match]:
    """Фразы между знаками конца предложения; точка внутри даты фразу не рвет."""
    for clause in CLAUSE_RE.finditer(text):
        if clause.group(0).strip():
            yield clause


def purchase_at(text: str, position: int, fallback: str | None) -> str | None:
    """Последний номер закупки до позиции, иначе первый в тексте, иначе найденный поиском."""
    before = [match.group(0) for match in PURCHASE_ID.finditer(text) if match.start() < position]
    if before:
        return before[-1]
    anywhere = PURCHASE_ID.findall(text)
    return anywhere[0] if anywhere else fallback


def date_field(clause: str, date_match: re.Match) -> tuple[str, re.Match] | None:
    """Поле даты по ближайшему слову-признаку; дата после 'вместо' не новая."""
    if REPLACED_DATE_RE.search(clause[: date_match.start()]):
        return None
    cues = list(DELIVERY_CUE_RE.finditer(clause))
    if not cues:
        return None
    cue = min(cues, key=lambda item: min(abs(item.start() - date_match.end()), abs(date_match.start() - item.end())))
    return ("shipment_date" if cue["shipment"] else "delivery_date"), cue


def extract_clause_facts(clause: str, occurred_at: str | None) -> tuple[list[tuple], list[str]]:
    """Факты одной фразы как (позиция, поле, значение, начало, конец) и коды проблем."""
    facts, problems = [], []
    for match in (match for pattern in UNIT_PRICE_PATTERNS for match in pattern.finditer(clause)):
        if not (match["currency"] or match["per"]):
            continue
        facts.append((match.start(), "unit_price", money_text(parse_amount(match)), match.start(), match.end()))
        code = currency_code(match["currency"])
        if code:
            facts.append((match.start(), "currency", code, match.start(), match.end()))
        else:
            problems.append("missing_currency")
    if not RECEIPT_RE.search(clause):
        for match in QUANTITY_RE.finditer(clause):
            facts.append((match.start(), "quantity", int(match["quantity"]), match.start(), match.end()))
            facts.append((match.start(), "unit", "шт", match.start(), match.end()))
    for pattern, field in ((TOTAL_RE, "total_amount"), (BUDGET_RE, "budget"), (DELIVERY_COST_RE, "delivery_cost")):
        for match in pattern.finditer(clause):
            facts.append((match.start(), field, money_text(parse_amount(match)), match.start(), match.end()))
    for match in PREPAYMENT_RE.finditer(clause):
        percent = 100 if match["full"] else int(match["percent"])
        facts.append((match.start(), "prepayment_percent", percent, match.start(), match.end()))
    for match in VAT_RE.finditer(clause):
        facts.append((match.start(), "vat_rate", int(match["rate"]), match.start(), match.end()))
        prefix = (match["prefix"] or "").lower()
        if match["included"] or prefix:
            facts.append((match.start(), "price_includes_vat", prefix != "без", match.start(), match.end()))
    for match in DATE_RE.finditer(clause):
        found = date_field(clause, match)
        if found is None:
            continue
        resolved = resolve_date(match, occurred_at)
        if resolved is None:
            problems.append("missing_source_time")
            continue
        field, cue = found
        start, end = min(match.start(), cue.start()), max(match.end(), cue.end())
        facts.append((match.start(), field, resolved.isoformat(), start, end))
    return facts, problems


def link_event(event: dict, tools: ModelTools, issues: list[dict], episode_id: str) -> tuple[str | None, str | None]:
    """Закупка для сообщения без номера через find_purchase; ошибка поиска становится проблемой."""
    result = tools.find_purchase(event["text"])
    if result.ok:
        return result.data["candidates"][0]["purchase_id"], None
    error = result.error
    if error.code in ("ambiguous", "not_found"):
        code = "ambiguous_purchase" if error.code == "ambiguous" else "purchase_not_found"
        issues.append(
            {
                "code": code,
                "after": episode_id,
                "event_id": event["id"],
                "candidates": error.details.get("candidates", []),
            }
        )
        return None, None
    issues.append({"code": TOOL_ISSUES.get(error.code, "tool_error"), "after": episode_id, "event_id": event["id"]})
    return None, error.code


def event_operations(event: dict, context: EpisodeContext, tools: ModelTools, issues: list[dict]) -> tuple:
    """Операции одного сообщения по закупкам и код ошибки инструмента, если был."""
    text, occurred_at = event["text"], event.get("occurred_at")
    tasks = find_tasks(text)
    masked = text
    for task in tasks:
        masked = masked[: task["start"]] + " " * (task["end"] - task["start"]) + masked[task["end"] :]
    facts, problems = [], []
    for clause in iter_clauses(masked):
        if CONDITIONAL_RE.search(clause.group(0)):
            continue
        clause_facts, clause_problems = extract_clause_facts(clause.group(0), occurred_at)
        problems.extend(clause_problems)
        offset = clause.start()
        facts.extend(
            (pos + offset, field, value, start + offset, end + offset) for pos, field, value, start, end in clause_facts
        )
    tool_error = None
    linked = None
    if (facts or tasks or problems) and not PURCHASE_ID.search(text):
        linked, tool_error = link_event(event, tools, issues, context.episode_id)
    operations = []
    for position, field, value, start, end in facts:
        purchase_id = purchase_at(text, position, linked)
        if purchase_id is not None:
            evidence = [{"event_id": event["id"], "quote": text[start:end].strip()}]
            operations.append(
                {"op": "set_field", "purchase_id": purchase_id, "field": field, "value": value, "evidence": evidence}
            )
    for task in tasks:
        purchase_id = purchase_at(text, task["start"], linked)
        if purchase_id is None:
            continue
        due = resolve_date(task["due"], occurred_at) if task["due"] else None
        operations.append(
            {
                "op": "create_task",
                "task": {
                    "id": "",
                    "purchase_id": purchase_id,
                    "title": task["title"],
                    "owner": task["owner"],
                    "due_date": due and due.isoformat(),
                    "status": "open",
                },
                "evidence": [{"event_id": event["id"], "quote": text[task["start"] : task["end"]]}],
            }
        )
    for code in dict.fromkeys(problems):
        purchase_id = purchase_at(text, len(text), linked)
        issues.append({"code": code, "after": context.episode_id, "event_id": event["id"], "purchase_id": purchase_id})
    return operations, tool_error


def operation_target(operation: dict, state: dict) -> tuple:
    """(закупка, что меняется): поле карточки, новая задача по названию или существующая задача."""
    if operation["op"] == "set_field":
        return operation["purchase_id"], operation["field"]
    if operation["op"] == "create_task":
        return operation["task"]["purchase_id"], "task", operation["task"]["title"]
    task = state["tasks"].get(operation["task_id"], {})
    return task.get("purchase_id"), "update_task", operation["task_id"]


def build_proposals(operations: list[dict], context: EpisodeContext, system_name: str) -> list[dict]:
    """Одно предложение на закупку: последнее значение поля, без повторов текущей карточки."""
    purchases = context.crm_state["purchases"]
    latest = {operation_target(operation, context.crm_state): operation for operation in operations}
    by_purchase = {}
    for key, operation in latest.items():
        purchase_id = key[0]
        if purchase_id not in purchases:
            continue
        fields = purchases[purchase_id]["fields"]
        # Сравнение оценщика: числа по значению, и неразборчивый ответ модели не роняет сборку
        if operation["op"] == "set_field" and SCORER.equal(
            fields.get(operation["field"]), operation["value"], operation["field"]
        ):
            continue
        by_purchase.setdefault(purchase_id, []).append(operation)
    proposals = []
    for purchase_id, purchase_operations in sorted(by_purchase.items()):
        has_quantity = any(op["op"] == "set_field" and op["field"] == "quantity" for op in purchase_operations)
        # Единица измерения - уточнение количества, отдельно от него не предлагается
        kept_operations = [
            op for op in purchase_operations if has_quantity or op["op"] != "set_field" or op["field"] != "unit"
        ]
        if not kept_operations:
            continue
        for number, operation in enumerate(op for op in kept_operations if op["op"] == "create_task"):
            operation["task"]["id"] = f"{context.episode_id}-{purchase_id}-T{number + 1}"
        proposals.append(
            {
                "id": f"{context.episode_id}:{system_name}:{purchase_id}",
                "available_after": context.episode_id,
                "based_on": {purchase_id: purchases[purchase_id]["revision"]},
                "operations": kept_operations,
            }
        )
    return proposals


def open_tasks(tasks: list[dict], purchase_id: str) -> list[dict]:
    """Открытые задачи закупки."""
    return [task for task in tasks if task["purchase_id"] == purchase_id and task["status"] == "open"]


def answer_fields(query: str, card: dict, tasks: list[dict], purchase_id: str) -> dict | None:
    """Ответ по полям карточки, если вопрос сводится к ним."""
    fields = card["fields"]
    if CONDITIONS_QUERY_RE.search(query):
        return {"fields": fields, "tasks": open_tasks(tasks, purchase_id)}
    unknown = [name for pattern, name in UNKNOWN_QUERY_FIELDS if pattern.search(query) and name not in fields]
    if unknown:
        return {**dict.fromkeys(unknown), "status": "unknown"}
    names = [name for pattern, group in QUERY_FIELDS if pattern.search(query) for name in group]
    if not names:
        return None
    answer = {name: fields.get(name) for name in dict.fromkeys(names)}
    if "remaining_quantity" in answer and answer["remaining_quantity"] is None:
        quantity, delivered = fields.get("quantity"), fields.get("delivered_quantity")
        answer["remaining_quantity"] = None if quantity is None or delivered is None else quantity - delivered
    return answer


def code_answer(text: str, state: dict, now: str | None) -> dict | None:
    """Вопрос -> утренняя сводка или поля названной закупки из CRM; None, если вопрос не сводится к ним."""
    if SUMMARY_QUERY_RE.search(text) and now:
        return morning_summary(state, now)
    named = [pid for pid in PURCHASE_ID.findall(text) if pid in state["purchases"]]
    if not named:
        return None
    return answer_fields(text, state["purchases"][named[0]], list(state["tasks"].values()), named[0])


class RuleSystem:
    """B0 без модели; инструмент вызывает только для сообщений без номера закупки."""

    name = "rules"

    def describe(self) -> dict:
        """Параметры для meta.json: модели и промптов у B0 нет."""
        return {
            "system": self.name,
            "model": None,
            "temperature": None,
            "seed": None,
            "prompt_hashes": {},
            "prompts": {},
        }

    def process_episode(self, context: EpisodeContext, tools: ModelTools) -> SystemOutput:
        """Сообщения эпизода -> предложения и проблемы."""
        issues, operations, tool_errors = [], [], []
        ordered = sorted(
            context.events, key=lambda event: (event.get("occurred_at") is None, event.get("occurred_at") or "")
        )
        for event in ordered:
            event_ops, tool_error = event_operations(event, context, tools, issues)
            operations.extend(event_ops)
            if tool_error:
                tool_errors.append(tool_error)
        stop_reason = TOOL_STOPS.get(tool_errors[0], StopReason.TOOL_ERROR) if tool_errors else StopReason.DONE
        return SystemOutput(
            proposals=build_proposals(operations, context, self.name), issues=issues, stop_reason=stop_reason
        )

    def answer_query(self, context: EpisodeContext, query: dict, tools: ModelTools) -> SystemOutput:
        """Вопрос пользователя -> ответ по текущей CRM, если он очевиден."""
        answer = code_answer(query["text"], context.crm_state, context.now)
        answers = [] if answer is None else [{"query_id": query["id"], "answer": answer, "evidence": []}]
        return SystemOutput(answers=answers)

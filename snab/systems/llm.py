"""LLM-системы на нативном Ollama: llm_single (B1) и agent (S1) через один путь кода.

Обе системы получают кусок (эпизод раннера), подтвержденные карточки как базу только для
чтения, черновик встречи из прошлых кусков того же источника и открытые вопросы, а модель
возвращает кандидатов изменений по JSON-схеме. S1 отличается одной ограниченной развилкой
из раздела 5 design doc: если в куске нет номера закупки или он ссылается на прежние
условия, модель до итогового диффа может вызвать до двух инструментов только для чтения.

Кандидатов сводит код (snab.merge): проверка цитат и полномочий, сведение по ключу
(purchase_id, field), предложения только после завершения источника. Каждый вызов модели
списывается с бюджета шагов харнесса, а остановка всегда получает явный stop_reason.
"""

import copy
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from math import ceil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from snab.contracts import EpisodeContext, StopReason, SystemOutput, ToolResult, TraceStep
from snab.merge import (
    Candidate,
    Merge,
    event_order,
    gate_authority,
    merge_candidates,
    same_value,
    to_decimal,
    to_operations,
)
from snab.systems.rules import (
    AMOUNT,
    CONDITIONS_QUERY_RE,
    CURRENCY,
    DATE_RE,
    QUANTITY_RE,
    SCALE,
    SUMMARY_QUERY_RE,
    build_proposals,
    code_answer,
    currency_code,
    open_tasks,
    parse_amount,
    resolve_date,
)
from snab.tools import PURCHASE_ID, ModelTools, same_stem, words

# Версия кода системы в описании прогона, внутри ступеней B1 и S1 раздела 8. В s1.1 код, а не модель,
# считает сводку, ответы по полям и задачам, статус, валюту, единицу и вопрос о времени сообщения
SYSTEM_VERSION = "s1.1"
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
TEMPERATURE = 0
SEED = 42
NUM_CTX = 8192
NS_PER_MS = 1_000_000
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROMPTS = {path.stem: path.read_text(encoding="utf-8").strip() for path in sorted(PROMPT_DIR.glob("*.md"))}
# Остаток окна на ответ: итоговый дифф на куске - до ~800 токенов (раздел 6 design doc)
REPLY_TOKENS = 1024
PROMPT_TOKEN_BUDGET = NUM_CTX - REPLY_TOKENS
# Оценка до вызова, с запасом для кириллицы в JSON; точное число пишет Ollama в prompt_eval_count
CHARS_PER_TOKEN = 2.5
MAX_OPEN_QUESTIONS = 5
PRIOR_REFERENCE = re.compile(
    r"как договаривались|прежн\w*|стар\w* письм\w*|в прошлый раз|ранее\b|\bтоже\b|по эт\w+ \w+", re.IGNORECASE
)
EVENT_KEYS = ("id", "author", "speaker_id", "occurred_at", "media_time", "text")
# Эти вопросы закупщику ставит код, модельные дубли отбрасываются
CODE_ISSUES = frozenset(
    {"unknown_speaker_authority", "conflicting_unconfirmed_claim", "tool_timeout", "step_budget_exhausted"}
)
FAILURE_ISSUES = {StopReason.BUDGET_EXHAUSTED: "step_budget_exhausted", StopReason.TIMEOUT: "model_timeout"}
TOOL_FAILURE_ISSUES = {"timeout": "tool_timeout", "budget_exhausted": "step_budget_exhausted"}
TOOL_STOP_REASONS = {"timeout": StopReason.TOOL_ERROR, "budget_exhausted": StopReason.BUDGET_EXHAUSTED}
LINKING_ISSUES = frozenset({"ambiguous_purchase", "purchase_not_found"})
# Вопросы, на которые может ответить следующий кусок: их текст идет в open_questions
OPEN_QUESTION_CODES = frozenset({"tool_timeout", "step_budget_exhausted", "ambiguous_purchase", "purchase_not_found"})
MONEY_FIELDS = frozenset({"unit_price", "total_amount", "budget", "delivery_cost"})
DATE_FIELDS = frozenset({"delivery_date", "shipment_date"})
NUMBER_RE = re.compile(AMOUNT + SCALE, re.IGNORECASE)
CURRENCY_RE = re.compile(CURRENCY, re.IGNORECASE)
# Вопрос о поручениях: answer.md разрешает модели только поля карточки, задачи отдает код
TASKS_QUERY_RE = re.compile(r"\bдолж(?:ен|на|ны)\b|поруч|\bзадач|\bкто отвечает", re.IGNORECASE)
# Кто, почему и откуда значение, что было раньше и в каком письме: полей карточки для ответа
# мало, отвечает модель
MODEL_QUERY_RE = re.compile(
    r"\bкто\b|почему|откуда|основан|доказательств|изменил|в начале|ранн\w* истори|письм", re.IGNORECASE
)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "find_purchase",
            "description": "Поставщик, товар, объект или номер -> кандидаты закупок с основаниями",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "товар, поставщик или объект"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_purchase",
            "description": "ID закупки -> текущая карточка, задачи и ревизия",
            "parameters": {
                "type": "object",
                "properties": {"purchase_id": {"type": "string"}},
                "required": ["purchase_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_evidence",
            "description": "ID закупки и период -> прежние сообщения и куски встреч с номером закупки",
            "parameters": {
                "type": "object",
                "properties": {
                    "purchase_id": {"type": "string"},
                    "period": {
                        "type": "object",
                        "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                    },
                },
                "required": ["purchase_id"],
            },
        },
    },
]
TOOL_ARGUMENTS = {"find_purchase": ("query",), "read_purchase": ("purchase_id",), "read_evidence": ("purchase_id",)}


class Quote(BaseModel):
    """Дословная цитата из сообщения."""

    event_id: str
    quote: str


# Поля карточки, которые модель достает из текста; статус карточки и сумму считает код
CardField = Literal[
    "unit_price",
    "total_amount",
    "budget",
    "delivery_cost",
    "quantity",
    "delivered_quantity",
    "prepayment_percent",
    "vat_rate",
    "price_includes_vat",
    "currency",
    "unit",
    "delivery_date",
    "shipment_date",
    "supplier",
]


class SetField(BaseModel):
    """Кандидат нового значения поля со статусом договоренности."""

    action: Literal["set_field"]
    purchase_id: str
    field: CardField
    value: str | int | float | bool | None
    status: Literal["agreed", "offer", "conditional"]
    evidence: list[Quote]


class CreateTask(BaseModel):
    """Новое поручение по закупке."""

    action: Literal["create_task"]
    purchase_id: str
    title: str
    owner: str | None
    due_date: str | None
    evidence: list[Quote]


class TaskChanges(BaseModel):
    """Изменяемые поля задачи."""

    title: str | None = None
    owner: str | None = None
    due_date: str | None = None
    status: Literal["open", "done", "cancelled"] | None = None


class UpdateTask(BaseModel):
    """Изменение существующей задачи."""

    action: Literal["update_task"]
    task_id: str
    fields: TaskChanges
    evidence: list[Quote]


class ExtractedIssue(BaseModel):
    """Вопрос закупщику или найденная проблема."""

    code: str
    event_id: str | None = None
    purchase_id: str | None = None


class Extraction(BaseModel):
    """Ответ модели на кусок.

    Ollama выводит поля схемы по алфавиту: с именем operations модель сначала писала issues
    и объявляла согласованный срок противоречием, поэтому список операций называется changes.
    По той же причине вид операции называется action: с ключом op первым шел evidence,
    после него грамматика не пускала к create_task, и задачи уходили в set_field.
    """

    changes: list[SetField | CreateTask | UpdateTask]
    issues: list[ExtractedIssue]


class QueryAnswer(BaseModel):
    """Ответ модели на вопрос закупщика."""

    answer: dict[str, Any]


class OllamaClient:
    """HTTP-клиент нативного Ollama на stdlib."""

    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self.base_url = base_url

    def chat(self, payload: dict, timeout_s: float) -> dict:
        """POST /api/chat без стриминга; зависший вызов дает TimeoutError."""
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read())
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError(f"ollama did not answer in {timeout_s} s") from error
            raise

    def model_digest(self, model: str) -> str | None:
        """Digest локальной модели из /api/tags или None, если ее нет."""
        with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=10) as response:
            models = json.loads(response.read())["models"]
        return next((item["digest"] for item in models if item["name"] == model), None)


@dataclass
class Session:
    """Рабочее состояние одной истории между кусками; CRM остается источником истины.

    drafts - кандидаты незавершенных источников, draft_issues - их противоречия до конца
    источника, committed - последние предложенные значения
    по ключу, open_questions - нерешенные вопросы прошлых кусков.
    """

    seen: list[tuple[str, str]] = field(default_factory=list)
    drafts: dict[str, list[Candidate]] = field(default_factory=dict)
    held_tasks: dict[str, list[dict]] = field(default_factory=dict)
    draft_issues: dict[str, list[dict]] = field(default_factory=dict)
    committed: dict[tuple[str, str], Candidate] = field(default_factory=dict)
    open_questions: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Branch:
    """Итог развилки с инструментами: сообщения для итогового вызова, шаги и сбой, если был."""

    messages: list[dict]
    steps: list[TraceStep]
    stop_reason: StopReason | None = None
    failed_tool: dict | None = None


def sha256_text(text: str) -> str:
    """sha256 строки в UTF-8."""
    return hashlib.sha256(text.encode()).hexdigest()


def estimate_tokens(text: str) -> int:
    """Оценка числа токенов до вызова для бюджета контекста."""
    return ceil(len(text) / CHARS_PER_TOKEN)


def source_of(event: dict) -> str:
    """Источник события: кусок встречи принадлежит записи, сообщение - самому себе."""
    return event.get("source_id") or event["id"]


def compact_event(event: dict) -> dict:
    """Событие для модели без служебных полей и пустых значений."""
    return {key: event[key] for key in EVENT_KEYS if event.get(key) is not None}


def needs_search(events: list[dict]) -> bool:
    """Развилка S1: в куске нет номера закупки или он ссылается на прежние условия."""
    return any(not PURCHASE_ID.search(event["text"]) or PRIOR_REFERENCE.search(event["text"]) for event in events)


def relevance(fields: dict, text: str) -> int:
    """Число слов карточки (поставщик, товар), совпавших со словами куска."""
    card_words = {word for value in fields.values() if isinstance(value, str) for word in words(value)}
    chunk_words = words(text)
    return sum(any(same_stem(card_word, word) for word in chunk_words) for card_word in card_words)


def trim_request(request: dict, fixed_tokens: int, targets: set[str], text: str) -> tuple[dict, dict]:
    """Запрос -> запрос в бюджете контекста и статистика для трейса.

    Порядок сокращения: сначала старые цитаты (открытые вопросы, затем черновик, от старых
    к новым), потом карточки, не упомянутые в куске: их можно получить через read_purchase.
    Карточки из куска, черновика и открытых вопросов не убираются никогда.
    """
    request = copy.deepcopy(request)
    quotes = [(item, "text") for item in request.get("open_questions", [])]
    quotes += [(item, "quote") for item in request.get("draft", [])]
    removable = sorted(
        (pid for pid in request["cards"] if pid not in targets),
        key=lambda pid: (relevance(request["cards"][pid], text), pid),
    )
    dropped_quotes, omitted = 0, []

    def size() -> int:
        """Оценка всего входа вызова."""
        return fixed_tokens + estimate_tokens(json.dumps(request, ensure_ascii=False))

    while size() > PROMPT_TOKEN_BUDGET:
        if dropped_quotes < len(quotes):
            item, key = quotes[dropped_quotes]
            item[key] = None
            dropped_quotes += 1
        elif removable:
            purchase_id = removable.pop(0)
            del request["cards"][purchase_id]
            request["tasks"] = [task for task in request["tasks"] if task["purchase_id"] != purchase_id]
            omitted.append(purchase_id)
        else:
            break
    request["omitted_purchase_ids"] = sorted([*request.get("omitted_purchase_ids", []), *omitted])
    stats = {
        "estimated_tokens": size(),
        "budget_tokens": PROMPT_TOKEN_BUDGET,
        "cards_shown": sorted(request["cards"]),
        "cards_omitted": request["omitted_purchase_ids"],
        "quotes_dropped": dropped_quotes,
        "over_budget": size() > PROMPT_TOKEN_BUDGET,
    }
    return request, stats


def searched_stats(stats: dict, messages: list[dict]) -> dict:
    """Статистика контекста после развилки: оценка всех сообщений итогового вызова со схемами инструментов.

    Аргументы tool_calls короткие и в оценку не входят.
    """
    contents = "".join(message["content"] for message in messages)
    estimated = estimate_tokens(contents + json.dumps(TOOL_SCHEMAS, ensure_ascii=False))
    return {**stats, "estimated_tokens": estimated, "over_budget": estimated > PROMPT_TOKEN_BUDGET}


def draft_entry(candidate: Candidate) -> dict:
    """Кандидат черновика для модели: значение, статус и последняя цитата."""
    quote = candidate.evidence[-1]
    return {
        "purchase_id": candidate.purchase_id,
        "field": candidate.field,
        "value": candidate.value,
        "status": candidate.status,
        "event_id": quote["event_id"],
        "quote": quote["quote"],
    }


def mentioned_purchases(texts: list[str], cards: dict) -> set[str]:
    """Номера закупок из текстов, которые есть в CRM."""
    return {pid for text in texts for pid in PURCHASE_ID.findall(text or "") if pid in cards}


def episode_request(context: EpisodeContext, session: Session, fixed_tokens: int) -> tuple[dict, dict]:
    """Вход куска: кусок, подтвержденные карточки, черновик того же источника и открытые вопросы."""
    state = context.crm_state
    sources = {source_of(event) for event in context.events}
    draft = [draft_entry(candidate) for source in sorted(sources) for candidate in session.drafts.get(source, [])]
    questions = [dict(question) for question in session.open_questions]
    request = {
        "events": [compact_event(event) for event in context.events],
        "cards": {pid: card["fields"] for pid, card in state["purchases"].items()},
        "omitted_purchase_ids": [],
        "tasks": list(state["tasks"].values()),
        "draft": draft,
        "open_questions": questions,
        "authority_registry": context.authority_registry,
    }
    text = " ".join(event["text"] for event in context.events)
    targets = mentioned_purchases([text], state["purchases"])
    targets |= {item["purchase_id"] for item in [*draft, *questions] if item.get("purchase_id")}
    return trim_request(request, fixed_tokens, targets, text)


def query_request(context: EpisodeContext, query: dict, fixed_tokens: int) -> tuple[dict, dict]:
    """Вход вопроса закупщика: вопрос, карточки, задачи и текущее время."""
    state = context.crm_state
    request = {
        "question": query["text"],
        "cards": {pid: card["fields"] for pid, card in state["purchases"].items()},
        "omitted_purchase_ids": [],
        "tasks": list(state["tasks"].values()),
        "now": context.now,
    }
    targets = mentioned_purchases([query["text"]], state["purchases"])
    return trim_request(request, fixed_tokens, targets, query["text"])


def verified_evidence(quotes: list[Quote], events: dict[str, dict]) -> list[dict]:
    """Только дословные цитаты из поступивших источников; остальные код отбрасывает."""
    return [
        quote.model_dump()
        for quote in quotes
        if quote.event_id in events and quote.quote.strip() and quote.quote in events[quote.event_id]["text"]
    ]


def task_operation(extracted: CreateTask | UpdateTask, evidence: list[dict]) -> dict:
    """Задача модели -> операция предложения в формате набора; ID новой задачи назначает код."""
    if isinstance(extracted, CreateTask):
        task = {
            "id": "",
            "purchase_id": extracted.purchase_id,
            "title": extracted.title,
            "owner": extracted.owner,
            "due_date": extracted.due_date,
            "status": "open",
        }
        return {"op": "create_task", "task": task, "evidence": evidence}
    fields = extracted.fields.model_dump(exclude_none=True)
    return {"op": "update_task", "task_id": extracted.task_id, "fields": fields, "evidence": evidence}


def value_quotes(value: object, evidence: list[dict], events: dict[str, dict]) -> tuple[dict, ...] | None:
    """Цитаты, в сообщении которых есть число value с учетом "млн" и "тыс"; для не числа None.

    Проверяется все сообщение: модель часто цитирует соседнюю фразу того же сообщения.
    """
    number = to_decimal(value)
    if number is None:
        return None
    return tuple(
        quote
        for quote in evidence
        if number in {parse_amount(match) for match in NUMBER_RE.finditer(events[quote["event_id"]]["text"])}
    )


def latest_event(evidence: list[dict], events: dict[str, dict]) -> dict:
    """Самое позднее событие среди цитат: по нему кандидат упорядочивается."""
    return max((events[quote["event_id"]] for quote in evidence), key=event_order)


def read_extraction(reply: Extraction, events: dict[str, dict]) -> tuple[list[Candidate], list[dict], int]:
    """Ответ модели -> кандидаты полей, операции задач и число отброшенных операций без цитат."""
    candidates, tasks, dropped = [], [], 0
    for extracted in reply.changes:
        evidence = verified_evidence(extracted.evidence, events)
        if not evidence:
            dropped += 1
            continue
        if not isinstance(extracted, SetField):
            tasks.append(task_operation(extracted, evidence))
            continue
        deciding = latest_event(evidence, events)
        candidates.append(
            Candidate(
                purchase_id=extracted.purchase_id,
                field=extracted.field,
                value=extracted.value,
                status=extracted.status,
                evidence=tuple(evidence),
                order=event_order(deciding),
                event_id=deciding["id"],
                grounds=value_quotes(extracted.value, evidence, events),
            )
        )
    return candidates, tasks, dropped


def undated_event(candidate: Candidate, events: dict[str, dict]) -> str | None:
    """Дата кандидата не следует из цитат, а дату в цитате без времени сообщения не посчитать -> ID сообщения."""
    undated = None
    for quote in candidate.evidence:
        occurred_at = events[quote["event_id"]].get("occurred_at")
        for match in DATE_RE.finditer(quote["quote"]):
            resolved = resolve_date(match, occurred_at)
            if resolved is None:
                undated = quote["event_id"]
            elif resolved.isoformat() == candidate.value:
                return None
    return undated


def check_grounding(candidates: list[Candidate], events: dict[str, dict]) -> tuple[list[Candidate], list[dict], int]:
    """Кандидаты -> допустимые значения, вопросы о времени сообщения и число отброшенных денег не числом.

    Цитата проверена только на дословность: 4B писала в цену "неразборчиво" и считала
    "завтра" от неизвестной даты. Такие значения код не предлагает, а на дату закупщик
    получает вопрос missing_source_time. Сверка денег с числами сообщений на записанном
    прогоне dev теряла верные значения: модель цитирует не то сообщение и пишет числа словами.
    """
    kept, issues, not_number = [], [], 0
    for candidate in candidates:
        if candidate.field in MONEY_FIELDS and to_decimal(candidate.value) is None:
            not_number += 1
            continue
        undated = undated_event(candidate, events) if candidate.field in DATE_FIELDS else None
        if undated is None:
            kept.append(candidate)
            continue
        issue = {"code": "missing_source_time", "event_id": undated, "purchase_id": candidate.purchase_id}
        if issue not in issues:
            issues.append(issue)
    return kept, issues, not_number


def token_quotes(operations: list[dict], events: dict[str, dict], pattern: re.Pattern) -> list[tuple[dict, str]]:
    """Цитаты операций со словом pattern и само слово; если в цитатах его нет - целые сообщения цитат."""
    quotes = [quote for operation in operations for quote in operation["evidence"]]
    found = [(quote, match.group(0)) for quote in quotes for match in pattern.finditer(quote["quote"])]
    if found:
        return found
    texts = {quote["event_id"]: events[quote["event_id"]]["text"] for quote in quotes}
    return [
        ({"event_id": event_id, "quote": text}, match.group(0))
        for event_id, text in texts.items()
        for match in pattern.finditer(text)
    ]


def unique_quotes(found: list[tuple[dict, str]]) -> list[dict]:
    """Цитаты без повторов в исходном порядке."""
    quotes = [quote for quote, _ in found]
    return [quote for index, quote in enumerate(quotes) if quote not in quotes[:index]]


def derived_units(operations: list[dict], cards: dict, events: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Деньги и количество без валюты и единицы -> операции currency и unit по словам цитат; без валюты - вопрос.

    4B почти не выводит валюту и единицу отдельной операцией, а "рублей" и "штук" рядом
    с числом код находит сам, как B0. Деньги, равные карточке, build_proposals не предложит,
    поэтому о валюте для них не спрашиваем. Единица в разметке ставится только вместе
    с первым количеством карточки: при смене количества ее нет.
    """
    derived, issues = [], []
    by_purchase = {}
    for operation in operations:
        if operation["op"] == "set_field":
            by_purchase.setdefault(operation["purchase_id"], {})[operation["field"]] = operation
    for purchase_id, changed in by_purchase.items():
        fields = cards.get(purchase_id, {}).get("fields", {})
        money = [
            changed[name]
            for name in sorted(MONEY_FIELDS)
            if name in changed and not same_value(fields.get(name), changed[name]["value"])
        ]
        if money and "currency" not in changed:
            found = token_quotes(money, events, CURRENCY_RE)
            codes = {currency_code(token.strip()) for _, token in found}
            if len(codes) == 1:
                derived.append(field_operation(purchase_id, "currency", codes.pop(), unique_quotes(found)))
            elif not found and fields.get("currency") is None:
                event_id = money[-1]["evidence"][-1]["event_id"]
                issues.append({"code": "missing_currency", "event_id": event_id, "purchase_id": purchase_id})
        if "quantity" in changed and "unit" not in changed and fields.get("quantity") is None:
            found = token_quotes([changed["quantity"]], events, QUANTITY_RE)
            if found:
                derived.append(field_operation(purchase_id, "unit", "шт", unique_quotes(found)))
    return [*operations, *derived], issues


def field_operation(purchase_id: str, field_name: str, value: str, evidence: list[dict]) -> dict:
    """Операция set_field, выведенная кодом."""
    return {"op": "set_field", "purchase_id": purchase_id, "field": field_name, "value": value, "evidence": evidence}


def llm_step(args: dict, prompt_text: str, response: dict | None, started: float, error: str | None) -> TraceStep:
    """Запись вызова модели: промпт, ответ, токены контекста и время до первого токена.

    tokens_in - prompt_eval_count Ollama: полный размер контекста вызова, в том числе
    при попадании в кэш префикса. Без стриминга время до первого токена приближаем
    загрузкой модели и чтением промпта.
    """
    response = response or {}
    message = response.get("message", {})
    return TraceStep(
        kind="llm",
        name=args["phase"],
        args=args,
        error=error,
        latency_ms=(time.perf_counter() - started) * 1000,
        tokens_in=response.get("prompt_eval_count", 0),
        tokens_out=response.get("eval_count", 0),
        model=MODEL,
        prompt_hash=sha256_text(prompt_text),
        prompt=prompt_text,
        output=json.dumps(message["tool_calls"], ensure_ascii=False)
        if message.get("tool_calls")
        else message.get("content"),
        ttft_ms=(response.get("load_duration", 0) + response.get("prompt_eval_duration", 0)) / NS_PER_MS
        if response
        else None,
    )


class LlmSystem:
    """B1 и S1: одно извлечение на кусок; у S1 перед ним ограниченная развилка с инструментами."""

    def __init__(self, name: str, use_tools: bool, client: OllamaClient | None = None) -> None:
        self.name = name
        self.use_tools = use_tools
        self.client = client or OllamaClient()
        self.session = Session()

    def describe(self) -> dict:
        """Модель с digest, параметры генерации, тексты и хэши промптов и схем для meta.json."""
        used = ["extract", "answer", "retry", *(["tools", "final"] if self.use_tools else [])]
        prompts = {name: PROMPTS[name] for name in used}
        schemas = {"extraction": Extraction.model_json_schema(), "answer": QueryAnswer.model_json_schema()}
        if self.use_tools:
            schemas["tools"] = TOOL_SCHEMAS
        return {
            "system": self.name,
            "system_version": SYSTEM_VERSION,
            "model": MODEL,
            "model_digest": self.client.model_digest(MODEL),
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": NUM_CTX,
            "prompt_token_budget": PROMPT_TOKEN_BUDGET,
            "chars_per_token_estimate": CHARS_PER_TOKEN,
            "prompt_hashes": {name: sha256_text(text) for name, text in prompts.items()},
            "prompts": prompts,
            "schema_sha256": sha256_text(json.dumps(schemas, ensure_ascii=False, sort_keys=True)),
        }

    def session_for(self, context: EpisodeContext) -> Session:
        """Состояние текущей истории; новая история или повтор начинают с чистого.

        Раннер не сообщает границу кейса, поэтому история продолжается, пока к уже виденным
        источникам дописываются новые: кусок обрабатывается только при новых источниках.
        """
        seen = [(event["id"], event["text"]) for event in context.seen_events]
        known = self.session.seen
        if len(seen) <= len(known) or seen[: len(known)] != known:
            self.session = Session()
        self.session.seen = seen
        return self.session

    def process_episode(self, context: EpisodeContext, tools: ModelTools) -> SystemOutput:
        """Кусок -> предложения по завершенным источникам, черновик и вопросы закупщику."""
        session = self.session_for(context)
        search = self.use_tools and needs_search(context.events)
        fixed = estimate_tokens(PROMPTS["extract"])
        if search:
            fixed += estimate_tokens(PROMPTS["tools"] + PROMPTS["final"] + json.dumps(TOOL_SCHEMAS, ensure_ascii=False))
        request, stats = episode_request(context, session, fixed)
        if stats["over_budget"]:
            # Ollama молча обрезал бы начало промпта вместе с инструкцией: кусок нужно нарезать мельче
            return self.overflow(context, session, tools, [], stats)
        messages = [
            {"role": "system", "content": PROMPTS["extract"]},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        branch = Branch(messages=messages, steps=[])
        if search:
            branch = self.search(messages, tools, stats)
            if branch.stop_reason is not None:
                return self.fail(context, session, tools, branch.steps, branch.stop_reason, branch.failed_tool)
            # Результаты инструментов (read_evidence отдает тексты целиком) растят вход итогового вызова
            stats = searched_stats(stats, branch.messages)
            if stats["over_budget"]:
                return self.overflow(context, session, tools, branch.steps, stats)
        schemas = TOOL_SCHEMAS if search else None
        args = {"phase": "extract", "context": stats}
        reply, steps, stop_reason = self.ask(branch.messages, Extraction, tools, args, schemas)
        steps = branch.steps + steps
        if reply is None:
            return self.fail(context, session, tools, steps, stop_reason, branch.failed_tool)
        return self.assemble(context, session, tools, reply, steps, branch.failed_tool)

    def answer_query(self, context: EpisodeContext, query: dict, tools: ModelTools) -> SystemOutput:
        """Вопрос закупщика -> ответ кодом, если он сводится к сводке, полям или задачам карточки, иначе модели."""
        answer = route_answer(query["text"], context.crm_state, context.now)
        if answer is not None:
            # Для сводки и просмотра карточки модель не нужна (раздел 2 design doc): 4B на сводке
            # возвращала поля карточки вместо окна и задач
            step = TraceStep(kind="system", name="code_answer", args={"query_id": query["id"]}, result=answer)
            answers = [{"query_id": query["id"], "answer": answer, "evidence": []}]
            return SystemOutput(answers=answers, trace_steps=[step])
        request, stats = query_request(context, query, estimate_tokens(PROMPTS["answer"]))
        messages = [
            {"role": "system", "content": PROMPTS["answer"]},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        reply, steps, stop_reason = self.ask(messages, QueryAnswer, tools, {"phase": "answer", "context": stats})
        answers = [] if reply is None else [{"query_id": query["id"], "answer": reply.answer, "evidence": []}]
        return SystemOutput(answers=answers, trace_steps=steps, stop_reason=stop_reason)

    def chat(
        self, messages: list[dict], tools: ModelTools, args: dict, schema: dict | None, tool_schemas: list | None
    ) -> tuple[dict | None, TraceStep | None, StopReason | None]:
        """Один вызов модели под бюджетом шагов -> ответ, шаг трейса, причина остановки или None."""
        if not tools.spend_model_call().ok:
            return None, None, StopReason.BUDGET_EXHAUSTED
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": TEMPERATURE, "seed": SEED, "num_ctx": NUM_CTX},
        }
        if schema is not None:
            payload["format"] = schema
        if tool_schemas is not None:
            payload["tools"] = tool_schemas
        prompt_text = json.dumps(messages, ensure_ascii=False)
        started = time.perf_counter()
        try:
            response = self.client.chat(payload, tools.budget.model_timeout_s)
        except TimeoutError:
            return None, llm_step(args, prompt_text, None, started, "timeout"), StopReason.TIMEOUT
        return response, llm_step(args, prompt_text, response, started, None), None

    def ask(
        self,
        messages: list[dict],
        reply_model: type[BaseModel],
        tools: ModelTools,
        args: dict,
        tool_schemas: list | None = None,
    ) -> tuple[BaseModel | None, list[TraceStep], StopReason]:
        """Вызов модели по схеме с одним повтором невалидного JSON -> ответ, шаги трейса, статус."""
        steps = []
        for attempt in range(1, tools.budget.max_json_retries + 2):
            schema = reply_model.model_json_schema()
            response, step, stop_reason = self.chat(messages, tools, {**args, "attempt": attempt}, schema, tool_schemas)
            if step is not None:
                steps.append(step)
            if stop_reason is not None:
                return None, steps, stop_reason
            content = response["message"]["content"]
            try:
                reply = reply_model.model_validate_json(content)
            except ValidationError as error:
                step.error = "invalid_output"
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"{PROMPTS['retry']} {error.errors()[0]['msg']}"},
                ]
                continue
            return reply, steps, StopReason.DONE
        return None, steps, StopReason.INVALID_OUTPUT

    def search(self, messages: list[dict], tools: ModelTools, stats: dict) -> Branch:
        """Развилка S1: до двух раундов вызова инструментов, один вызов модели остается на дифф.

        Таймаут или отказ инструмента по лимиту не останавливает кусок: итоговый вызов идет
        по карточкам, но изменения непривязанных сообщений код только показывает закупщику.
        Если сбой забрал и вызовы модели, итоговый вызов откажет по бюджету шагов.
        """
        messages = [*messages, {"role": "user", "content": PROMPTS["tools"]}]
        steps, failed_tool = [], None
        for round_number in range(1, tools.budget.max_model_calls):
            if tools.tool_calls_left == 0:
                break
            args = {"phase": "search", "round": round_number, "context": stats}
            response, step, stop_reason = self.chat(messages, tools, args, None, TOOL_SCHEMAS)
            if step is not None:
                steps.append(step)
            if stop_reason is not None:
                return Branch(messages, steps, stop_reason)
            calls = response["message"].get("tool_calls") or []
            if not calls:
                break
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            # Вызовы сверх лимита тоже идут в инструменты: отказ budget_exhausted попадает в трейс
            for call in calls:
                name, arguments = call["function"]["name"], call["function"].get("arguments")
                # Аргументы - выход модели: не словарь дает invalid_args, как и пропущенный ключ
                arguments = arguments if isinstance(arguments, dict) else {}
                result = self.run_tool(name, arguments, tools)
                payload = json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False)
                messages.append({"role": "tool", "tool_name": name, "content": payload})
                if not result.ok and result.error.code in TOOL_FAILURE_ISSUES:
                    failed_tool = {"code": result.error.code, "name": name, "purchase_id": arguments.get("purchase_id")}
            # Следующий раунд ушел бы с обрезанным промптом; остановку пишет process_episode
            if failed_tool is not None or searched_stats(stats, messages)["over_budget"]:
                break
        messages.append({"role": "user", "content": PROMPTS["final"]})
        return Branch(messages, steps, None, failed_tool)

    def run_tool(self, name: str, arguments: dict, tools: ModelTools) -> ToolResult:
        """Вызов инструмента по имени из ответа модели; неизвестное имя или аргументы - invalid_args."""
        handlers = {
            "find_purchase": lambda: tools.find_purchase(str(arguments["query"])),
            "read_purchase": lambda: tools.read_purchase(str(arguments["purchase_id"])),
            "read_evidence": lambda: tools.read_evidence(str(arguments["purchase_id"]), arguments.get("period")),
        }
        if name not in handlers or any(key not in arguments for key in TOOL_ARGUMENTS[name]):
            return ToolResult.failure("invalid_args", f"unknown tool or missing arguments: {name}")
        return handlers[name]()

    def overflow(
        self, context: EpisodeContext, session: Session, tools: ModelTools, steps: list[TraceStep], stats: dict
    ) -> SystemOutput:
        """Вход не помещается в окно -> остановка с context_overflow и оценкой размера в трейсе."""
        step = TraceStep(kind="system", name="context_overflow", args={"context": stats}, error="budget_exhausted")
        return self.fail(context, session, tools, [*steps, step], StopReason.BUDGET_EXHAUSTED, None, "context_overflow")

    def fail(
        self,
        context: EpisodeContext,
        session: Session,
        tools: ModelTools,
        steps: list[TraceStep],
        stop_reason: StopReason,
        failed_tool: dict | None,
        code: str | None = None,
    ) -> SystemOutput:
        """Кусок без итогового диффа: вопрос закупщику и открытый вопрос для следующих кусков."""
        purchase_id = None
        if failed_tool is not None:
            code = TOOL_FAILURE_ISSUES[failed_tool["code"]]
            purchase_id = failed_tool["purchase_id"]
        code = code or FAILURE_ISSUES.get(stop_reason, stop_reason.value)
        issue = {"code": code, "after": context.episode_id, "event_id": context.events[0]["id"]}
        if purchase_id is not None:
            issue["purchase_id"] = purchase_id
        if not tools.closed:
            ask_again(session, [{**issue, "event_id": event["id"]} for event in context.events], context)
        return SystemOutput(issues=[issue], trace_steps=steps, stop_reason=stop_reason)

    def assemble(
        self,
        context: EpisodeContext,
        session: Session,
        tools: ModelTools,
        reply: Extraction,
        steps: list[TraceStep],
        failed_tool: dict | None,
    ) -> SystemOutput:
        """Ответ модели -> проверенные кандидаты, сведение по источникам и предложения."""
        events = {event["id"]: event for event in context.seen_events}
        cards = context.crm_state["purchases"]
        candidates, tasks, dropped = read_extraction(reply, events)
        candidates, issues, not_number = check_grounding(candidates, events)
        candidates, authority_issues = gate_authority(candidates, events, context.authority_registry, cards)
        issues += authority_issues
        if failed_tool is not None:
            candidates, held_issues = hold_unlinked(candidates, context, failed_tool)
            issues += held_issues
        open_sources = open_sources_of(context.events)
        draft_new, ready_new = [], []
        for candidate in candidates:
            in_draft = source_of(events[candidate.event_id]) in open_sources
            (draft_new if in_draft else ready_new).append(candidate)
        closing = {source_of(event) for event in context.events} - open_sources
        held = [candidate for source in closing for candidate in session.drafts.get(source, [])]
        committed = [
            candidate
            for candidate in session.committed.values()
            if same_value(cards.get(candidate.purchase_id, {}).get("fields", {}).get(candidate.field), candidate.value)
        ]
        merged = merge_candidates([*committed, *held, *ready_new])
        operations, unit_issues = derived_units(to_operations(merged.final, cards), cards, events)
        issues += unit_issues
        ready_tasks = [
            task for task in tasks if source_of(events[task["evidence"][-1]["event_id"]]) not in open_sources
        ]
        ready_tasks += [task for source in closing for task in session.held_tasks.get(source, [])]
        proposals = build_proposals([*operations, *ready_tasks], context, self.name)
        issues += merged.issues
        issues += [issue for source in closing for issue in session.draft_issues.get(source, [])]
        issues += [
            issue.model_dump(exclude_none=True) for issue in reply.issues if keeps_issue(issue, proposals, context)
        ]
        issues = [{**issue, "after": context.episode_id} for issue in issues]
        if not tools.closed:
            remember(session, context, merged, draft_new, tasks, open_sources, closing, proposals)
            ask_again(session, issues, context)
        steps.append(
            TraceStep(
                kind="system",
                name="merge",
                args={"open_sources": sorted(open_sources), "closing_sources": sorted(closing)},
                result={
                    "dropped_without_quote": dropped,
                    "dropped_not_number": not_number,
                    "final": [[*key, candidate.value, candidate.status] for key, candidate in merged.final.items()],
                    "pending": [[*candidate.key, candidate.value, candidate.status] for candidate in merged.pending],
                    "drafts": {source: len(items) for source, items in session.drafts.items()},
                },
            )
        )
        if failed_tool is not None:
            stop_reason = TOOL_STOP_REASONS[failed_tool["code"]]
        else:
            stop_reason = StopReason.CLARIFY if issues else StopReason.DONE
        return SystemOutput(proposals=proposals, issues=issues, trace_steps=steps, stop_reason=stop_reason)


def route_answer(text: str, state: dict, now: str | None) -> dict | None:
    """Вопрос -> ответ кодом по CRM; None, если отвечать должна модель.

    Сводку и действующие условия код собирает целиком. Вопрос о поручении ("кто и к какому
    сроку должен") идет к задачам раньше полей: иначе "срок" в нем дал бы дату поставки.
    """
    if SUMMARY_QUERY_RE.search(text) or CONDITIONS_QUERY_RE.search(text):
        return code_answer(text, state, now)
    if TASKS_QUERY_RE.search(text):
        return tasks_answer(text, state)
    if MODEL_QUERY_RE.search(text):
        return None
    return code_answer(text, state, now)


def tasks_answer(text: str, state: dict) -> dict | None:
    """Вопрос, кто что должен сделать по названной закупке -> ее открытые задачи из CRM."""
    named = [pid for pid in PURCHASE_ID.findall(text) if pid in state["purchases"]]
    if not named:
        return None
    return {"tasks": open_tasks(list(state["tasks"].values()), named[0])}


def keeps_issue(issue: ExtractedIssue, proposals: list[dict], context: EpisodeContext) -> bool:
    """Проверка вопроса модели кодом: вопросы, которые ставит код или опровергает сам кусок, отбрасываются.

    Вопрос о привязке не нужен сообщению с номером существующей закупки, а без event_id
    закупщику не на что ответить. Пакет по закупке и есть разрешение противоречия:
    оставшиеся противоречия находит сведение.
    """
    if issue.code in CODE_ISSUES:
        return False
    if issue.code in LINKING_ISSUES:
        if issue.event_id is None:
            return False
        texts = [event["text"] for event in context.events if event["id"] == issue.event_id]
        return not mentioned_purchases(texts, context.crm_state["purchases"])
    proposed = {pid for proposal in proposals for pid in proposal["based_on"]}
    return not (issue.code == "unresolved_terms" and issue.purchase_id in proposed)


def open_sources_of(events: list[dict]) -> set[str]:
    """Источники куска, запись которых еще не закончилась: их кандидаты остаются в черновике."""
    sources = {source_of(event) for event in events}
    return sources - {source_of(event) for event in events if event.get("source_complete", True)}


def remember(
    session: Session,
    context: EpisodeContext,
    merged: Merge,
    draft_new: list[Candidate],
    tasks: list[dict],
    open_sources: set[str],
    closing: set[str],
    proposals: list[dict],
) -> None:
    """Черновики незавершенных источников, предложенные значения и закрытые вопросы."""
    events = {event["id"]: event for event in context.seen_events}
    for source in open_sources:
        own = [candidate for candidate in draft_new if source_of(events[candidate.event_id]) == source]
        draft = merge_candidates([*session.drafts.get(source, []), *own])
        session.drafts[source] = [*draft.final.values(), *draft.pending]
        # Противоречие убирает оба значения из черновика: без этого вопрос пропал бы к концу записи
        session.draft_issues[source] = session.draft_issues.get(source, []) + draft.issues
        session.held_tasks[source] = session.held_tasks.get(source, []) + [
            task for task in tasks if source_of(events[task["evidence"][-1]["event_id"]]) == source
        ]
    for source in closing:
        session.drafts.pop(source, None)
        session.held_tasks.pop(source, None)
        session.draft_issues.pop(source, None)
    proposed = {
        (operation["purchase_id"], operation["field"])
        for proposal in proposals
        for operation in proposal["operations"]
        if operation["op"] == "set_field"
    }
    for key, candidate in merged.final.items():
        if key in proposed:
            session.committed[key] = replace(candidate, committed=True)
    touched = {pid for proposal in proposals for pid in proposal["based_on"]}
    # Вопрос без предполагаемой закупки закрывает предложение, которое цитирует его событие
    cited = {
        quote["event_id"]
        for proposal in proposals
        for operation in proposal["operations"]
        for quote in operation["evidence"]
    }
    session.open_questions = [
        question
        for question in session.open_questions
        if question["purchase_id"] not in touched and question["event_id"] not in cited
    ]


def ask_again(session: Session, issues: list[dict], context: EpisodeContext) -> None:
    """Вопросы, на которые может ответить следующий кусок, попадают в open_questions с текстом события."""
    texts = {event["id"]: event["text"] for event in context.seen_events}
    for issue in issues:
        if issue["code"] not in OPEN_QUESTION_CODES or issue.get("event_id") not in texts:
            continue
        question = {
            "code": issue["code"],
            "purchase_id": issue.get("purchase_id"),
            "event_id": issue["event_id"],
            "text": texts[issue["event_id"]],
        }
        session.open_questions = [*session.open_questions, question][-MAX_OPEN_QUESTIONS:]


def hold_unlinked(
    candidates: list[Candidate], context: EpisodeContext, failed_tool: dict
) -> tuple[list[Candidate], list[dict]]:
    """После сбоя поиска изменения сообщений без номера закупки не предлагаются.

    Модель привязала их по карточкам, но поиск это не подтвердил: закупщик получает
    вопрос с предполагаемой закупкой, а следующий кусок видит его как открытый вопрос.
    """
    unlinked = {event["id"] for event in context.events if not PURCHASE_ID.search(event["text"])}
    kept, issues = [], []
    code = TOOL_FAILURE_ISSUES[failed_tool["code"]]
    for candidate in candidates:
        if not all(quote["event_id"] in unlinked for quote in candidate.evidence):
            kept.append(candidate)
            continue
        issue = {"code": code, "event_id": candidate.event_id, "purchase_id": candidate.purchase_id}
        if issue not in issues:
            issues.append(issue)
    if not issues:
        cards = context.crm_state["purchases"]
        issues = [
            {"code": code, "event_id": event["id"], "purchase_id": likely_purchase(event["text"], cards)}
            for event in context.events
            if event["id"] in unlinked
        ]
    return kept, issues


def likely_purchase(text: str, cards: dict) -> str | None:
    """Единственная карточка с наибольшим совпадением слов -> предполагаемая закупка для вопроса."""
    scores = {pid: relevance(card["fields"], text) for pid, card in cards.items()}
    best = max(scores.values(), default=0)
    leaders = [pid for pid, score in scores.items() if score == best]
    return leaders[0] if best > 0 and len(leaders) == 1 else None

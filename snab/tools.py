"""Инструменты модели только для чтения, бюджет шагов и инъекция сбоев.

find_purchase, read_purchase и read_evidence видят текущую CRM и только те источники,
которые уже поступили к этому шагу. Бюджет шагов принадлежит харнессу: каждый вызов
инструмента и модели его тратит. Сбои из input.fault_injections система не видит:
она получает обычную типизированную ошибку.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from snab.contracts import StopReason, ToolResult
from snab.crm import MockCrm
from snab.trace import Tracer

# Сбой инструмента -> вопрос закупщику и stop_reason куска; одна карта на B0 и LLM-системы
TOOL_FAILURE_ISSUES = {"timeout": "tool_timeout", "budget_exhausted": "step_budget_exhausted"}
TOOL_STOP_REASONS = {"timeout": StopReason.TOOL_ERROR, "budget_exhausted": StopReason.BUDGET_EXHAUSTED}

PURCHASE_ID = re.compile(r"\bP-[A-Z0-9]+\b")
WORD = re.compile(r"[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)*")
DESCRIPTIVE_FIELDS = ("supplier", "title", "item", "object")
MIN_WORD_LENGTH = 3


@dataclass(frozen=True)
class StepBudget:
    """Лимиты на кусок из раздела 5 design doc: 3 вызова модели, 1 повтор JSON, 90 с на вызов."""

    max_tool_calls: int = 2
    max_model_calls: int = 3
    max_json_retries: int = 1
    model_timeout_s: float = 90.0
    tool_timeout_s: float = 2.0
    # ~3 минуты на кусок: 30 минут модели на 9 кусков встречи
    episode_timeout_s: float = 180.0


DEFAULT_BUDGET = StepBudget()


def words(text: str) -> list[str]:
    """Слова текста в нижнем регистре, достаточно длинные для сравнения."""
    return [word for word in WORD.findall(text.lower()) if len(word) >= MIN_WORD_LENGTH]


def same_stem(left: str, right: str) -> bool:
    """Грубое сравнение словоформ: общий префикс без двух последних букв короткого слова.

    Слова с цифрами (артикулы вроде Р-208) совпадают только целиком.
    """
    if left == right:
        return True
    if any(char.isdigit() for char in left + right):
        return False
    shorter = min(len(left), len(right))
    prefix = 0
    while prefix < shorter and left[prefix] == right[prefix]:
        prefix += 1
    return shorter >= 4 and prefix >= max(MIN_WORD_LENGTH, shorter - 2)


class FaultInjector:
    """Срабатывает в точке `at` на операции `operation`; `once` снимает инъекцию после срабатывания."""

    def __init__(self, injections: list[dict]) -> None:
        self.pending = [dict(injection) for injection in injections]
        self.fired = []
        self.point = None

    def take(self, operation: str) -> str | None:
        """Возвращает имя сбоя для операции в текущей точке или None."""
        for injection in self.pending:
            if injection["at"] == self.point and injection["operation"] == operation:
                if injection.get("once", True):
                    self.pending.remove(injection)
                self.fired.append(injection)
                return injection["fault"]
        return None

    def unconsumed(self) -> list[dict]:
        """Инъекции, до которых система так и не дошла: кейс тогда unsupported."""
        return [dict(injection) for injection in self.pending if injection not in self.fired]


class ModelTools:
    """Инструменты модели и счетчики бюджета шагов текущего куска."""

    def __init__(
        self, crm: MockCrm, faults: FaultInjector, tracer: Tracer, budget: StepBudget = DEFAULT_BUDGET
    ) -> None:
        self.crm = crm
        self.faults = faults
        self.tracer = tracer
        self.budget = budget
        self.tool_calls_left = budget.max_tool_calls
        self.model_calls_left = budget.max_model_calls
        self.closed = False
        self.sources = []

    def start_episode(self) -> None:
        """Новый кусок получает свежий бюджет шагов."""
        self.tool_calls_left = self.budget.max_tool_calls
        self.model_calls_left = self.budget.max_model_calls

    def close(self) -> None:
        """После таймаута системы ее зависший поток больше не читает CRM."""
        self.closed = True

    def add_sources(self, events: list[dict]) -> None:
        """Делает источники доступными для read_evidence."""
        self.sources.extend(events)

    def spend_model_call(self) -> ToolResult:
        """Система списывает вызов модели до запроса; отказ, если лимит куска исчерпан."""
        if self.model_calls_left <= 0:
            return ToolResult.failure(
                "budget_exhausted", f"at most {self.budget.max_model_calls} model calls per chunk"
            )
        self.model_calls_left -= 1
        return ToolResult.success(model_calls_left=self.model_calls_left)

    def find_purchase(self, query: str) -> ToolResult:
        """Поставщик, товар или номер -> кандидаты закупок с основаниями."""
        return self.call("find_purchase", {"query": query}, lambda: self.search_purchases(query))

    def read_purchase(self, purchase_id: str) -> ToolResult:
        """ID -> текущая карточка, задачи, ревизия."""
        return self.call("read_purchase", {"purchase_id": purchase_id}, lambda: self.load_purchase(purchase_id))

    def read_evidence(self, purchase_id: str, period: dict | None = None) -> ToolResult:
        """ID и период {from, to} -> фрагменты уже поступивших источников."""
        args = {"purchase_id": purchase_id, "period": period}
        return self.call("read_evidence", args, lambda: self.collect_evidence(purchase_id, period or {}))

    def call(self, name: str, args: dict, handler: Callable[[], ToolResult]) -> ToolResult:
        """Общая обвязка: бюджет, инъекция сбоя, таймаут и запись в трейс.

        Обработчики читают SQLite в памяти и не ждут внешнего I/O; настоящий дедлайн
        стоит на вызове системы целиком, а здесь медленный ответ помечается таймаутом.
        """
        started = time.perf_counter()
        result = self.run_tool(name, handler)
        latency = time.perf_counter() - started
        if result.ok and latency > self.budget.tool_timeout_s:
            result = ToolResult.failure("timeout", f"{name} took {latency:.2f} s")
        self.tracer.record(
            "tool",
            name,
            args,
            result=result.model_dump(),
            error=result.error and result.error.code,
            latency_ms=latency * 1000,
        )
        return result

    def run_tool(self, name: str, handler: Callable[[], ToolResult]) -> ToolResult:
        """Сначала бюджет, потом инъекция: отказ по лимиту не тратит запланированный сбой."""
        if self.closed:
            return ToolResult.failure("timeout", "system call already timed out")
        if self.tool_calls_left <= 0:
            return ToolResult.failure("budget_exhausted", f"at most {self.budget.max_tool_calls} tool calls per chunk")
        self.tool_calls_left -= 1
        fault = self.faults.take(name)
        if fault == "timeout":
            return ToolResult.failure("timeout", "tool call did not answer within the time limit")
        if fault == "step_budget_exhausted":
            self.tool_calls_left = self.model_calls_left = 0
            return ToolResult.failure("budget_exhausted", "step budget exhausted")
        if fault is not None:
            raise ValueError(f"unknown fault {fault!r} for {name}")
        return handler()

    def search_purchases(self, query: str) -> ToolResult:
        """Кандидаты по номеру, иначе по совпадению слов с поставщиком и товаром."""
        purchases = self.crm.snapshot()["purchases"]
        named = [purchase_id for purchase_id in dict.fromkeys(PURCHASE_ID.findall(query)) if purchase_id in purchases]
        if named:
            return ToolResult.success(candidates=[{"purchase_id": pid, "reason": "номер в тексте"} for pid in named])
        query_words = words(query)
        scores = {}
        for purchase_id, card in purchases.items():
            card_words = {word for field in DESCRIPTIVE_FIELDS for word in words(str(card["fields"].get(field) or ""))}
            matched = {card_word for card_word in card_words if any(same_stem(card_word, word) for word in query_words)}
            if matched:
                scores[purchase_id] = sorted(matched)
        best = max((len(matched) for matched in scores.values()), default=0)
        candidates = [
            {"purchase_id": pid, "reason": "совпали слова: " + ", ".join(matched)}
            for pid, matched in sorted(scores.items())
            if len(matched) == best
        ]
        if not candidates:
            return ToolResult.failure("not_found", "no purchase matches the query", candidates=[])
        if len(candidates) > 1:
            return ToolResult.failure(
                "ambiguous", "several purchases match equally", candidates=[item["purchase_id"] for item in candidates]
            )
        return ToolResult.success(candidates=candidates)

    def load_purchase(self, purchase_id: str) -> ToolResult:
        """Карточка и ее задачи."""
        card = self.crm.read_card(purchase_id)
        if card is None:
            return ToolResult.failure("not_found", f"purchase {purchase_id} not found")
        tasks = [task for task in self.crm.snapshot()["tasks"].values() if task["purchase_id"] == purchase_id]
        return ToolResult.success(card=card["fields"], revision=card["revision"], tasks=tasks)

    def collect_evidence(self, purchase_id: str, period: dict) -> ToolResult:
        """Источники с номером закупки в заданном периоде по времени сообщения."""
        fragments = []
        for event in self.sources:
            occurred = (event.get("occurred_at") or "")[:10]
            in_period = (not period.get("from") or occurred >= period["from"]) and (
                not period.get("to") or occurred <= period["to"]
            )
            if purchase_id in PURCHASE_ID.findall(event["text"]) and in_period:
                fragments.append({"event_id": event["id"], "quote": event["text"]})
        if not fragments:
            return ToolResult.failure("not_found", f"no evidence for {purchase_id}")
        return ToolResult.success(fragments=fragments)

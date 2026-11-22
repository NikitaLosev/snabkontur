"""Инструменты модели: только поступившие источники, лимит вызовов, однократный сбой."""

import pytest

from snab.crm import MockCrm
from snab.tools import FaultInjector, ModelTools
from snab.trace import Tracer

STATE = {
    "purchases": {
        "P-1": {"revision": 1, "fields": {"supplier": "Парус", "title": "кабель"}},
        "P-2": {"revision": 1, "fields": {"supplier": "Парус", "title": "крепёж"}},
    },
    "tasks": {},
}
EVENT_OLD = {"id": "e1", "occurred_at": "2026-11-01T10:00:00+03:00", "text": "По P-1 цена 100 рублей."}
EVENT_NEW = {"id": "e2", "occurred_at": "2026-11-05T10:00:00+03:00", "text": "По P-1 цена 90 рублей."}


def make_tools(injections: list[dict] | None = None) -> ModelTools:
    """Инструменты над CRM из STATE с заданными инъекциями."""
    crm = MockCrm()
    crm.load(STATE)
    faults = FaultInjector(injections or [])
    faults.point = "step1"
    return ModelTools(crm, faults, Tracer("run", "case", 1))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("У Паруса по кабелю сдвинули срок", ["P-1"]),  # поставщик и товар перевешивают одного поставщика
        ("По P-2 без изменений", ["P-2"]),
    ],
)
def test_find_purchase_picks_single_best_candidate(query: str, expected: list[str]) -> None:
    result = make_tools().find_purchase(query)

    assert [candidate["purchase_id"] for candidate in result.data["candidates"]] == expected


def test_find_purchase_reports_tie_as_ambiguous() -> None:
    result = make_tools().find_purchase("Парус подтвердил")

    assert result.error.code == "ambiguous"
    assert result.error.details == {"candidates": ["P-1", "P-2"]}


def test_read_evidence_sees_only_ingested_sources_in_period() -> None:
    tools = make_tools()
    tools.add_sources([EVENT_OLD])
    before_new_source = tools.read_evidence("P-1")
    tools.add_sources([EVENT_NEW])
    in_period = tools.read_evidence("P-1", {"from": "2026-11-03", "to": "2026-11-30"})

    assert [fragment["event_id"] for fragment in before_new_source.data["fragments"]] == ["e1"]
    assert [fragment["event_id"] for fragment in in_period.data["fragments"]] == ["e2"]


def test_calls_over_chunk_budget_are_refused() -> None:
    tools = make_tools()
    results = [tools.read_purchase("P-1") for _ in range(3)]
    tools.start_episode()

    assert [result.ok for result in results] == [True, True, False]
    assert results[2].error.code == "budget_exhausted"
    assert tools.read_purchase("P-1").ok


def test_injected_timeout_fires_once_at_its_point() -> None:
    tools = make_tools([{"at": "step1", "operation": "find_purchase", "fault": "timeout", "once": True}])
    first = tools.find_purchase("P-1")
    second = tools.find_purchase("P-1")

    assert first.error.code == "timeout" and first.error.retryable
    assert second.ok
    assert tools.faults.unconsumed() == []


def test_spent_budget_refuses_without_consuming_fault() -> None:
    tools = make_tools([{"at": "step1", "operation": "find_purchase", "fault": "timeout"}])
    tools.read_purchase("P-1")
    tools.read_purchase("P-1")

    refused = tools.find_purchase("P-1")

    assert refused.error.code == "budget_exhausted"
    assert tools.tool_calls_left == 0
    assert len(tools.faults.unconsumed()) == 1


def test_step_budget_fault_drains_tool_and_model_budget() -> None:
    tools = make_tools([{"at": "step1", "operation": "read_evidence", "fault": "step_budget_exhausted"}])

    evidence = tools.read_evidence("P-1")

    assert evidence.error.code == "budget_exhausted"
    assert tools.read_purchase("P-1").error.code == "budget_exhausted"
    assert tools.spend_model_call().error.code == "budget_exhausted"


def test_model_calls_limited_per_chunk() -> None:
    tools = make_tools()
    spent = [tools.spend_model_call().ok for _ in range(tools.budget.max_model_calls + 1)]

    assert spent == [True] * tools.budget.max_model_calls + [False]

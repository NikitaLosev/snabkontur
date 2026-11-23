"""Раннер на синтетическом кейсе: сбои системы, проверка цитат и сопоставление при подтверждении."""

import threading

from snab.contracts import SystemOutput
from snab.run import CaseReplay
from snab.tools import StepBudget

TEXT = "По P-1 бюджет 4 млн рублей. Анна, запроси счёт до 6 октября 2026 года."
INITIAL_STATE = {"purchases": {"P-1": {"revision": 1, "fields": {"budget": "5000000.00"}}}, "tasks": {}}
SET_BUDGET = {
    "op": "set_field",
    "purchase_id": "P-1",
    "field": "budget",
    "value": "4000000.00",
    "evidence": [{"event_id": "e1", "quote": "бюджет 4 млн рублей"}],
}
TASK = {"purchase_id": "P-1", "owner": "Анна", "due_date": "2026-10-06", "status": "open"}


def create_task(task_id: str, title: str) -> dict:
    """Операция создания задачи с цитатой из e1."""
    quote = "Анна, запроси счёт до 6 октября 2026 года"
    return {
        "op": "create_task",
        "task": dict(TASK, id=task_id, title=title),
        "evidence": [{"event_id": "e1", "quote": quote}],
    }


def make_case(operations: list[dict]) -> dict:
    """Кейс формата 0.1: одно сообщение и подтверждение gold-предложения p1."""
    approve = {"id": "a1", "kind": "approve", "proposal_id": "p1", "expected_revision": {"P-1": 1}, "change_id": "c1"}
    return {
        "id": "SYN",
        "schema_version": "0.1",
        "dataset_version": "synthetic",
        "input": {
            "initial_state": INITIAL_STATE,
            "events": [{"id": "e1", "occurred_at": "2026-10-01T10:00:00+03:00", "text": TEXT}],
            "user_actions": [approve],
        },
        "gold": {
            "proposals": [{"id": "p1", "based_on": {"P-1": 1}, "operations": operations}],
            "checks": [{"after": "a1", "expected_state": INITIAL_STATE}],
            "expected_issues": [],
            "forbidden_actions": [],
            "answers": [],
        },
    }


class FixedSystem:
    """Возвращает одно и то же предложение на эпизод."""

    name = "fixed"

    def __init__(self, operations: list[dict]) -> None:
        self.operations = operations

    def describe(self) -> dict:
        """Описывать нечего."""
        return {"system": self.name}

    def process_episode(self, context, tools) -> SystemOutput:
        """Предложение sys-1 с заданными операциями."""
        proposal = {"id": "sys-1", "based_on": {"P-1": 1}, "operations": self.operations}
        return SystemOutput(proposals=[proposal])

    def answer_query(self, context, query, tools) -> SystemOutput:
        """Запросов в кейсе нет."""
        return SystemOutput()


class CrashingSystem(FixedSystem):
    """Падает на эпизоде."""

    def process_episode(self, context, tools) -> SystemOutput:
        """Ошибка внутри системы."""
        raise RuntimeError("model client crashed")


class HangingSystem(FixedSystem):
    """Не отвечает дольше лимита куска."""

    def __init__(self) -> None:
        super().__init__([])
        self.release = threading.Event()
        self.tools = None

    def process_episode(self, context, tools) -> SystemOutput:
        """Запоминает инструменты и ждет, пока тест не отпустит поток."""
        self.tools = tools
        self.release.wait(timeout=5)
        return SystemOutput()


def system_steps(replay: CaseReplay) -> list[tuple]:
    """(имя, ошибка, статус) записей шагов системы."""
    return [
        (record["name"], record["error"], record["stop_reason"])
        for record in replay.tracer.records
        if record["kind"] == "system"
    ]


def final_state(prediction: dict) -> dict:
    """Снимок CRM после последнего действия."""
    return prediction["gold"]["checks"][-1]["expected_state"]


def test_task_with_other_title_is_approved() -> None:
    case = make_case([SET_BUDGET, create_task("T-1", "Запросить счёт")])
    replay = CaseReplay(case, FixedSystem([create_task("sys-T1", "Запросить счёт у поставщика")]), "run")

    state = final_state(replay.run())

    assert state["purchases"]["P-1"]["revision"] == 2
    assert [task["title"] for task in state["tasks"].values()] == ["Запросить счёт у поставщика"]


def test_unverified_quote_is_scored_but_never_approved() -> None:
    paraphrased = dict(SET_BUDGET, evidence=[{"event_id": "e1", "quote": "бюджет теперь 4 миллиона"}])
    replay = CaseReplay(make_case([SET_BUDGET]), FixedSystem([paraphrased]), "run")

    prediction = replay.run()

    assert [proposal["id"] for proposal in prediction["gold"]["proposals"]] == ["sys-1"]
    assert final_state(prediction) == INITIAL_STATE
    assert dict(replay.invalid_proposals) == {"unverified_quote": 1}
    assert ("process_episode", "invalid_output", "invalid_output") in system_steps(replay)


def test_system_crash_becomes_step_status_without_write() -> None:
    replay = CaseReplay(make_case([SET_BUDGET]), CrashingSystem([]), "run")

    prediction = replay.run()

    assert final_state(prediction) == INITIAL_STATE
    assert ("process_episode", "RuntimeError", "invalid_output") in system_steps(replay)


def test_hanging_system_times_out_and_its_tools_close() -> None:
    system = HangingSystem()
    replay = CaseReplay(make_case([SET_BUDGET]), system, "run", budget=StepBudget(episode_timeout_s=0.05))

    prediction = replay.run()
    system.release.set()

    assert final_state(prediction) == INITIAL_STATE
    assert ("process_episode", "timeout", "timeout") in system_steps(replay)
    assert system.tools.read_purchase("P-1").error.code == "timeout"
    assert replay.tools.read_purchase("P-1").ok

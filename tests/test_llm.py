"""llm_single на поддельном клиенте Ollama: сборка предложений, повтор JSON, бюджет и таймаут."""

import json

import pytest

from snab.contracts import EpisodeContext, StopReason
from snab.crm import MockCrm
from snab.systems.llm import LlmSingleSystem
from snab.tools import DEFAULT_BUDGET, FaultInjector, ModelTools, StepBudget
from snab.trace import Tracer

TEXT = "По P-1 бюджет 4 млн рублей."
STATE = {"purchases": {"P-1": {"revision": 3, "fields": {"budget": "5000000.00"}}}, "tasks": {}}
CONTEXT = EpisodeContext(
    episode_id="step1",
    events=[{"id": "e1", "text": TEXT}],
    seen_events=[{"id": "e1", "text": TEXT}],
    crm_state=STATE,
)
VALID = {
    "changes": [
        {
            "op": "set_field",
            "purchase_id": "P-1",
            "field": "budget",
            "value": "4000000.00",
            "evidence": [{"event_id": "e1", "quote": "бюджет 4 млн рублей"}],
        },
        {
            "op": "create_task",
            "purchase_id": "P-1",
            "title": "Запросить счёт",
            "owner": "Анна",
            "due_date": None,
            "evidence": [{"event_id": "e1", "quote": TEXT}],
        },
    ],
    "issues": [{"code": "unresolved_terms", "event_id": "e1"}],
}


class FakeClient:
    """Отдает заготовленные ответы по очереди; TimeoutError в очереди имитирует зависание."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.payloads = []

    def chat(self, payload: dict, timeout_s: float) -> dict:
        """Следующий ответ в формате /api/chat."""
        self.payloads.append(payload)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": reply}, "prompt_eval_count": 100, "eval_count": 20}

    def model_digest(self, model: str) -> str:
        """Фиксированный digest."""
        return "sha256:fake"


def make_tools(budget: StepBudget = DEFAULT_BUDGET) -> ModelTools:
    """Инструменты над CRM из STATE с заданным бюджетом."""
    crm = MockCrm()
    crm.load(STATE)
    return ModelTools(crm, FaultInjector([]), Tracer("run", "case", 1), budget)


def test_reply_becomes_one_proposal_per_purchase() -> None:
    output = LlmSingleSystem(FakeClient([json.dumps(VALID)])).process_episode(CONTEXT, make_tools())

    assert output.proposals == [
        {
            "id": "step1:llm_single:P-1",
            "available_after": "step1",
            "based_on": {"P-1": 3},
            "operations": [
                VALID["changes"][0],
                {
                    "op": "create_task",
                    "task": {
                        "id": "step1-P-1-T1",
                        "purchase_id": "P-1",
                        "title": "Запросить счёт",
                        "owner": "Анна",
                        "due_date": None,
                        "status": "open",
                    },
                    "evidence": [{"event_id": "e1", "quote": TEXT}],
                },
            ],
        }
    ]
    assert output.issues == [{"code": "unresolved_terms", "after": "step1", "event_id": "e1"}]
    assert [(step.kind, step.tokens_in, step.tokens_out) for step in output.trace_steps] == [("llm", 100, 20)]


def test_invalid_json_is_retried_once() -> None:
    client = FakeClient(["не JSON", json.dumps(VALID)])

    output = LlmSingleSystem(client).process_episode(CONTEXT, make_tools())

    assert output.stop_reason == StopReason.DONE
    assert [step.error for step in output.trace_steps] == ["invalid_output", None]
    assert client.payloads[1]["messages"][-2] == {"role": "assistant", "content": "не JSON"}


@pytest.mark.parametrize(
    ("replies", "budget", "stop_reason", "issues"),
    [
        (["{}", "{}"], StepBudget(), StopReason.INVALID_OUTPUT, []),
        (["{}"], StepBudget(max_model_calls=1), StopReason.BUDGET_EXHAUSTED, [{"code": "step_budget_exhausted"}]),
        ([TimeoutError("slow")], StepBudget(), StopReason.TIMEOUT, []),
    ],
)
def test_failed_extraction_proposes_nothing(
    replies: list, budget: StepBudget, stop_reason: StopReason, issues: list
) -> None:
    output = LlmSingleSystem(FakeClient(replies)).process_episode(CONTEXT, make_tools(budget))

    assert output.proposals == []
    assert output.stop_reason == stop_reason
    assert [{"code": issue["code"]} for issue in output.issues] == issues


def test_describe_records_model_and_prompts() -> None:
    meta = LlmSingleSystem(FakeClient([])).describe()

    assert meta["model_digest"] == "sha256:fake"
    assert meta["temperature"] == 0
    assert set(meta["prompt_hashes"]) == set(meta["prompts"]) == {"extract", "answer", "retry"}

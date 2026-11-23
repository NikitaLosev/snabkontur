"""B1 и S1 на поддельном клиенте Ollama: сборка, повтор JSON, бюджет, развилка, сбои, куски, контекст."""

import json

import pytest

from snab.contracts import EpisodeContext, StopReason, SystemOutput
from snab.crm import MockCrm
from snab.systems import build_system
from snab.systems.llm import PROMPT_TOKEN_BUDGET, Extraction, LlmSystem, estimate_tokens, trim_request, whole_sentence
from snab.tools import DEFAULT_BUDGET, FaultInjector, ModelTools, StepBudget
from snab.trace import Tracer

TEXT = "По P-1 бюджет 4 млн рублей."
FIELDS = {
    "supplier": "Атлас",
    "title": "Ремни",
    "budget": "5000000.00",
    "currency": "RUB",
    "delivery_date": "2026-11-15",
}
STATE = {"purchases": {"P-1": {"revision": 3, "fields": FIELDS}}, "tasks": {}}
EVENT = {"id": "e1", "text": TEXT}
OFFER_CHUNK = {"id": "m2", "source_id": "meet", "text": "По P-1 при полной предоплате могу по 12000 рублей отдать."}
UNLINKED = {"id": "e3", "text": "По ремням у Атласа дату согласовали на 22 ноября 2026 года."}


def change(field: str, value: object, event: dict, quote: str, status: str = "agreed") -> dict:
    """Операция set_field в формате ответа модели."""
    return {
        "action": "set_field",
        "purchase_id": "P-1",
        "field": field,
        "value": value,
        "status": status,
        "evidence": [{"event_id": event["id"], "quote": quote}],
    }


BUDGET_CHANGE = change("budget", "4000000.00", EVENT, "бюджет 4 млн рублей")
VALID = {
    "changes": [
        BUDGET_CHANGE,
        {
            "action": "create_task",
            "purchase_id": "P-1",
            "title": "Запросить счёт",
            "owner": "Анна",
            "due_date": None,
            "evidence": [{"event_id": "e1", "quote": TEXT}],
        },
    ],
    "issues": [{"code": "untrusted_instruction", "event_id": "e1"}],
}
DATE_REPLY = json.dumps(
    {"changes": [change("delivery_date", "2026-11-22", UNLINKED, "22 ноября 2026 года")], "issues": []}
)


def tool_call(name: str, **arguments: object) -> dict:
    """Сообщение модели с одним вызовом инструмента."""
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


class FakeClient:
    """Отдает заготовленные ответы по очереди: строку контента, сообщение с tool_calls или исключение."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.payloads = []

    def chat(self, payload: dict, timeout_s: float) -> dict:
        """Следующий ответ в формате /api/chat."""
        self.payloads.append(payload)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = reply if isinstance(reply, dict) else {"role": "assistant", "content": reply}
        return {"message": message, "prompt_eval_count": 100, "eval_count": 20}

    def model_digest(self, model: str) -> str:
        """Фиксированный digest."""
        return "sha256:fake"

    def request(self, index: int) -> dict:
        """JSON-вход куска из вызова номер index."""
        return json.loads(self.payloads[index]["messages"][1]["content"])


def make_tools(state: dict = STATE, faults: list | None = None, budget: StepBudget = DEFAULT_BUDGET) -> ModelTools:
    """Инструменты над CRM из state; точка инъекции - step1."""
    crm = MockCrm()
    crm.load(state)
    injector = FaultInjector(faults or [])
    injector.point = "step1"
    return ModelTools(crm, injector, Tracer("run", "case", 1), budget)


def make_context(
    events: list[dict], seen: list[dict] | None = None, episode: str = "step1", state: dict = STATE, **extra
) -> EpisodeContext:
    """Контекст куска: новые события и все поступившие к нему."""
    return EpisodeContext(
        episode_id=episode, events=events, seen_events=[*(seen or []), *events], crm_state=state, **extra
    )


def single(replies: list) -> tuple[LlmSystem, FakeClient]:
    """B1 на поддельном клиенте."""
    client = FakeClient(replies)
    return LlmSystem("llm_single", use_tools=False, client=client), client


def agent(replies: list) -> tuple[LlmSystem, FakeClient]:
    """S1 на поддельном клиенте."""
    client = FakeClient(replies)
    return LlmSystem("agent", use_tools=True, client=client), client


def hybrid(replies: list) -> tuple[LlmSystem, FakeClient]:
    """S1.2 на поддельном клиенте: инструменты вызывает код."""
    client = FakeClient(replies)
    return LlmSystem("hybrid", use_tools=False, client=client, code_routing=True), client


def test_reply_becomes_one_proposal_per_purchase() -> None:
    system, _ = single([json.dumps(VALID)])

    output = system.process_episode(make_context([EVENT]), make_tools())

    assert output.proposals == [
        {
            "id": "step1:llm_single:P-1",
            "available_after": "step1",
            "based_on": {"P-1": 3},
            "operations": [
                {
                    "op": "set_field",
                    "purchase_id": "P-1",
                    "field": "budget",
                    "value": "4000000.00",
                    "evidence": BUDGET_CHANGE["evidence"],
                },
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
    assert output.issues == [{"code": "untrusted_instruction", "event_id": "e1", "after": "step1"}]
    assert output.stop_reason == StopReason.CLARIFY
    assert [(step.kind, step.tokens_in) for step in output.trace_steps] == [("llm", 100), ("system", 0)]


@pytest.mark.parametrize(
    ("issue", "kept"),
    [
        ({"code": "ambiguous_purchase", "event_id": "e1"}, False),
        ({"code": "ambiguous_purchase", "event_id": "e3"}, True),
        ({"code": "unresolved_terms", "event_id": "e1", "purchase_id": "P-1"}, False),
        ({"code": "unknown_speaker_authority", "event_id": "e3"}, False),
        # Вопрос о привязке без сообщения: закупщику не на что ответить
        ({"code": "ambiguous_purchase", "purchase_id": "P-1"}, False),
    ],
)
def test_model_issue_checked_by_code(issue: dict, kept: bool) -> None:
    # e1 называет P-1 и дает пакет по P-1; e3 без номера
    reply = {"changes": [BUDGET_CHANGE], "issues": [issue]}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([EVENT, UNLINKED]), make_tools())

    assert [item["code"] for item in output.issues] == ([issue["code"]] if kept else [])


def test_quote_not_in_source_drops_operation() -> None:
    reply = {"changes": [change("budget", "4000000.00", EVENT, "бюджет 3 млн")], "issues": []}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([EVENT]), make_tools())

    assert (output.proposals, output.stop_reason) == ([], StopReason.DONE)


def test_invalid_json_is_retried_once() -> None:
    system, client = single(["не JSON", json.dumps(VALID)])

    output = system.process_episode(make_context([EVENT]), make_tools())

    assert len(output.proposals) == 1
    assert [step.error for step in output.trace_steps if step.kind == "llm"] == ["invalid_output", None]
    assert client.payloads[1]["messages"][-2] == {"role": "assistant", "content": "не JSON"}


@pytest.mark.parametrize(
    ("replies", "budget", "stop_reason", "issue_code"),
    [
        (["{}", "{}"], StepBudget(), StopReason.INVALID_OUTPUT, "invalid_output"),
        (["{}"], StepBudget(max_model_calls=1), StopReason.BUDGET_EXHAUSTED, "step_budget_exhausted"),
        ([TimeoutError("slow")], StepBudget(), StopReason.TIMEOUT, "model_timeout"),
    ],
)
def test_failed_extraction_proposes_nothing_and_says_why(
    replies: list, budget: StepBudget, stop_reason: StopReason, issue_code: str
) -> None:
    system, _ = single(replies)

    output = system.process_episode(make_context([EVENT]), make_tools(budget=budget))

    assert output.proposals == []
    assert output.stop_reason == stop_reason
    assert [issue["code"] for issue in output.issues] == [issue_code]


def test_chunk_over_context_budget_stops_without_model_call() -> None:
    meeting = {"id": "e24", "text": "По P-1 обсуждали маркировку. " * 1000}
    system, client = single([])

    output = system.process_episode(make_context([meeting]), make_tools())

    assert client.payloads == []
    assert output.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert output.issues == [{"code": "context_overflow", "after": "step1", "event_id": "e24"}]


def test_single_never_searches() -> None:
    system, client = single([DATE_REPLY])

    output = system.process_episode(make_context([UNLINKED]), make_tools())

    assert len(output.proposals) == 1
    assert ["tools" in payload for payload in client.payloads] == [False]


def test_agent_skips_branch_when_chunk_names_purchase() -> None:
    system, client = agent([json.dumps(VALID)])

    system.process_episode(make_context([EVENT]), make_tools())

    assert len(client.payloads) == 1
    assert "tools" not in client.payloads[0]


def test_agent_searches_then_extracts() -> None:
    system, client = agent([tool_call("find_purchase", query="ремни Атлас"), "нет", DATE_REPLY])
    tools = make_tools()

    output = system.process_episode(make_context([UNLINKED]), tools)

    assert [op["value"] for proposal in output.proposals for op in proposal["operations"]] == ["2026-11-22"]
    assert [(record["name"], record["error"]) for record in tools.tracer.records] == [("find_purchase", None)]
    assert [step.name for step in output.trace_steps] == ["search", "search", "extract", "merge"]
    tool_message = client.payloads[2]["messages"][-2]
    assert tool_message["role"] == "tool" and "P-1" in tool_message["content"]


def test_agent_spends_at_most_three_model_calls() -> None:
    searches = [tool_call("read_purchase", purchase_id="P-1"), tool_call("read_purchase", purchase_id="P-1")]
    system, client = agent([*searches, "не JSON"])

    output = system.process_episode(make_context([UNLINKED]), make_tools())

    assert len(client.payloads) == 3
    assert output.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert output.proposals == []


def test_find_purchase_timeout_asks_purchaser_and_next_chunk_sees_question() -> None:
    fault = [{"at": "step1", "operation": "find_purchase", "fault": "timeout", "once": True}]
    follow_up = {"id": "e4", "text": "Склад тоже подтвердил 22 ноября 2026 года."}
    replies = [tool_call("find_purchase", query="ремни"), DATE_REPLY, "нет", json.dumps({"changes": [], "issues": []})]
    system, client = agent(replies)

    output = system.process_episode(make_context([UNLINKED]), make_tools(faults=fault))
    system.process_episode(make_context([follow_up], seen=[UNLINKED], episode="step2"), make_tools())

    assert output.proposals == []
    assert output.stop_reason == StopReason.TOOL_ERROR
    assert output.issues == [{"code": "tool_timeout", "event_id": "e3", "purchase_id": "P-1", "after": "step1"}]
    assert client.request(3)["open_questions"] == [
        {"code": "tool_timeout", "purchase_id": "P-1", "event_id": "e3", "text": UNLINKED["text"]}
    ]


def test_read_evidence_budget_fault_stops_without_extraction() -> None:
    fault = [{"at": "step1", "operation": "read_evidence", "fault": "step_budget_exhausted", "once": True}]
    old_letter = {"id": "e3", "text": "По этим ремням из старого письма новая цена есть, номер не помню."}
    system, client = agent([tool_call("read_evidence", purchase_id="P-1")])

    output = system.process_episode(make_context([old_letter]), make_tools(faults=fault))

    assert len(client.payloads) == 1
    assert output.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert output.issues == [
        {"code": "step_budget_exhausted", "after": "step1", "event_id": "e3", "purchase_id": "P-1"}
    ]


def test_tool_calls_over_limit_are_refused_visibly() -> None:
    calls = [
        {"function": {"name": "find_purchase", "arguments": {"query": "ремни Атлас"}}},
        {"function": {"name": "read_purchase", "arguments": {"purchase_id": "P-1"}}},
        {"function": {"name": "read_evidence", "arguments": {"purchase_id": "P-1"}}},
    ]
    system, client = agent([{"role": "assistant", "content": "", "tool_calls": calls}, DATE_REPLY])
    tools = make_tools()

    output = system.process_episode(make_context([UNLINKED]), tools)

    # Лимит два вызова: третий отклонен харнессом, а не выброшен молча
    assert [(record["name"], record["error"]) for record in tools.tracer.records] == [
        ("find_purchase", None),
        ("read_purchase", None),
        ("read_evidence", "budget_exhausted"),
    ]
    assert [message["role"] for message in client.payloads[1]["messages"]].count("tool") == 3
    assert output.proposals == []
    assert output.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert output.issues == [
        {"code": "step_budget_exhausted", "event_id": "e3", "purchase_id": "P-1", "after": "step1"}
    ]


def test_oversized_tool_results_stop_before_final_call() -> None:
    old_letter = {"id": "e3", "text": "По этим ремням из старого письма новая цена есть, номер не помню."}
    earlier = {"id": "e0", "text": "По P-1 обсуждали маркировку. " * 700}
    system, client = agent([tool_call("read_evidence", purchase_id="P-1")])
    tools = make_tools()
    tools.add_sources([earlier])

    output = system.process_episode(make_context([old_letter]), tools)

    assert len(client.payloads) == 1
    assert output.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert output.issues == [{"code": "context_overflow", "after": "step1", "event_id": "e3"}]
    overflow = output.trace_steps[-1]
    assert overflow.name == "context_overflow"
    assert overflow.args["context"]["estimated_tokens"] > PROMPT_TOKEN_BUDGET


def test_contradiction_in_open_meeting_is_asked_after_source_ends() -> None:
    both = {"id": "m2", "source_id": "meet", "text": "По P-1 Анна: берём по 100. Борис: берём по 120."}
    close = {"id": "m8", "source_id": "meet", "text": "По P-1 на этом всё."}
    first = {
        "changes": [
            change("unit_price", "100", both, "берём по 100"),
            change("unit_price", "120", both, "берём по 120"),
        ],
        "issues": [],
    }
    system, _ = single([json.dumps(first), json.dumps({"changes": [], "issues": []})])

    partial = system.process_episode(make_context([{**both, "source_complete": False}], episode="step3"), make_tools())
    final = system.process_episode(
        make_context([{**close, "source_complete": True}], seen=[both], episode="step4"), make_tools()
    )

    assert (partial.proposals, partial.issues) == ([], [])
    assert final.proposals == []
    assert final.issues == [
        {"code": "unresolved_terms", "purchase_id": "P-1", "field": "unit_price", "event_ids": ["m2"], "after": "step4"}
    ]


def test_question_without_purchase_closes_when_proposal_cites_its_event() -> None:
    fault = [{"at": "step1", "operation": "find_purchase", "fault": "timeout", "once": True}]
    # Ни одно слово не совпадает с карточкой: предполагаемой закупки у вопроса нет
    vague = {"id": "e5", "text": "Дату согласовали на 22 ноября 2026 года."}
    linked = {"id": "e6", "text": "По P-1 подтверждаю."}
    later = {"id": "e7", "text": "По P-1 больше ничего."}
    empty = json.dumps({"changes": [], "issues": []})
    cites_vague = json.dumps(
        {"changes": [change("delivery_date", "2026-11-22", vague, "22 ноября 2026 года")], "issues": []}
    )
    system, client = agent([tool_call("find_purchase", query="дата"), empty, cites_vague, empty])

    system.process_episode(make_context([vague]), make_tools(faults=fault))
    system.process_episode(make_context([linked], seen=[vague], episode="step2"), make_tools())
    system.process_episode(make_context([later], seen=[vague, linked], episode="step3"), make_tools())

    assert client.request(2)["open_questions"] == [
        {"code": "tool_timeout", "purchase_id": None, "event_id": "e5", "text": vague["text"]}
    ]
    assert client.request(3)["open_questions"] == []


def test_meeting_chunks_become_one_package_after_source_ends() -> None:
    offer = {**OFFER_CHUNK, "occurred_at": "2026-11-03T10:15:00+03:00", "chunk_index": 2}
    accept = {
        "id": "m8",
        "source_id": "meet",
        "occurred_at": "2026-11-03T11:10:00+03:00",
        "chunk_index": 8,
        "text": "По P-1 финансы подтвердили, по 12000 рублей берём.",
    }
    state = {
        "purchases": {
            "P-1": {"revision": 1, "fields": {"unit_price": "13000.00", "currency": "RUB", "prepayment_percent": 0}}
        },
        "tasks": {},
    }
    first = {
        "changes": [
            change("unit_price", "12000.00", offer, "по 12000 рублей", "conditional"),
            change("prepayment_percent", 100, offer, "при полной предоплате", "conditional"),
        ],
        "issues": [],
    }
    second = {"changes": [change("unit_price", "12000.00", accept, "по 12000 рублей берём")], "issues": []}
    system, client = single([json.dumps(first), json.dumps(second)])

    def chunk(event: dict, complete: bool, seen: list[dict], episode: str) -> EpisodeContext:
        """Кусок встречи с пометкой конца записи от раннера."""
        return EpisodeContext(
            episode_id=episode,
            events=[{**event, "source_complete": complete}],
            seen_events=[*seen, event],
            crm_state=state,
        )

    partial = system.process_episode(chunk(offer, False, [], "step3"), make_tools(state))
    final = system.process_episode(chunk(accept, True, [offer], "step4"), make_tools(state))

    assert partial.proposals == []
    assert [entry["status"] for entry in client.request(1)["draft"]] == ["conditional", "conditional"]
    operations = final.proposals[0]["operations"]
    assert [(op["field"], op["value"]) for op in operations] == [
        ("prepayment_percent", 100),
        ("unit_price", "12000.00"),
    ]
    assert [quote["event_id"] for quote in operations[0]["evidence"]] == ["m2", "m8"]


def test_repeated_case_starts_with_empty_session() -> None:
    offer = OFFER_CHUNK
    reply = json.dumps(
        {"changes": [change("unit_price", "12000.00", offer, "по 12000 рублей", "conditional")], "issues": []}
    )
    system, client = single([reply, reply])
    context = EpisodeContext(
        episode_id="step3", events=[{**offer, "source_complete": False}], seen_events=[offer], crm_state=STATE
    )

    system.process_episode(context, make_tools())
    system.process_episode(context, make_tools())

    assert client.request(1)["draft"] == []


def test_unknown_speaker_change_is_question_not_proposal() -> None:
    voice = {"id": "e2", "speaker_id": "SPEAKER_01", "text": "По P-1 лимит теперь 800000 рублей."}
    reply = {"changes": [change("budget", "800000.00", voice, "лимит теперь 800000 рублей")], "issues": []}
    system, _ = single([json.dumps(reply)])
    registry = {"Марина": {"can_confirm": ["budget"]}}

    output = system.process_episode(make_context([voice], authority_registry=registry), make_tools())

    assert output.proposals == []
    assert output.issues == [
        {"code": "unknown_speaker_authority", "event_id": "e2", "purchase_id": "P-1", "after": "step1"}
    ]
    assert output.stop_reason == StopReason.CLARIFY


def card(pid: str, supplier: str) -> dict:
    """Компактная карточка с поставщиком и длинным описанием."""
    return {"supplier": supplier, "title": "Позиция " + "x" * 400}


def trimmed(fixed_tokens: int) -> tuple[dict, dict]:
    """Запрос с целевой карточкой P-1, двумя посторонними и двумя старыми цитатами."""
    request = {
        "events": [{"id": "e9", "text": "По P-1 и поставщику Вектор срок сдвинули."}],
        "cards": {"P-1": card("P-1", "Атлас"), "P-2": card("P-2", "Вектор"), "P-3": card("P-3", "Орбита")},
        "omitted_purchase_ids": [],
        "tasks": [],
        "draft": [{"purchase_id": "P-1", "quote": "q" * 300}],
        "open_questions": [{"purchase_id": "P-1", "text": "t" * 300}],
    }
    return trim_request(request, fixed_tokens, {"P-1"}, request["events"][0]["text"])


def test_context_trim_drops_old_quotes_first() -> None:
    _, untouched = trimmed(0)
    without_quotes = estimate_tokens("q" * 300) + estimate_tokens("t" * 300)

    request, stats = trimmed(PROMPT_TOKEN_BUDGET - untouched["estimated_tokens"] + without_quotes // 2)

    assert stats["quotes_dropped"] == 2
    assert request["open_questions"][0]["text"] is None and request["draft"][0]["quote"] is None
    assert sorted(request["cards"]) == ["P-1", "P-2", "P-3"]


def test_context_trim_then_drops_unmentioned_cards_but_never_target() -> None:
    request, stats = trimmed(PROMPT_TOKEN_BUDGET)

    # Орбита не совпала со словами куска и уходит раньше Вектора
    assert stats["cards_omitted"] == ["P-2", "P-3"] and request["omitted_purchase_ids"] == ["P-2", "P-3"]
    assert list(request["cards"]) == ["P-1"]
    assert stats["over_budget"] is True


def test_context_trim_keeps_everything_that_fits() -> None:
    request, stats = trimmed(0)

    assert (stats["quotes_dropped"], stats["cards_omitted"], stats["over_budget"]) == (0, [], False)


@pytest.mark.parametrize(
    "text",
    [
        "Кто поставщик по P-1?",
        # В карточке есть срок, но не письмо и не основание: подмножество полей ответом не считается
        "Срок из последнего доставленного письма по P-1 сейчас основной?",
        "Когда ожидаем поставку по P-1 и на чём основан срок?",
        "Какой лимит сейчас по P-1 и откуда он взялся?",
    ],
)
def test_answer_query_returns_model_answer(text: str) -> None:
    system, _ = single([json.dumps({"answer": {"supplier": "Атлас"}})])
    query = {"id": "a1", "text": text}

    output = system.answer_query(make_context([]), query, make_tools())

    assert output.answers == [{"query_id": "a1", "answer": {"supplier": "Атлас"}, "evidence": []}]


def task(task_id: str, due_date: str | None, status: str = "open") -> dict:
    """Задача P-1."""
    return {
        "id": task_id,
        "purchase_id": "P-1",
        "title": "Сверить",
        "owner": "Анна",
        "due_date": due_date,
        "status": status,
    }


TASK_STATE = {
    "purchases": STATE["purchases"],
    "tasks": {
        "T1": task("T1", "2026-11-10"),
        "T2": task("T2", "2026-11-16"),
        "T3": task("T3", None),
        "T4": task("T4", "2026-11-12", "done"),
    },
}


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        (
            "Что у меня на сегодня и ближайшие два дня? Просрочку тоже покажи.",
            # Окно 14-16 ноября: T1 просрочена, T2 в окне, у T3 нет срока, T4 выполнена; поставка P-1 15 ноября
            {
                "window_start": "2026-11-14",
                "window_end": "2026-11-16",
                "overdue_task_ids": ["T1"],
                "upcoming_task_ids": ["T2"],
                "unscheduled_task_ids": ["T3"],
                "deliveries": [{"purchase_id": "P-1", "delivery_date": "2026-11-15"}],
            },
        ),
        ("Какой лимит по P-1?", {"budget": "5000000.00", "currency": "RUB"}),
        (
            "Кто ещё что должен сделать по P-1?",
            {"tasks": [task("T1", "2026-11-10"), task("T2", "2026-11-16"), task("T3", None)]},
        ),
        # "срок" здесь о поручении, а не о дате поставки
        (
            "Кто и к какому сроку должен принять остаток по P-1?",
            {"tasks": [task("T1", "2026-11-10"), task("T2", "2026-11-16"), task("T3", None)]},
        ),
    ],
)
def test_summary_fields_and_tasks_answered_by_code_without_model(text: str, answer: dict) -> None:
    system, client = single([])
    context = make_context([], state=TASK_STATE, now="2026-11-14T08:00:00+03:00")

    output = system.answer_query(context, {"id": "a1", "text": text}, make_tools(TASK_STATE))

    assert client.payloads == []
    assert output.answers == [{"query_id": "a1", "answer": answer, "evidence": []}]


def test_extraction_schema_names_operation_first() -> None:
    # Ollama строит грамматику с ключами по алфавиту: первый ключ варианта выбирает операцию
    schema = Extraction.model_json_schema()
    variants = [
        schema["$defs"][ref["$ref"].split("/")[-1]] for ref in schema["properties"]["changes"]["items"]["anyOf"]
    ]

    first_keys = {min(variant["properties"]) for variant in variants}
    actions = {variant["properties"]["action"]["const"] for variant in variants}

    assert (first_keys, actions) == ({"action"}, {"set_field", "create_task", "update_task"})


def operations_of(output: SystemOutput) -> list[tuple]:
    """(поле, значение, цитаты) операций всех предложений."""
    return [
        (op["field"], op["value"], [quote["quote"] for quote in op["evidence"]])
        for proposal in output.proposals
        for op in proposal["operations"]
    ]


def test_partial_receipt_sets_status_by_code() -> None:
    state = {
        "purchases": {"P-1": {"revision": 2, "fields": {"quantity": 10, "delivered_quantity": 0, "status": "ordered"}}},
        "tasks": {},
    }
    receipt = {"id": "e7", "text": "По P-1 приняли 4 штуки, это первая часть заказа."}
    reply = {"changes": [change("delivered_quantity", 4, receipt, "приняли 4 штуки")], "issues": []}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([receipt], state=state), make_tools(state))

    assert operations_of(output) == [
        ("delivered_quantity", 4, ["приняли 4 штуки"]),
        ("status", "partially_delivered", ["приняли 4 штуки"]),
    ]


BARE = {"purchases": {"P-1": {"revision": 1, "fields": {"supplier": "Атлас"}}}, "tasks": {}}


def test_currency_and_unit_follow_words_in_quotes() -> None:
    order = {"id": "e8", "text": "По P-1 берём 8 штук по 125 рублей."}
    reply = {
        "changes": [change("quantity", 8, order, "8 штук"), change("unit_price", "125.00", order, "по 125 рублей")],
        "issues": [],
    }
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([order], state=BARE), make_tools(BARE))

    assert operations_of(output) == [
        ("quantity", 8, ["8 штук"]),
        ("unit_price", "125.00", ["по 125 рублей"]),
        ("currency", "RUB", ["по 125 рублей"]),
        ("unit", "шт", ["8 штук"]),
    ]
    assert output.issues == []


def test_currency_from_message_when_quote_has_none() -> None:
    order = {"id": "e8", "text": "По P-1 по 125, рублей хватает."}
    reply = {"changes": [change("unit_price", "125.00", order, "по 125")], "issues": []}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([order], state=BARE), make_tools(BARE))

    assert operations_of(output) == [("unit_price", "125.00", ["по 125"]), ("currency", "RUB", [order["text"]])]


def test_price_without_currency_asks_purchaser() -> None:
    price = {"id": "e9", "text": "По P-1 цена 125 за штуку."}
    reply = {"changes": [change("unit_price", "125.00", price, "цена 125 за штуку")], "issues": []}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([price], state=BARE), make_tools(BARE))

    assert operations_of(output) == [("unit_price", "125.00", ["цена 125 за штуку"])]
    assert output.issues == [{"code": "missing_currency", "event_id": "e9", "purchase_id": "P-1", "after": "step1"}]


def test_money_equal_to_card_asks_no_currency() -> None:
    state = {"purchases": {"P-1": {"revision": 1, "fields": {"budget": "5000000.00", "currency": None}}}, "tasks": {}}
    same = {"id": "e9", "text": "По P-1 бюджет 5000000 без изменений."}
    reply = {"changes": [change("budget", "5000000.00", same, "бюджет 5000000")], "issues": []}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([same], state=state), make_tools(state))

    assert (output.proposals, output.issues) == ([], [])


def test_changed_quantity_gets_no_unit() -> None:
    # Единица ставится только с первым количеством карточки, как в разметке D-WOR-09
    state = {"purchases": {"P-1": {"revision": 1, "fields": {"quantity": 10}}}, "tasks": {}}
    order = {"id": "e8", "text": "По P-1 нужно 8 штук."}
    reply = {"changes": [change("quantity", 8, order, "8 штук")], "issues": []}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([order], state=state), make_tools(state))

    assert operations_of(output) == [("quantity", 8, ["8 штук"])]


@pytest.mark.parametrize(
    ("occurred_at", "proposed", "issues"),
    [
        (None, [], [{"code": "missing_source_time", "event_id": "e5", "purchase_id": "P-1", "after": "step1"}]),
        ("2026-11-19T10:00:00+03:00", [("delivery_date", "2026-11-20", ["поставим завтра"])], []),
    ],
)
def test_relative_date_needs_message_time(occurred_at: str | None, proposed: list, issues: list) -> None:
    soon = {"id": "e5", "occurred_at": occurred_at, "text": "По P-1 поставим завтра."}
    reply = {"changes": [change("delivery_date", "2026-11-20", soon, "поставим завтра")], "issues": []}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([soon]), make_tools())

    assert (operations_of(output), output.issues) == (proposed, issues)


def test_money_that_is_not_a_number_is_not_proposed() -> None:
    blurred = {"id": "e6", "text": "По P-1 цена неразборчиво, рублей."}
    reply = {"changes": [change("unit_price", "неразборчиво", blurred, "цена неразборчиво")], "issues": []}
    system, _ = single([json.dumps(reply)])

    output = system.process_episode(make_context([blurred]), make_tools())

    assert (output.proposals, output.issues) == ([], [])


@pytest.mark.parametrize("amount", ["800000", "восемьсот тысяч"])
def test_unrelated_authorized_quote_does_not_pass_unknown_speaker_value(amount: str) -> None:
    marina = {"id": "e1", "author": "Марина", "text": "По P-1 подтвердил лимит 1000000 рублей."}
    voice = {"id": "e2", "speaker_id": "SPEAKER_01", "text": f"По P-1 лимит теперь {amount} рублей."}
    pooled = change("budget", "800000.00", voice, f"лимит теперь {amount}")
    pooled["evidence"].insert(0, {"event_id": "e1", "quote": "подтвердил лимит 1000000"})
    system, _ = single([json.dumps({"changes": [pooled], "issues": []})])
    registry = {"Марина": {"can_confirm": ["budget"]}}

    output = system.process_episode(make_context([marina, voice], authority_registry=registry), make_tools())

    assert output.proposals == []
    assert output.issues == [
        {"code": "unknown_speaker_authority", "event_id": "e2", "purchase_id": "P-1", "after": "step1"}
    ]


@pytest.mark.parametrize(
    ("name", "version", "extra_prompts"),
    [("llm_single", "s1.1", set()), ("agent", "s1.1", {"tools", "final"}), ("hybrid", "s1.2", {"hybrid"})],
)
def test_describe_records_model_prompts_and_schemas(name: str, version: str, extra_prompts: set) -> None:
    system = build_system(name)
    system.client = FakeClient([])

    meta = system.describe()

    assert (meta["system"], meta["system_version"], meta["model_digest"]) == (name, version, "sha256:fake")
    expected = {"extract", "answer", "retry"} | extra_prompts
    assert set(meta["prompt_hashes"]) == set(meta["prompts"]) == expected


EMPTY_REPLY = json.dumps({"changes": [], "issues": []})


def cards_state(*cards: tuple[str, str, str]) -> dict:
    """CRM из карточек (ID, поставщик, товар)."""
    purchases = {
        pid: {"revision": 1, "fields": {"supplier": supplier, "title": title}} for pid, supplier, title in cards
    }
    return {"purchases": purchases, "tasks": {}}


TWO_SUPPLIERS = cards_state(("P-1", "Атлас", "Ремни"), ("P-2", "Вектор", "Сальники"))


def route_step(output: SystemOutput) -> dict:
    """Результат шага route из трейса системы."""
    return next(step.result for step in output.trace_steps if step.name == "route")


def merge_step(output: SystemOutput) -> dict:
    """Результат шага merge из трейса системы."""
    return next(step.result for step in output.trace_steps if step.name == "merge")


def test_hybrid_skips_search_when_every_event_names_purchase() -> None:
    system, client = hybrid([json.dumps({"changes": [BUDGET_CHANGE], "issues": []})])
    tools = make_tools()

    output = system.process_episode(make_context([EVENT]), tools)

    assert tools.tracer.records == []
    assert len(client.payloads) == 1 and "tools" not in client.payloads[0]
    assert client.request(0)["links"] == []
    assert [op["value"] for proposal in output.proposals for op in proposal["operations"]] == ["4000000.00"]


@pytest.mark.parametrize(
    ("state", "text", "calls", "links"),
    [
        # Запрос - слова события, совпавшие с поставщиком и товаром, а не фраза модели
        (
            STATE,
            UNLINKED["text"],
            [("find_purchase", {"query": "ремням атласа"})],
            [{"event_id": "e3", "purchase_id": "P-1"}],
        ),
        (
            cards_state(("P-1", "Атлас", "Ремни"), ("P-2", "Атлас", "Ремни")),
            UNLINKED["text"],
            [("find_purchase", {"query": "ремням атласа"})],
            [{"event_id": "e3", "purchase_id": None, "candidates": ["P-1", "P-2"]}],
        ),
        # Слов карточек нет: искать нечего, единственная закупка CRM дает выведенную привязку
        (
            STATE,
            "Склад подтвердил 22 ноября 2026 года.",
            [],
            [{"event_id": "e3", "purchase_id": "P-1", "inferred": True}],
        ),
        (TWO_SUPPLIERS, "Склад подтвердил 22 ноября 2026 года.", [], []),
    ],
)
def test_hybrid_links_unnamed_event_by_code(state: dict, text: str, calls: list, links: list) -> None:
    system, client = hybrid([EMPTY_REPLY])
    tools = make_tools(state)

    system.process_episode(make_context([{"id": "e3", "text": text}], state=state), tools)

    assert [(record["name"], record["args"]) for record in tools.tracer.records] == calls
    assert client.request(0)["links"] == links


def test_hybrid_reads_evidence_for_prior_reference_without_search() -> None:
    letter = {"id": "e0", "text": "По P-1 цена 1200 рублей за штуку."}
    reference = {"id": "e1", "text": "По P-1 цена как договаривались в старом письме."}
    system, client = hybrid([EMPTY_REPLY])
    tools = make_tools()
    tools.add_sources([letter, reference])

    system.process_episode(make_context([reference], seen=[letter]), tools)

    assert [(record["name"], record["args"]) for record in tools.tracer.records] == [
        ("read_evidence", {"purchase_id": "P-1", "period": None})
    ]
    # Само сообщение куска модель уже видит в events
    assert client.request(0)["earlier"] == [{"event_id": "e0", "quote": letter["text"]}]


@pytest.mark.parametrize(
    ("text", "link"),
    [
        # Поиск подтвердил закупку: пометки нет
        ("По ремням у Атласа дату согласовали на 22 ноября 2026 года.", None),
        # Искать нечего, а в истории названа одна закупка: привязка выведена кодом
        ("Склад подтвердил 22 ноября 2026 года.", "inferred"),
        # Поиск нашел P-2, а модель взяла P-1: привязку выбрала модель
        ("По сальникам у Вектора дату согласовали на 22 ноября 2026 года.", "model"),
    ],
)
def test_hybrid_marks_inferred_link_in_diff(text: str, link: str | None) -> None:
    story = {"id": "e1", "text": "По P-1 счёт пришёл."}
    event = {"id": "e4", "text": text}
    reply = {"changes": [change("delivery_date", "2026-11-22", event, "22 ноября 2026 года")], "issues": []}
    system, _ = hybrid([json.dumps(reply)])

    output = system.process_episode(make_context([event], seen=[story], state=TWO_SUPPLIERS), make_tools(TWO_SUPPLIERS))

    assert [op.get("link") for proposal in output.proposals for op in proposal["operations"]] == [link]


def test_hybrid_keeps_inferred_mark_when_meeting_closes_in_later_chunk() -> None:
    story = {"id": "e1", "text": "По P-1 счёт пришёл."}
    meeting = {"source_id": "meet", "occurred_at": "2026-11-03T10:15:00+03:00", "source_complete": False}
    first = {**meeting, "id": "m1", "chunk_index": 1, "text": "Склад подтвердил 22 ноября 2026 года."}
    last = {**meeting, "id": "m2", "chunk_index": 2, "text": "На этом всё.", "source_complete": True}
    reply = {"changes": [change("delivery_date", "2026-11-22", first, "22 ноября 2026 года")], "issues": []}
    system, _ = hybrid([json.dumps(reply), EMPTY_REPLY])
    tools = make_tools(TWO_SUPPLIERS)
    system.process_episode(make_context([first], seen=[story], state=TWO_SUPPLIERS), tools)

    final = system.process_episode(
        make_context([last], seen=[story, first], episode="step2", state=TWO_SUPPLIERS), make_tools(TWO_SUPPLIERS)
    )

    # Привязку m1 код вывел в прошлом куске: пометка доходит до предложения при закрытии записи
    assert [(op["field"], op.get("link")) for op in final.proposals[0]["operations"]] == [("delivery_date", "inferred")]


def test_hybrid_holds_change_of_ambiguous_event_and_asks() -> None:
    state = cards_state(("P-1", "Атлас", "Ремни"), ("P-2", "Атлас", "Ремни"))
    # Модель не послушалась hybrid.md и взяла P-1 из двух подходящих карточек, а вопрос задала без кандидатов
    reply = {**json.loads(DATE_REPLY), "issues": [{"code": "ambiguous_purchase", "event_id": "e3"}]}
    system, _ = hybrid([json.dumps(reply)])

    output = system.process_episode(make_context([UNLINKED], state=state), make_tools(state))

    assert output.proposals == []
    assert output.issues == [
        {"code": "ambiguous_purchase", "event_id": "e3", "candidates": ["P-1", "P-2"], "after": "step1"}
    ]


def test_hybrid_infers_link_from_purchase_proposed_earlier_in_story() -> None:
    follow_up = {"id": "e4", "text": "Склад тоже подтвердил."}
    system, client = hybrid([DATE_REPLY, EMPTY_REPLY])
    system.process_episode(make_context([UNLINKED], state=TWO_SUPPLIERS), make_tools(TWO_SUPPLIERS))

    # Номера нет ни в одном сообщении: закупку истории дает предложение прошлого куска
    system.process_episode(
        make_context([follow_up], seen=[UNLINKED], episode="step2", state=TWO_SUPPLIERS), make_tools(TWO_SUPPLIERS)
    )

    assert client.request(1)["links"] == [{"event_id": "e4", "purchase_id": "P-1", "inferred": True}]


def test_hybrid_gives_text_candidates_and_counts_missed_agreed_ones() -> None:
    event = {
        "id": "e1",
        "text": "По P-1 согласовали 8 штук по 125 рублей. Подтвердим по 120 рублей, если оплатим вперёд.",
    }
    system, client = hybrid([EMPTY_REPLY])

    output = system.process_episode(make_context([event], state=BARE), make_tools(BARE))

    # Валюту и единицу выводит код: в подсказках их нет; условная цена тоже подсказка
    assert client.request(0)["candidates"] == [
        {"event_id": "e1", "field": "unit_price", "value": "125.00", "quote": "по 125 рублей"},
        {"event_id": "e1", "field": "quantity", "value": 8, "quote": "8 штук"},
        {"event_id": "e1", "field": "unit_price", "value": "120.00", "quote": "по 120 рублей"},
    ]
    # Пропущенные договоренности код не добавляет, только считает; условная цена не в счете
    assert output.proposals == []
    assert (merge_step(output)["missed_candidate"], merge_step(output)["missed"]) == (
        2,
        [["e1", "unit_price", "125.00"], ["e1", "quantity", 8]],
    )


@pytest.mark.parametrize(
    ("name", "text", "field", "value", "quote", "kept"),
    [
        ("hybrid", "По P-1 берём 8 штук по 125 рублей.", "unit_price", "130.00", "по 125 рублей", False),
        ("hybrid", "По P-1 берём 8 штук по 125 рублей.", "unit_price", "125.00", "по 125 рублей", True),
        # B1 числа с текстом не сверяет
        ("llm_single", "По P-1 берём 8 штук по 125 рублей.", "unit_price", "130.00", "по 125 рублей", True),
        (
            "hybrid",
            "По P-1 поставку согласовали на 22 ноября 2026 года.",
            "delivery_date",
            "2026-11-23",
            "22 ноября 2026 года",
            False,
        ),
        # Правила цену без "по" и "за штуку" не берут, но число есть в сообщении
        ("hybrid", "По P-1 теперь 900 рублей.", "unit_price", "900.00", "теперь 900 рублей", True),
        # Без слова о поставке правила дату не берут, но она дословно в цитате
        (
            "hybrid",
            "По P-1 крайний день 22 ноября 2026 года.",
            "delivery_date",
            "2026-11-22",
            "22 ноября 2026 года",
            True,
        ),
        # Число словами правила не разбирают: код его не проверяет
        ("hybrid", "По P-1 берём восемь штук.", "quantity", 8, "восемь штук", True),
        # Слово "одну" или "тысяч" рядом с цифрами сверку не выключает: 130 и 13000 в тексте нет
        (
            "hybrid",
            "По P-1 берём 8 штук по 125 рублей за одну штуку.",
            "unit_price",
            "130.00",
            "по 125 рублей за одну штуку",
            False,
        ),
        ("hybrid", "По P-1 берём по 12 тысяч рублей.", "unit_price", "13000.00", "по 12 тысяч рублей", False),
        ("hybrid", "По P-1 берём по 12 тысяч рублей.", "unit_price", "12000.00", "по 12 тысяч рублей", True),
    ],
)
def test_hybrid_keeps_only_numbers_found_in_text(
    name: str, text: str, field: str, value: object, quote: str, kept: bool
) -> None:
    event = {"id": "e1", "text": text}
    reply = {"changes": [change(field, value, event, quote)], "issues": []}
    client = FakeClient([json.dumps(reply)])
    system = LlmSystem(name, use_tools=False, client=client, code_routing=name == "hybrid")

    output = system.process_episode(make_context([event], state=BARE), make_tools(BARE))

    values = [op["value"] for proposal in output.proposals for op in proposal["operations"] if op["field"] == field]
    assert values == ([value] if kept else [])


def test_hybrid_number_from_neighbour_message_of_chunk_is_grounded() -> None:
    price = {"id": "e1", "text": "По P-1 берём по 125 рублей."}
    invoice = {"id": "e2", "text": "По P-1 счёт подходит."}
    reply = {"changes": [change("unit_price", "125.00", invoice, "счёт подходит")], "issues": []}
    system, _ = hybrid([json.dumps(reply)])

    output = system.process_episode(make_context([price, invoice], state=BARE), make_tools(BARE))

    assert [op["value"] for op in output.proposals[0]["operations"] if op["field"] == "unit_price"] == ["125.00"]
    assert merge_step(output)["dropped_ungrounded"] == 0


def test_hybrid_meeting_acceptance_without_number_is_grounded_by_draft() -> None:
    offer = {**OFFER_CHUNK, "occurred_at": "2026-11-03T10:15:00+03:00", "chunk_index": 2}
    accept = {
        "id": "m8",
        "source_id": "meet",
        "occurred_at": "2026-11-03T11:10:00+03:00",
        "chunk_index": 8,
        "text": "По P-1 на этом сошлись.",
    }
    state = {
        "purchases": {"P-1": {"revision": 1, "fields": {"unit_price": "13000.00", "prepayment_percent": 0}}},
        "tasks": {},
    }
    first = {
        "changes": [
            change("unit_price", "12000.00", offer, "по 12000 рублей", "conditional"),
            change("prepayment_percent", 100, offer, "при полной предоплате", "conditional"),
        ],
        "issues": [],
    }
    second = {"changes": [change("unit_price", "12000.00", accept, "на этом сошлись")], "issues": []}
    system, _ = hybrid([json.dumps(first), json.dumps(second)])
    chunk = {**offer, "source_complete": False}
    system.process_episode(make_context([chunk], episode="step3", state=state), make_tools(state))

    final = system.process_episode(
        make_context([{**accept, "source_complete": True}], seen=[offer], episode="step4", state=state),
        make_tools(state),
    )

    assert [(op["field"], op["value"]) for op in final.proposals[0]["operations"]] == [
        ("prepayment_percent", 100),
        ("unit_price", "12000.00"),
    ]


def test_hybrid_find_timeout_holds_unlinked_change() -> None:
    fault = [{"at": "step1", "operation": "find_purchase", "fault": "timeout", "once": True}]
    system, client = hybrid([DATE_REPLY])
    tools = make_tools(faults=fault)

    output = system.process_episode(make_context([UNLINKED]), tools)

    assert [(record["name"], record["error"]) for record in tools.tracer.records] == [("find_purchase", "timeout")]
    assert len(client.payloads) == 1
    assert output.proposals == []
    assert output.stop_reason == StopReason.TOOL_ERROR
    assert output.issues == [{"code": "tool_timeout", "event_id": "e3", "purchase_id": "P-1", "after": "step1"}]


def test_hybrid_does_not_call_tools_over_budget() -> None:
    state = cards_state(("P-1", "Атлас", "Ремни"), ("P-2", "Вектор", "Сальники"), ("P-3", "Орбита", "Клапаны"))
    events = [
        {"id": f"e{index}", "text": f"{item} приедут завтра."}
        for index, item in enumerate(["Ремни", "Сальники", "Клапаны"])
    ]
    valves = change("prepayment_percent", 100, events[2], "Клапаны приедут завтра")
    system, _ = hybrid([json.dumps({"changes": [{**valves, "purchase_id": "P-3"}], "issues": []})])
    tools = make_tools(state)

    output = system.process_episode(make_context(events, state=state), tools)

    # Лимит два вызова: третий поиск код не делает и пишет пропущенным, а не получает отказ харнесса
    assert [(record["args"]["query"], record["error"]) for record in tools.tracer.records] == [
        ("ремни", None),
        ("сальники", None),
    ]
    assert route_step(output)["skipped"] == [["find_purchase", "клапаны"]]
    assert output.stop_reason == StopReason.DONE
    # Поиск не подтвердил P-3: привязку модели закупщик видит в диффе
    assert [(op["purchase_id"], op.get("link")) for op in output.proposals[0]["operations"]] == [("P-3", "model")]


@pytest.mark.parametrize(
    ("text", "quote", "expected"),
    [
        (
            "Договорились на 24 ноября 2026 года.",
            "Договорились на 24 ноября 2026 года",
            "Договорились на 24 ноября 2026 года.",
        ),
        ("Счет пришел. Берем по 2.5 тыс за штуку! Доставка завтра", "по 2.5 тыс", "Берем по 2.5 тыс за штуку!"),
        # Цитата уже кончается точкой: следующее предложение не захватывается
        ("Цена 900 рублей. Срок 5 дней.", "Цена 900 рублей.", "Цена 900 рублей."),
        ("Срок 5 дней, без НДС", "без НДС", "Срок 5 дней, без НДС"),
        # Цитата начинается внутри числа: точка в "2.5" не граница предложения
        ("Берем по 2.5 тыс за штуку!", "5 тыс", "Берем по 2.5 тыс за штуку!"),
    ],
)
def test_whole_sentence_extends_quote_to_sentence(text, quote, expected):
    events = {"e1": {"id": "e1", "text": text}}
    assert whole_sentence({"event_id": "e1", "quote": quote}, events) == {"event_id": "e1", "quote": expected}

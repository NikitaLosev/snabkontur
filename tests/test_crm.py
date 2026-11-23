"""Свойства mock-CRM: идемпотентность, устаревшая ревизия, откат и атомарность."""

import pytest

from snab.crm import MockCrm, morning_summary

INITIAL_STATE = {
    "purchases": {"P-1": {"revision": 1, "fields": {"budget": "5000000.00", "delivery_date": "2026-10-10"}}},
    "tasks": {},
}
SET_BUDGET = {
    "op": "set_field",
    "purchase_id": "P-1",
    "field": "budget",
    "value": "4000000.00",
    "evidence": [{"event_id": "e1", "quote": "бюджет 4 млн"}],
}
SET_UNIT = {"op": "set_field", "purchase_id": "P-1", "field": "unit", "value": "шт", "evidence": []}
CREATE_TASK = {
    "op": "create_task",
    "evidence": [],
    "task": {
        "id": "T-1",
        "purchase_id": "P-1",
        "title": "Запросить счёт",
        "owner": "Анна",
        "due_date": "2026-10-06",
        "status": "open",
    },
}


def set_date(value: str) -> dict:
    """Операция смены срока поставки."""
    return {"op": "set_field", "purchase_id": "P-1", "field": "delivery_date", "value": value, "evidence": []}


@pytest.fixture
def crm() -> MockCrm:
    """CRM с одной карточкой P-1 ревизии 1."""
    crm = MockCrm()
    crm.load(INITIAL_STATE)
    return crm


def test_repeated_change_id_writes_once(crm: MockCrm) -> None:
    first = crm.apply([SET_BUDGET, CREATE_TASK], "c1", {"P-1": 1})
    repeat = crm.apply([SET_BUDGET, CREATE_TASK], "c1", {"P-1": 1})

    assert first.data["status"] == "applied"
    assert repeat.data["status"] == "already_applied"
    assert crm.snapshot()["purchases"]["P-1"]["revision"] == 2
    assert list(crm.snapshot()["tasks"]) == ["T-1"]


def test_stale_revision_is_typed_conflict_without_write(crm: MockCrm) -> None:
    crm.apply([set_date("2026-10-12")], "c1", {"P-1": 1})
    before = crm.snapshot()

    result = crm.apply([SET_BUDGET], "c2", {"P-1": 1})

    assert result.error.code == "conflict"
    assert result.error.details == {"purchase_id": "P-1", "current_revision": 2}
    assert crm.snapshot() == before


def test_undo_restores_values_as_new_revision(crm: MockCrm) -> None:
    crm.apply([SET_BUDGET, SET_UNIT, CREATE_TASK], "c1", {"P-1": 1})

    result = crm.undo("c1", "u1")

    assert result.data["status"] == "undone"
    assert crm.snapshot() == {
        "purchases": {"P-1": {"revision": 3, "fields": {"budget": "5000000.00", "delivery_date": "2026-10-10"}}},
        "tasks": {},
    }


def test_undo_blocked_by_later_edit_of_same_field(crm: MockCrm) -> None:
    crm.apply([SET_BUDGET], "c1", {"P-1": 1})
    crm.apply([dict(SET_BUDGET, value="4500000.00")], "m1", {"P-1": 2}, kind="manual_edit")
    before = crm.snapshot()

    result = crm.undo("c1", "u1")

    assert result.error.code == "conflict"
    assert result.error.details == {"purchase_id": "P-1", "fields": ["budget"]}
    assert crm.snapshot() == before


def test_undo_not_blocked_by_later_edit_of_other_field(crm: MockCrm) -> None:
    crm.apply([SET_BUDGET], "c1", {"P-1": 1})
    crm.apply([set_date("2026-10-20")], "m1", {"P-1": 2}, kind="manual_edit")

    crm.undo("c1", "u1")

    assert crm.snapshot()["purchases"]["P-1"] == {
        "revision": 4,
        "fields": {"budget": "5000000.00", "delivery_date": "2026-10-20"},
    }


def test_undo_allowed_after_later_change_was_undone(crm: MockCrm) -> None:
    crm.apply([set_date("2026-10-12")], "c1", {"P-1": 1})
    crm.apply([set_date("2026-10-20")], "c2", {"P-1": 2})
    crm.undo("c2", "u2")

    result = crm.undo("c1", "u1")

    assert result.data["status"] == "undone"
    assert crm.snapshot()["purchases"]["P-1"]["fields"]["delivery_date"] == "2026-10-10"


def test_create_task_with_existing_id_is_refused_and_keeps_task(crm: MockCrm) -> None:
    crm.apply([CREATE_TASK], "c1", {"P-1": 1})
    before = crm.snapshot()

    result = crm.apply([dict(CREATE_TASK, task=dict(CREATE_TASK["task"], title="Другое"))], "c2", {"P-1": 2})

    assert result.error.code == "invalid_args"
    assert crm.snapshot() == before


def test_failure_after_first_operation_rolls_back_and_retry_applies(crm: MockCrm) -> None:
    failed = crm.apply([SET_BUDGET, set_date("2026-10-12")], "c1", {"P-1": 1}, fail_after_first_operation=True)
    after_failure = crm.snapshot()
    retried = crm.apply([SET_BUDGET, set_date("2026-10-12")], "c1", {"P-1": 1})

    assert failed.error.code == "write_failed" and failed.error.retryable
    assert after_failure == INITIAL_STATE
    assert retried.data["status"] == "applied"
    assert crm.snapshot()["purchases"]["P-1"] == {
        "revision": 2,
        "fields": {"budget": "4000000.00", "delivery_date": "2026-10-12"},
    }


def test_write_drops_summary_and_rebuild_cites_applied_evidence() -> None:
    crm = MockCrm()
    summary = {"purchase_id": "P-1", "based_on_revision": 1, "facts": {"budget": "5000000.00"}, "text": "5 млн"}
    stale = crm.load(INITIAL_STATE, {"summary": summary})

    applied = crm.apply([SET_BUDGET], "c1", {"P-1": 1})
    dropped = crm.derived_views()
    crm.rebuild_summary("P-1")

    assert stale == []
    assert applied.data["invalidated_views"] == ["summary"]
    assert dropped == {"summary": None}
    assert crm.derived_views() == {
        "summary": {
            "purchase_id": "P-1",
            "based_on_revision": 2,
            "facts": {"budget": "4000000.00"},
            "evidence": [{"event_id": "e1", "quote": "бюджет 4 млн"}],
        }
    }


def test_morning_summary_window_and_statuses() -> None:
    task = {"purchase_id": "P-1", "title": "t", "owner": "Анна"}
    state = {
        "purchases": {
            "P-1": {"revision": 1, "fields": {"delivery_date": "2026-11-07"}},
            "P-2": {"revision": 1, "fields": {"delivery_date": "2026-11-08"}},
        },
        "tasks": {
            "late": {**task, "id": "late", "due_date": "2026-11-04", "status": "open"},
            "today": {**task, "id": "today", "due_date": "2026-11-05", "status": "open"},
            "edge": {**task, "id": "edge", "due_date": "2026-11-07", "status": "open"},
            "after": {**task, "id": "after", "due_date": "2026-11-08", "status": "open"},
            "done": {**task, "id": "done", "due_date": "2026-11-04", "status": "done"},
            "undated": {**task, "id": "undated", "due_date": None, "status": "open"},
        },
    }

    assert morning_summary(state, "2026-11-05T08:00:00+03:00") == {
        "window_start": "2026-11-05",
        "window_end": "2026-11-07",
        "overdue_task_ids": ["late"],
        "upcoming_task_ids": ["edge", "today"],
        "unscheduled_task_ids": ["undated"],
        "deliveries": [{"purchase_id": "P-1", "delivery_date": "2026-11-07"}],
    }

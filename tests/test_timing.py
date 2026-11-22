"""Замер времени: группы и контрбаланс режимов, правильность по gold, активное время и сводка."""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from snab.timing import (
    MANUAL,
    REVIEW,
    TimingConfig,
    assign_cases,
    assigned_mode,
    build_server,
    gold_operations,
    main,
    parse_args,
    score_operations,
    suggested_group,
    visible_events,
)

CASE_IDS = ("C1", "C2", "C3", "C4")
SAVED_AT = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
TASK = {"id": "T-1", "purchase_id": "P-1", "title": "Подтвердить приёмку", "owner": "Анна", "due_date": "2026-10-06"}
GOLD_OPS = [
    {"op": "set_field", "purchase_id": "P-1", "field": "unit_price", "value": "90.00", "evidence": []},
    {"op": "set_field", "purchase_id": "P-1", "field": "delivery_date", "value": "2026-10-15", "evidence": []},
    {"op": "create_task", "task": dict(TASK, status="open"), "evidence": []},
]
SHORT_CASE = {
    "id": "C1",
    "input": {
        "initial_state": {
            "purchases": {"P-1": {"revision": 1, "fields": {"unit_price": "100.00", "delivery_date": "2026-10-10"}}},
            "tasks": {},
        },
        "events": [{"id": "e1", "occurred_at": "2026-10-04T12:00:00+03:00", "text": "По P-1 цена 90, срок 15.10."}],
        "user_actions": [],
    },
    "gold": {"proposals": [{"id": "p1", "based_on": {"P-1": 1}, "operations": GOLD_OPS}]},
}


def episode(number: int, actions: list[str]) -> dict:
    """Эпизод timeline с одним источником и действиями заданных видов."""
    return {
        "episode_id": f"step{number}",
        "events": [{"id": f"e{number}", "occurred_at": "2026-11-03T10:00:00+03:00", "text": f"шаг {number}"}],
        "user_actions": [{"id": f"a{number}{kind}", "kind": kind} for kind in actions],
    }


def set_price(value: str, step: str) -> dict:
    """Предложение gold из одной операции цены, доступное после шага."""
    op = {"op": "set_field", "purchase_id": "P-1", "field": "unit_price", "value": value, "evidence": []}
    return {"id": f"p-{step}", "available_after": step, "based_on": {"P-1": 1}, "operations": [op]}


TIMELINE_CASE = {
    "id": "C2",
    "input": {
        "initial_state": SHORT_CASE["input"]["initial_state"],
        "timeline": [episode(1, ["reject"]), episode(2, ["approve"]), episode(3, ["approve", "query"])],
    },
    "gold": {"proposals": [set_price("95.00", "step2"), set_price("80.00", "step3")]},
}


@pytest.mark.parametrize(
    ("group", "modes"),
    [("A", [MANUAL, REVIEW, MANUAL, REVIEW]), ("B", [REVIEW, MANUAL, REVIEW, MANUAL])],
)
def test_group_alternates_modes_from_its_own_start(group: str, modes: list[str]) -> None:
    assert assign_cases(group, CASE_IDS) == [
        {"case_id": case_id, "mode": mode} for case_id, mode in zip(CASE_IDS, modes, strict=True)
    ]


def test_groups_a_and_b_take_opposite_modes_on_every_case() -> None:
    pairs = zip(assign_cases("A", CASE_IDS), assign_cases("B", CASE_IDS), strict=True)

    assert all({a["mode"], b["mode"]} == {MANUAL, REVIEW} for a, b in pairs)


@pytest.mark.parametrize(
    ("participant", "group"),
    # Первый байт sha256: «аня» 81 и «борис» 189 оба нечетные, поэтому без явного выбора
    # они попали бы в одну группу; «ева» 138 четный
    [("Аня", "B"), ("  аня ", "B"), ("Борис", "B"), ("Ева", "A")],
)
def test_suggested_group_follows_name_hash_parity(participant: str, group: str) -> None:
    assert suggested_group(participant) == group


def test_assigned_mode_rejects_case_outside_list() -> None:
    with pytest.raises(ValueError, match="not assigned"):
        assigned_mode("A", CASE_IDS, "C9")


def test_timeline_stops_at_first_approve() -> None:
    assert [event["id"] for event in visible_events(TIMELINE_CASE)] == ["e1", "e2"]
    assert [op["value"] for op in gold_operations(TIMELINE_CASE)] == ["95.00"]


def test_score_matches_money_dates_and_task_owner_by_hand() -> None:
    submitted = [
        {"op": "set_field", "purchase_id": "P-1", "field": "unit_price", "value": "90"},
        {"op": "set_field", "purchase_id": "P-1", "field": "delivery_date", "value": "2026-10-15"},
        {"op": "set_field", "purchase_id": "P-1", "field": "currency", "value": "USD"},
    ]

    score = score_operations(submitted, GOLD_OPS)

    # 2 из 3 предложенных верны, задача Анны пропущена
    assert (score["tp"], score["fp"], score["fn"]) == (2, 1, 1)
    assert score["precision"] == pytest.approx(2 / 3)
    assert score["recall"] == pytest.approx(2 / 3)


def test_task_matches_on_owner_and_due_date_not_title() -> None:
    task = {"purchase_id": "P-1", "title": "Приёмку подтвердить", "owner": "анна", "due_date": "2026-10-06"}

    score = score_operations([{"op": "create_task", "task": task}], GOLD_OPS[2:])

    assert (score["tp"], score["fp"], score["fn"]) == (1, 0, 0)


@pytest.fixture
def server_url(tmp_path: Path) -> Iterator[str]:
    config = TimingConfig(
        cases={"C1": SHORT_CASE, "C2": TIMELINE_CASE},
        predictions={},
        case_ids=("C1", "C2"),
        out_dir=tmp_path,
        predictions_path="runs/fake/predictions.jsonl",
        clock=lambda: SAVED_AT,
    )
    server = build_server(config, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def call(request: urllib.request.Request | str) -> tuple[int, dict]:
    """Запрос к серверу -> HTTP-статус и JSON ответа, в том числе ошибочного."""
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def post_result(url: str, payload: dict) -> int:
    """POST /result -> HTTP-статус ответа."""
    request = urllib.request.Request(
        f"{url}/result", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    return call(request)[0]


def get_json(url: str, path: str, **query: str) -> tuple[int, dict]:
    """GET path с параметрами -> HTTP-статус и JSON ответа."""
    return call(f"{url}{path}?{urllib.parse.urlencode(query)}")


def browser_clock(*seconds: int) -> list[str]:
    """Секунды от 10:00:00 -> ISO-метки, какие шлет страница."""
    return [datetime(2026, 11, 22, 10, s // 60, s % 60, tzinfo=UTC).isoformat() for s in seconds]


def attempt_payload(pauses: list[dict], participant: str = "Ева", group: str = "A", case_id: str = "C1") -> dict:
    """Сдача кейса с ценой 90 и датой в формате ДД.ММ.ГГГГ."""
    started, finished = browser_clock(0, 120)
    operations = [
        {"op": "set_field", "purchase_id": "P-1", "field": "unit_price", "value": "90"},
        {"op": "set_field", "purchase_id": "P-1", "field": "delivery_date", "value": "15.10.2026"},
    ]
    return {
        "participant": participant,
        "group": group,
        "case_id": case_id,
        "started_at": started,
        "finished_at": finished,
        "pauses": pauses,
        "operations": operations,
        "review_rows": [],
    }


def test_result_record_excludes_pauses_and_scores_against_gold(server_url: str, tmp_path: Path) -> None:
    paused, resumed = browser_clock(30, 60)

    status = post_result(server_url, attempt_payload([{"paused_at": paused, "resumed_at": resumed}]))

    [record] = [json.loads(line) for line in (tmp_path / "Ева.jsonl").read_text(encoding="utf-8").splitlines()]
    assert status == 200
    assert (record["active_seconds"], record["wall_seconds"]) == (90, 120)
    # Группа A начинает с ручного ввода
    assert (record["group"], record["mode"]) == ("A", MANUAL)
    assert record["operations"][1]["value"] == "2026-10-15"
    assert (record["correctness"]["tp"], record["correctness"]["fn"]) == (2, 1)
    assert record["saved_at"] == SAVED_AT.isoformat()


def test_second_attempt_on_same_case_is_rejected(server_url: str, tmp_path: Path) -> None:
    statuses = [post_result(server_url, attempt_payload([])) for _ in range(2)]

    assert statuses == [200, 409]
    assert len((tmp_path / "Ева.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_pause_outside_attempt_is_rejected(server_url: str, tmp_path: Path) -> None:
    paused, resumed = browser_clock(100, 200)

    assert post_result(server_url, attempt_payload([{"paused_at": paused, "resumed_at": resumed}])) == 400
    assert not (tmp_path / "Ева.jsonl").exists()


def test_same_parity_names_can_choose_different_groups(server_url: str, tmp_path: Path) -> None:
    statuses = [
        post_result(server_url, attempt_payload([], "Аня", "A")),
        post_result(server_url, attempt_payload([], "Борис", "B")),
    ]

    records = [json.loads((tmp_path / f"{name}.jsonl").read_text(encoding="utf-8")) for name in ("Аня", "Борис")]
    assert statuses == [200, 200]
    assert [(r["group"], r["mode"]) for r in records] == [("A", MANUAL), ("B", REVIEW)]


def test_group_is_locked_after_first_submission(server_url: str, tmp_path: Path) -> None:
    assert get_json(server_url, "/api/group", participant="Ева") == (200, {"suggested": "A", "locked": None})
    post_result(server_url, attempt_payload([], group="B"))

    # Смена группы дала бы второй режим тем же кейсам: C1 в B проверка, в A ручной ввод
    assert get_json(server_url, "/api/group", participant="Ева") == (200, {"suggested": "A", "locked": "B"})
    assert get_json(server_url, "/api/cases", participant="Ева", group="A")[0] == 400
    assert get_json(server_url, "/api/case", participant="Ева", group="A", id="C2")[0] == 400
    assert post_result(server_url, attempt_payload([], group="A", case_id="C2")) == 400
    status, view = get_json(server_url, "/api/case", participant="Ева", group="B", id="C2")
    assert (status, view["mode"]) == (200, MANUAL)
    assert len((tmp_path / "Ева.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_record_without_group_locks_hash_group(server_url: str, tmp_path: Path) -> None:
    # Запись до выбора группы: тогда режим решала четность хэша, у «Ева» это группа A
    write_jsonl(tmp_path / "Ева.jsonl", [timing_record("Ева", MANUAL, 60, 1, 0, 0)])

    assert get_json(server_url, "/api/group", participant="Ева")[1]["locked"] == "A"
    assert get_json(server_url, "/api/cases", participant="Ева", group="B")[0] == 400


@pytest.mark.parametrize("group", ["", "C", "a"])
def test_unknown_group_is_rejected(server_url: str, group: str) -> None:
    status, body = get_json(server_url, "/api/cases", participant="Ева", group=group)

    assert status == 400
    assert "group must be one of" in body["error"]


def timing_record(
    participant: str, mode: str, seconds: float, tp: int, fp: int, fn: int, group: str = "A", case_id: str = "C1"
) -> dict:
    """Минимальная запись замера для сводки."""
    correctness = {"tp": tp, "fp": fp, "fn": fn}
    return {
        "participant": participant,
        "case_id": case_id,
        "group": group,
        "mode": mode,
        "active_seconds": seconds,
        "correctness": correctness,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def test_summary_aggregates_modes_across_files(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "a.jsonl", [timing_record("a", MANUAL, 100, 2, 0, 1), timing_record("a", REVIEW, 30, 1, 1, 0)]
    )
    write_jsonl(
        tmp_path / "b.jsonl",
        [timing_record("b", MANUAL, 200, 3, 1, 0, group="B"), timing_record("b", MANUAL, 120, 1, 0, 0, group="B")],
    )

    main(["--summary", "--out-dir", str(tmp_path)])

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    manual, review = summary["modes"][MANUAL], summary["modes"][REVIEW]
    assert (manual["n"], manual["participants"], manual["median_active_seconds"]) == (3, 2, 120)
    assert manual["mean_active_seconds"] == pytest.approx(140)
    # manual: tp 6, fp 1, fn 1; точно без ошибок только третья запись
    assert (manual["precision"], manual["recall"]) == (pytest.approx(6 / 7), pytest.approx(6 / 7))
    assert manual["exact_share"] == pytest.approx(1 / 3)
    assert (review["precision"], review["recall"]) == (0.5, 1.0)
    assert (summary["groups"], summary["balanced"]) == ({"A": 1, "B": 1}, True)
    assert summary["review_to_manual_median"] == pytest.approx(0.25)


def test_summary_without_counterbalance_gives_no_ratio(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Все в группе A: C1 только вручную, C2 только проверкой, режим смешан с кейсом
    records = [
        timing_record(name, mode, seconds, 1, 0, 0, case_id=case_id)
        for name in ("a", "c")
        for mode, seconds, case_id in ((MANUAL, 100, "C1"), (REVIEW, 30, "C2"))
    ]
    write_jsonl(tmp_path / "a.jsonl", records)

    main(["--summary", "--out-dir", str(tmp_path)])

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert (summary["groups"], summary["cases_in_one_mode"]) == ({"A": 2, "B": 0}, ["C1", "C2"])
    assert summary["balanced"] is False
    assert "review_to_manual_median" not in summary
    assert "отношение медиан не считается; кейсы в одном режиме: C1, C2" in capsys.readouterr().out


def test_cli_refuses_held_out_test_split() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--cases", "eval/test.jsonl", "--predictions", "x.jsonl"])

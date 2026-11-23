"""B0 на шести кейсах dev от CLI до отчета оценщика."""

import json
from pathlib import Path

import pytest

from snab.run import main, parse_args

SMOKE_IDS = ("E04", "E05", "E06", "D-SUM-01", "D-WOR-16", "D-WOR-17")


def read_jsonl(path: Path) -> list[dict]:
    """Строки JSONL файла прогона."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Папка одного прогона B0 на SMOKE_IDS, общая для тестов модуля."""
    out = tmp_path_factory.mktemp("smoke")
    ids = out / "ids.txt"
    ids.write_text("\n".join(SMOKE_IDS), encoding="utf-8")
    main(["--system", "rules", "--split", "dev", "--ids", str(ids), "--out", str(out / "run")])
    return out / "run"


def test_report_covers_exactly_selected_cases(run_dir: Path) -> None:
    report = json.loads((run_dir / "eval-report.json").read_text(encoding="utf-8"))

    assert sorted(report["selection"]["selected_ids"]) == sorted(SMOKE_IDS)
    assert report["coverage"]["missing_cases"] == []
    assert report["coverage"]["invalid_lines"] == []


def test_crm_chains_of_simple_cases_match_gold(run_dir: Path) -> None:
    report = json.loads((run_dir / "eval-report.json").read_text(encoding="utf-8"))
    chains = {case["id"]: case["metrics"]["crm_chains"]["numerator"] for case in report["cases"]}

    assert {case_id: chains[case_id] for case_id in ("E04", "E05", "E06", "D-SUM-01")} == dict.fromkeys(
        ("E04", "E05", "E06", "D-SUM-01"), 1
    )


def test_repeated_approve_writes_once(run_dir: Path) -> None:
    writes = [
        record["result"]["status"]
        for record in read_jsonl(run_dir / "trace.jsonl")
        if record["case_id"] == "E04" and record["kind"] == "crm"
    ]

    assert writes == ["applied", "already_applied"]


def test_injected_search_timeout_reported_without_write(run_dir: Path) -> None:
    trace = read_jsonl(run_dir / "trace.jsonl")
    prediction = next(row for row in read_jsonl(run_dir / "predictions.jsonl") if row["id"] == "D-WOR-16")
    revisions = {
        check["after"]: check["expected_state"]["purchases"]["P-D716"]["revision"]
        for check in prediction["gold"]["checks"]
    }

    timeouts = [record for record in trace if record["case_id"] == "D-WOR-16" and record["error"] == "timeout"]
    assert [(record["episode"], record["name"]) for record in timeouts] == [("D-WOR-16-step3", "find_purchase")]
    assert {"code": "tool_timeout", "after": "D-WOR-16-step3", "event_id": "D-WOR-16-e3"} in prediction["gold"][
        "expected_issues"
    ]
    assert revisions["D-WOR-16-step3"] == 1


def test_run_files_describe_the_run(run_dir: Path) -> None:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert meta["system"] == "rules"
    assert meta["selected_case_ids"] == list(SMOKE_IDS)
    assert set(meta["packages"]) == {"pydantic", "langchain-openai"}
    assert len(meta["scorer_sha256"]) == 64
    assert "snab/run.py" in meta["code_sha256"]
    assert meta["budgets"]["max_model_calls"] == 3
    assert summary["cases"] == len(SMOKE_IDS)
    assert list(summary["unsupported_fault_injections"]) == ["D-WOR-17"]


def test_case_with_unreached_fault_is_unsupported_in_report(run_dir: Path) -> None:
    report = json.loads((run_dir / "eval-report.json").read_text(encoding="utf-8"))
    unsupported = {case["id"]: case["unsupported"] for case in report["cases"]}

    assert unsupported == {case_id: case_id == "D-WOR-17" for case_id in SMOKE_IDS}
    assert list(report["unsupported_fault_injections"]) == ["D-WOR-17"]


def test_heldout_split_requires_explicit_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--system", "rules", "--split", "test"])


def test_nonempty_out_is_refused(tmp_path: Path) -> None:
    # Старый eval-report.json рядом с новой трассой смешал бы два прогона в одном отчете
    (tmp_path / "eval-report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(["--system", "rules", "--out", str(tmp_path)])


def test_cases_outside_repository_are_refused(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(["--system", "rules", "--cases", str(cases)])


def test_tool_timeout_stops_rules_like_llm_systems(run_dir: Path) -> None:
    # D-WOR-16: поиск не ответил; B0 и LLM-системы пишут один stop_reason по контракту раздела 5
    trace = [row for row in read_jsonl(run_dir / "trace.jsonl") if row.get("case_id") == "D-WOR-16"]
    assert "tool_error" in {row.get("stop_reason") for row in trace}

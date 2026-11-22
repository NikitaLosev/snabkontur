"""Отчет по прогонам: статус кейса, micro против macro, Вилсон, срезы, сравнение и CLI."""

import json
from pathlib import Path

import pytest

from snab import run as runner
from snab.report import (
    ALL_LABEL,
    FAILED,
    HISTORY_LABEL,
    PASSED,
    PENDING,
    SHORT_LABEL,
    aggregate,
    build_report,
    case_status,
    load_run,
    main,
    wilson,
)

DATASET_SHA = "a" * 64
PENDING_EVIDENCE = {"type": "unsupported_or_pending_evidence"}
MISSING_FACT = {"type": "missing_fact"}


def metric(good: int, total: int, pending: int = 0) -> dict:
    """Метрика кейса в формате eval/score.py."""
    return {"numerator": good, "denominator": total, "pending": pending}


def make_case(
    case_id: str,
    category: str,
    recall: tuple[int, int] = (1, 1),
    precision: tuple[int, int] = (1, 1),
    chains: tuple[int, int] = (1, 1),
    details: list[dict] | None = None,
    unsupported: bool = False,
    extra_answers: int = 0,
) -> dict:
    """Кейс eval-report: без details и не unsupported он полностью совпал с gold."""
    details = details or []
    return {
        "id": case_id,
        "category": category,
        "metrics": {
            "diff_precision": metric(*precision),
            "diff_recall": metric(*recall),
            "conflict_recall": metric(0, 0),
            "crm_chains": metric(*chains),
            "counts": {"extra_answers": extra_answers, "duplicate_answers": 0},
        },
        "details": details,
        "fully_matches_structured_gold": not details and not unsupported and not extra_answers,
        "unsupported": unsupported,
    }


def trace_record(case_id: str, kind: str, name: str, latency_ms: float, tokens: int = 0, repeat: int = 1) -> dict:
    """Строка trace.jsonl с полями, которые читает отчет."""
    return {
        "case_id": case_id,
        "repeat": repeat,
        "kind": kind,
        "name": name,
        "latency_ms": latency_ms,
        "tokens_in": tokens,
        "tokens_out": 0,
    }


def write_run(path: Path, cases: list[dict], trace: list[dict] | None = None, dataset: str = DATASET_SHA) -> Path:
    """Папка прогона с eval-report.json, summary.json, meta.json и trace.jsonl."""
    path.mkdir(parents=True)
    (path / "eval-report.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    summary = {
        "wall_time_s": 1.0,
        "episode_latency_ms": {"p50": 1.0, "p95": 2.0},
        "case_latency_ms": {"p50": 3.0, "p95": 4.0},
        "llm_calls": 0,
        "tool_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "peak_rss_mb": 60.0,
        "cost": {"local_cpu_s": 0.5},
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    meta = {
        "system": "rules",
        "model": None,
        "git_sha": "0" * 40,
        "git_dirty": False,
        "dataset_version": ["v1"],
        "dataset_sha256_lf": dataset,
        "repeat": 1,
    }
    (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = [json.dumps(record) for record in trace or []]
    (path / "trace.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return path


def slice_of(report_run: dict, label: str) -> dict:
    """Срез прогона из report.json по названию."""
    return next(item for item in report_run["slices"] if item["slice"] == label)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (make_case("C", "crm"), PASSED),
        (make_case("C", "crm", details=[PENDING_EVIDENCE, PENDING_EVIDENCE]), PENDING),
        (make_case("C", "crm", details=[PENDING_EVIDENCE, MISSING_FACT]), FAILED),
        # Инъекция сбоя не дошла до системы: кейс провален, даже если факты сошлись
        (make_case("C", "workflow", unsupported=True), FAILED),
        (make_case("C", "summary", extra_answers=1), FAILED),
    ],
)
def test_case_status(case: dict, expected: str) -> None:
    assert case_status(case) == expected


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [(5, 10, (0.2366, 0.7634)), (0, 10, (0.0, 0.2775)), (10, 10, (0.7225, 1.0))],
)
def test_wilson_matches_reference_values(successes: int, total: int, expected: tuple[float, float]) -> None:
    assert wilson(successes, total) == pytest.approx(list(expected), abs=1e-4)


def test_wilson_without_observations_is_undefined() -> None:
    assert wilson(0, 0) is None


def test_long_history_dominates_micro_but_not_macro(tmp_path: Path) -> None:
    cases = [
        make_case("D-MEM-01", "memory", recall=(1, 10), details=[MISSING_FACT]),
        make_case("E01", "extraction", recall=(1, 1)),
        make_case("E02", "extraction", recall=(1, 1)),
        # Кейс без фактов в диффе не входит в macro
        make_case("D-SUM-01", "summary", recall=(0, 0)),
    ]
    run = load_run(write_run(tmp_path / "a", cases))

    result = aggregate(list(run.cases.values()))

    assert result["diff_recall_micro"]["value"] == pytest.approx(3 / 12)
    assert result["diff_recall_macro"] == {"value": pytest.approx((0.1 + 1 + 1) / 3), "cases": 3}
    assert result["facts"] == 12
    assert result["task_success"]["passed"] == 3


def test_slices_split_short_cases_histories_and_each_long_history(tmp_path: Path) -> None:
    cases = [
        make_case("E01", "extraction"),
        make_case("D-WOR-01", "workflow"),
        make_case("D-MEM-01", "memory"),
        make_case("D-MEM-02", "memory", details=[MISSING_FACT]),
    ]
    report = build_report([load_run(write_run(tmp_path / "a", cases))])
    run = report["runs"][0]

    assert [item["slice"] for item in run["slices"]] == [
        "extraction", "workflow", "memory", SHORT_LABEL, HISTORY_LABEL, ALL_LABEL, "D-MEM-01", "D-MEM-02",
    ]  # fmt: skip
    assert slice_of(run, SHORT_LABEL)["cases"] == 1
    assert slice_of(run, HISTORY_LABEL)["task_success"]["passed"] == 2
    assert slice_of(run, "D-MEM-02")["task_success"]["passed"] == 0
    assert all(item["small_sample"] for item in run["slices"])


def test_cost_by_slice_reads_first_repeat_of_trace(tmp_path: Path) -> None:
    trace = [
        trace_record("E01", "system", "process_episode", 10.0),
        trace_record("E01", "llm", "extract", 5000.0, tokens=100),
        trace_record("E01", "llm", "search", 300.0, tokens=20),
        trace_record("E01", "llm", "extract", 5000.0, tokens=100, repeat=2),
        trace_record("D-MEM-01", "system", "process_episode", 30.0),
        trace_record("D-MEM-01", "system", "process_episode", 20.0),
    ]
    cases = [make_case("E01", "extraction"), make_case("D-MEM-01", "memory")]

    run = build_report([load_run(write_run(tmp_path / "a", cases, trace))])["runs"][0]

    assert slice_of(run, "extraction")["llm_calls"] == 2
    assert slice_of(run, "extraction")["tokens"] == 120
    assert slice_of(run, ALL_LABEL)["episode_latency_ms"] == {"p50": 20.0, "p95": 30.0}


def test_comparison_lists_regressions_and_bootstrap_delta(tmp_path: Path) -> None:
    base = [make_case(f"E{i:02}", "extraction", recall=(1, 2)) for i in range(10)]
    other = [make_case("E00", "extraction", recall=(0, 2), details=[MISSING_FACT])] + [
        make_case(f"E{i:02}", "extraction", recall=(2, 2)) for i in range(1, 10)
    ]
    runs = [load_run(write_run(tmp_path / "a", base)), load_run(write_run(tmp_path / "b", other))]

    comparison = build_report(runs)["comparisons"][0]
    row = next(item for item in comparison["slices"] if item["slice"] == "extraction")

    assert comparison["regressions"] == [{"id": "E00", "category": "extraction", "status": FAILED}]
    assert comparison["fixed"] == []
    assert row["task_success"]["delta"] == pytest.approx(-0.1)
    # Один кейс из десяти потерял 0.5 recall, девять получили по 0.5: среднее +0.4
    assert row["diff_recall_macro"]["delta"] == pytest.approx(0.4)
    low, high = row["diff_recall_macro"]["ci95"]
    assert low <= 0.4 <= high
    assert low > 0


def test_identical_runs_have_zero_deltas(tmp_path: Path) -> None:
    cases = [make_case("E01", "extraction", recall=(1, 2)), make_case("D-MEM-01", "memory", details=[MISSING_FACT])]
    runs = [load_run(write_run(tmp_path / "a", cases)), load_run(write_run(tmp_path / "b", cases))]

    rows = build_report(runs)["comparisons"][0]["slices"]

    assert {row["task_success"]["delta"] for row in rows} == {0.0}
    assert {tuple(row["task_success"]["ci95"]) for row in rows} == {(0.0, 0.0)}


def test_comparison_across_dataset_versions_fails(tmp_path: Path) -> None:
    cases = [make_case("E01", "extraction")]
    runs = [load_run(write_run(tmp_path / "a", cases)), load_run(write_run(tmp_path / "b", cases, dataset="b" * 64))]

    with pytest.raises(ValueError, match="разных версиях набора"):
        build_report(runs)


def test_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    base = write_run(tmp_path / "a", [make_case("E01", "extraction"), make_case("D-MEM-01", "memory")])
    other = write_run(
        tmp_path / "b", [make_case("E01", "extraction", details=[MISSING_FACT]), make_case("D-MEM-01", "memory")]
    )
    out = tmp_path / "out" / "REPORT.md"

    assert main([str(base), str(other), "--out", str(out)]) == 0

    markdown = out.read_text(encoding="utf-8")
    report = json.loads(out.with_name("report.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in report["comparisons"][0]["regressions"]] == ["E01"]
    assert "| E01 | extraction | failed |" in markdown
    assert "решение по классу не принимаем" in markdown
    assert not set(markdown) & set("—–«»“”")


def test_cli_rejects_folder_without_report(tmp_path: Path) -> None:
    assert main([str(tmp_path), "--out", str(tmp_path / "REPORT.md")]) == 2


@pytest.fixture(scope="module")
def rules_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Настоящий прогон B0 на кейсах трех классов."""
    out = tmp_path_factory.mktemp("report")
    ids = out / "ids.txt"
    ids.write_text("\n".join(("E04", "E05", "D-SUM-01", "D-WOR-16", "D-MEM-01")), encoding="utf-8")
    runner.main(["--system", "rules", "--split", "dev", "--ids", str(ids), "--out", str(out / "run")])
    return out / "run"


def test_category_micro_matches_scorer_groups(rules_run: Path) -> None:
    groups = json.loads((rules_run / "eval-report.json").read_text(encoding="utf-8"))["groups"]
    run = build_report([load_run(rules_run)])["runs"][0]

    for category, metrics in groups.items():
        for name in ("diff_precision", "diff_recall", "crm_chains"):
            ours = slice_of(run, category)[f"{name}_micro" if name != "crm_chains" else name]
            assert (ours["numerator"], ours["denominator"]) == (
                metrics[name]["numerator"],
                metrics[name]["denominator"],
            )

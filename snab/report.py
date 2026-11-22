"""Отчет по прогонам: python -m snab.report runs/<a> [runs/<b> ...] --out runs/REPORT.md.

Micro precision/recall по всем фактам дает длинным историям больший вес, поэтому рядом
стоят срезы по классам, короткие кейсы против историй и каждая длинная история отдельно.
Первый прогон - база, каждый следующий сравнивается с ней парным бутстрепом по кейсам.
Рядом с REPORT.md пишется report.json с теми же числами.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from snab.run import percentile

CATEGORIES = ("extraction", "decisions", "entities", "trust", "crm", "summary", "workflow", "memory")
HISTORY_CATEGORIES = ("workflow", "memory")
LONG_HISTORY_CATEGORY = "memory"
# Лекция 7 недели: на классе меньше 20 кейсов решение не принимаем
MIN_CASES_FOR_DECISION = 20
BOOTSTRAP_SAMPLES = 2000
SEED = 42
Z95 = 1.96
PASSED, FAILED, PENDING = "passed", "failed", "pending"
# Факт, по которому без судьи нет вердикта: сам по себе не провал кейса
PENDING_DETAIL = "unsupported_or_pending_evidence"
RATIO_METRICS = ("diff_precision", "diff_recall", "conflict_recall", "crm_chains")
SHORT_LABEL = "короткие кейсы"
HISTORY_LABEL = "истории (workflow+memory)"
ALL_LABEL = "все кейсы"


@dataclass(frozen=True)
class Run:
    name: str
    cases: dict[str, dict]
    summary: dict
    meta: dict


def case_status(case: dict) -> str:
    """Кейс eval-report -> passed, failed или pending.

    passed - fully_matches_structured_gold: все структурные проверки сошлись, нет
    лишних и повторных ответов, инъекция сбоя дошла до системы. pending - нет ни одного
    определенного расхождения, но часть оснований ждет судью (--judge off). В долю успеха
    pending не входит, она консервативна.
    """
    if case["fully_matches_structured_gold"]:
        return PASSED
    if case["unsupported"]:
        return FAILED
    counts = case["metrics"]["counts"]
    if counts.get("extra_answers") or counts.get("duplicate_answers"):
        return FAILED
    if all(detail["type"] == PENDING_DETAIL for detail in case["details"]):
        return PENDING
    return FAILED


def trace_by_case(path: Path, repeat: int) -> dict[str, dict]:
    """trace.jsonl -> по кейсу задержки эпизодов, вызовы LLM и токены первого повтора."""
    stats = defaultdict(lambda: {"episode_ms": [], "llm_calls": 0, "tokens": 0})
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["repeat"] != repeat:
            continue
        case = stats[record["case_id"]]
        if record["kind"] == "system" and record["name"] == "process_episode":
            case["episode_ms"].append(record["latency_ms"])
        elif record["kind"] == "llm":
            case["llm_calls"] += 1
            case["tokens"] += record["tokens_in"] + record["tokens_out"]
    return stats


def case_row(case: dict, trace: dict) -> dict:
    """Кейс eval-report и его трасса -> плоская строка для агрегатов."""
    metrics = case["metrics"]
    row = {"id": case["id"], "category": case["category"], "status": case_status(case)}
    for name in RATIO_METRICS:
        metric = metrics[name]
        row[name] = (metric["numerator"], metric["denominator"], metric["pending"])
    row["facts"] = metrics["diff_recall"]["denominator"]
    row.update(trace)
    return row


def report_path(run_dir: Path) -> Path:
    """eval-report.json, а у прогона с --repeat отчет первого повтора."""
    single = run_dir / "eval-report.json"
    return single if single.exists() else run_dir / "eval-report-r1.json"


def load_run(run_dir: Path) -> Run:
    """Папка прогона -> кейсы с трассой, summary.json и meta.json."""
    path = report_path(run_dir)
    if not path.exists():
        raise ValueError(f"{run_dir}: нет eval-report.json, это не папка прогона snab.run")
    report = json.loads(path.read_text(encoding="utf-8"))
    trace = trace_by_case(run_dir / "trace.jsonl", repeat=1)
    empty = {"episode_ms": [], "llm_calls": 0, "tokens": 0}
    cases = {case["id"]: case_row(case, trace.get(case["id"], empty)) for case in report["cases"]}
    return Run(
        name=run_dir.name,
        cases=cases,
        summary=json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
        meta=json.loads((run_dir / "meta.json").read_text(encoding="utf-8")),
    )


def wilson(successes: int, total: int) -> list[float] | None:
    """Доверительный интервал Вилсона 95% для доли; без наблюдений интервала нет."""
    if not total:
        return None
    share = successes / total
    center = share + Z95**2 / (2 * total)
    half = Z95 * sqrt(share * (1 - share) / total + Z95**2 / (4 * total**2))
    scale = 1 + Z95**2 / total
    # При 0 или n успехах округление дает -0.000 вместо границы
    return [max(0.0, (center - half) / scale), min(1.0, (center + half) / scale)]


def micro(rows: list[dict], metric: str) -> dict:
    """Сумма числителей на сумму знаменателей; при pending это нижняя граница."""
    good = sum(row[metric][0] for row in rows)
    total = sum(row[metric][1] for row in rows)
    pending = sum(row[metric][2] for row in rows)
    return {"numerator": good, "denominator": total, "pending": pending, "value": good / total if total else None}


def macro(rows: list[dict], metric: str) -> dict:
    """Среднее значений по кейсам, где знаменатель не ноль, и число таких кейсов."""
    values = [row[metric][0] / row[metric][1] for row in rows if row[metric][1]]
    return {"value": sum(values) / len(values) if values else None, "cases": len(values)}


def case_value(row: dict, metric: str) -> float | None:
    """Значение метрики кейса или None, если она на нем не определена."""
    good, total, _ = row[metric]
    return good / total if total else None


def aggregate(rows: list[dict]) -> dict:
    """Строки кейсов среза -> качество, успех кейсов с интервалом и стоимость."""
    passed = sum(row["status"] == PASSED for row in rows)
    chains = micro(rows, "crm_chains")
    episode_ms = [value for row in rows for value in row["episode_ms"]]
    return {
        "cases": len(rows),
        "small_sample": len(rows) < MIN_CASES_FOR_DECISION,
        "diff_precision_micro": micro(rows, "diff_precision"),
        "diff_recall_micro": micro(rows, "diff_recall"),
        "diff_precision_macro": macro(rows, "diff_precision"),
        "diff_recall_macro": macro(rows, "diff_recall"),
        "conflict_recall": micro(rows, "conflict_recall"),
        "crm_chains": dict(chains, ci95=wilson(chains["numerator"], chains["denominator"])),
        "task_success": {
            "passed": passed,
            "pending": sum(row["status"] == PENDING for row in rows),
            "value": passed / len(rows) if rows else None,
            "ci95": wilson(passed, len(rows)),
        },
        "facts": sum(row["facts"] for row in rows),
        "episode_latency_ms": {"p50": percentile(episode_ms, 0.5), "p95": percentile(episode_ms, 0.95)},
        "llm_calls": sum(row["llm_calls"] for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
    }


def slices(cases: dict[str, dict]) -> list[tuple[str, list[str]]]:
    """Срезы отчета: классы, короткие против историй, все кейсы и каждая длинная история."""
    by_category = defaultdict(list)
    for case_id, row in cases.items():
        by_category[row["category"]].append(case_id)
    unknown = set(by_category) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"неизвестные классы кейсов: {sorted(unknown)}")
    result = [(category, by_category[category]) for category in CATEGORIES if by_category[category]]
    history = [case_id for case_id, row in cases.items() if row["category"] in HISTORY_CATEGORIES]
    short = [case_id for case_id, row in cases.items() if row["category"] not in HISTORY_CATEGORIES]
    result += [(SHORT_LABEL, short), (HISTORY_LABEL, history), (ALL_LABEL, list(cases))]
    return result + [(case_id, [case_id]) for case_id in by_category[LONG_HISTORY_CATEGORY]]


def bootstrap_delta(pairs: list[tuple[float, float]], rng: random.Random) -> dict | None:
    """Пары (A, B) по кейсам -> средняя разность B - A и перцентильный интервал 95%."""
    if not pairs:
        return None
    deltas = [b - a for a, b in pairs]
    means = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(BOOTSTRAP_SAMPLES))
    return {
        "delta": sum(deltas) / len(deltas),
        "ci95": [means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]],
        "cases": len(pairs),
    }


def value_delta(base: dict, other: dict) -> float | None:
    """Разность значений агрегатов B - A, если оба определены."""
    if base["value"] is None or other["value"] is None:
        return None
    return other["value"] - base["value"]


def compare_runs(base: Run, other: Run) -> dict:
    """База и прогон -> разности по срезам, бутстреп и регрессии по кейсам."""
    if base.meta["dataset_sha256_lf"] != other.meta["dataset_sha256_lf"]:
        raise ValueError(f"{base.name} и {other.name} оценены на разных версиях набора")
    common = {case_id: base.cases[case_id] for case_id in base.cases if case_id in other.cases}
    rng = random.Random(SEED)
    rows = []
    for label, ids in slices(common):
        a = aggregate([base.cases[case_id] for case_id in ids])
        b = aggregate([other.cases[case_id] for case_id in ids])
        success = [(base.cases[i]["status"] == PASSED, other.cases[i]["status"] == PASSED) for i in ids]
        recall = [
            (case_value(base.cases[i], "diff_recall"), case_value(other.cases[i], "diff_recall"))
            for i in ids
            if base.cases[i]["diff_recall"][1]
        ]
        rows.append(
            {
                "slice": label,
                "cases": len(ids),
                "small_sample": len(ids) < MIN_CASES_FOR_DECISION,
                "task_success": bootstrap_delta([(float(x), float(y)) for x, y in success], rng),
                "diff_recall_macro": bootstrap_delta(recall, rng),
                "diff_precision_micro": value_delta(a["diff_precision_micro"], b["diff_precision_micro"]),
                "diff_recall_micro": value_delta(a["diff_recall_micro"], b["diff_recall_micro"]),
                "crm_chains": value_delta(a["crm_chains"], b["crm_chains"]),
            }
        )
    regressions = [
        {"id": case_id, "category": row["category"], "status": other.cases[case_id]["status"]}
        for case_id, row in common.items()
        if row["status"] == PASSED and other.cases[case_id]["status"] != PASSED
    ]
    fixed = [
        case_id
        for case_id, row in common.items()
        if row["status"] != PASSED and other.cases[case_id]["status"] == PASSED
    ]
    return {
        "base": base.name,
        "run": other.name,
        "common_cases": len(common),
        "only_in_base": sorted(set(base.cases) - set(other.cases)),
        "only_in_run": sorted(set(other.cases) - set(base.cases)),
        "slices": rows,
        "regressions": regressions,
        "fixed": fixed,
    }


def run_cost(summary: dict) -> dict:
    """summary.json -> время, вызовы, токены и память прогона."""
    return {
        "wall_time_s": summary["wall_time_s"],
        "episode_latency_ms": summary["episode_latency_ms"],
        "case_latency_ms": summary["case_latency_ms"],
        "llm_calls": summary["llm_calls"],
        "tool_calls": summary["tool_calls"],
        "tokens_in": summary["tokens_in"],
        "tokens_out": summary["tokens_out"],
        "peak_rss_mb": summary["peak_rss_mb"],
        "local_cpu_s": summary["cost"]["local_cpu_s"],
    }


def build_report(runs: list[Run]) -> dict:
    """Прогоны -> report.json: срезы каждого прогона и сравнения с первым."""
    return {
        "definitions": {
            "task_success": "доля кейсов с fully_matches_structured_gold; pending в успех не входит",
            "micro": "сумма числителей на сумму знаменателей по кейсам среза",
            "macro": "среднее значений по кейсам, где метрика определена",
            "ci95": "Вилсон для долей, перцентильный парный бутстреп по кейсам для разностей",
            "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": SEED},
            "min_cases_for_decision": MIN_CASES_FOR_DECISION,
        },
        "runs": [
            {
                "name": run.name,
                "system": run.meta["system"],
                "model": run.meta["model"],
                "git_sha": run.meta["git_sha"],
                "git_dirty": run.meta["git_dirty"],
                "dataset_version": run.meta["dataset_version"],
                "repeat": run.meta["repeat"],
                "cost": run_cost(run.summary),
                "slices": [
                    dict(aggregate([run.cases[case_id] for case_id in ids]), slice=label)
                    for label, ids in slices(run.cases)
                ],
            }
            for run in runs
        ],
        "comparisons": [compare_runs(runs[0], other) for other in runs[1:]],
    }


def fmt(value: float | None, digits: int = 3, sign: bool = False) -> str:
    """Число для таблицы; неопределенное значение - прочерк."""
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def fmt_ratio(metric: dict) -> str:
    """Доля с числителем и знаменателем, звездочка - часть проверок ждет судью."""
    if not metric["denominator"]:
        return "-"
    mark = "*" if metric["pending"] else ""
    return f"{fmt(metric['value'])}{mark} ({metric['numerator']}/{metric['denominator']})"


def fmt_ci(ci: list[float] | None, sign: bool = False) -> str:
    """Интервал в квадратных скобках."""
    return f"[{fmt(ci[0], sign=sign)}; {fmt(ci[1], sign=sign)}]" if ci else ""


def fmt_macro(metric: dict) -> str:
    """Macro-значение и число кейсов, где оно определено."""
    return f"{fmt(metric['value'])} (n={metric['cases']})" if metric["cases"] else "-"


def fmt_delta(result: dict | None) -> str:
    """Разность бутстрепа с интервалом."""
    return f"{fmt(result['delta'], sign=True)} {fmt_ci(result['ci95'], sign=True)}" if result else "-"


def small_note(item: dict) -> str:
    """Пометка для класса или сводного среза меньше порога; строка истории - одна история."""
    if item["slice"] not in (*CATEGORIES, SHORT_LABEL, HISTORY_LABEL, ALL_LABEL):
        return "одна история"
    return f"<{MIN_CASES_FOR_DECISION}: решение по классу не принимаем" if item["small_sample"] else ""


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Таблица Markdown."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def quality_table(slices_: list[dict]) -> list[str]:
    """Срезы прогона -> таблица качества."""
    header = [
        "Срез", "Кейсы", "Diff P micro", "Diff R micro", "Diff P macro", "Diff R macro",
        "Conflict R", "CRM-цепочки, 95% ДИ", "Успех кейса, 95% ДИ", "Ждут судью", "Факты", "Примечание",
    ]  # fmt: skip
    rows = [
        [
            item["slice"],
            str(item["cases"]),
            fmt_ratio(item["diff_precision_micro"]),
            fmt_ratio(item["diff_recall_micro"]),
            fmt_macro(item["diff_precision_macro"]),
            fmt_macro(item["diff_recall_macro"]),
            fmt_ratio(item["conflict_recall"]),
            f"{fmt_ratio(item['crm_chains'])} {fmt_ci(item['crm_chains']['ci95'])}".strip(),
            f"{fmt(item['task_success']['value'])} ({item['task_success']['passed']}/{item['cases']}) "
            f"{fmt_ci(item['task_success']['ci95'])}",
            str(item["task_success"]["pending"]),
            str(item["facts"]),
            small_note(item),
        ]
        for item in slices_
    ]
    return table(header, rows)


def cost_table(slices_: list[dict]) -> list[str]:
    """Срезы прогона -> задержка эпизода по трассе, вызовы LLM и токены."""
    header = ["Срез", "Кейсы", "Эпизод P50, мс", "Эпизод P95, мс", "Вызовы LLM", "Токены"]
    rows = [
        [
            item["slice"],
            str(item["cases"]),
            fmt(item["episode_latency_ms"]["p50"], 1),
            fmt(item["episode_latency_ms"]["p95"], 1),
            str(item["llm_calls"]),
            str(item["tokens"]),
        ]
        for item in slices_
    ]
    return table(header, rows)


def run_section(run: dict) -> list[str]:
    """Раздел одного прогона: версия, стоимость и таблицы срезов."""
    cost = run["cost"]
    lines = [
        f"## Прогон {run['name']}",
        "",
        f"Система `{run['system']}`, модель `{run['model'] or 'нет'}`, коммит `{run['git_sha'][:7]}`"
        f"{' с незакоммиченными правками' if run['git_dirty'] else ''}, набор {', '.join(run['dataset_version'])}"
        f", повторов {run['repeat']}",
        "",
        "Стоимость прогона по summary.json:",
        "",
    ]
    lines += table(
        ["Время, с", "Эпизод P50/P95, мс", "Кейс P50/P95, мс", "Вызовы LLM", "Инструменты", "Токены in/out",
         "Пик RSS, МБ", "CPU, с"],
        [[
            fmt(cost["wall_time_s"], 1),
            f"{fmt(cost['episode_latency_ms']['p50'], 1)} / {fmt(cost['episode_latency_ms']['p95'], 1)}",
            f"{fmt(cost['case_latency_ms']['p50'], 1)} / {fmt(cost['case_latency_ms']['p95'], 1)}",
            str(cost["llm_calls"]),
            str(cost["tool_calls"]),
            f"{cost['tokens_in']} / {cost['tokens_out']}",
            fmt(cost["peak_rss_mb"], 1),
            fmt(cost["local_cpu_s"], 1),
        ]],
    )  # fmt: skip
    lines += ["", "### Качество по срезам", ""] + quality_table(run["slices"])
    lines += ["", "### Стоимость по срезам", ""] + cost_table(run["slices"])
    return lines + [""]


def comparison_section(comparison: dict) -> list[str]:
    """Раздел сравнения прогона с базой."""
    lines = [
        f"## {comparison['run']} против {comparison['base']}",
        "",
        f"Общих кейсов {comparison['common_cases']}. Разность B - A; для успеха кейса и macro recall "
        f"в скобках 95% интервал парного бутстрепа по кейсам ({BOOTSTRAP_SAMPLES} выборок, seed {SEED})",
        "",
    ]
    if comparison["only_in_base"] or comparison["only_in_run"]:
        lines += [
            f"Только в базе: {', '.join(comparison['only_in_base']) or 'нет'}. "
            f"Только в прогоне: {', '.join(comparison['only_in_run']) or 'нет'}",
            "",
        ]
    lines += table(
        ["Срез", "Кейсы", "Успех кейса", "Diff R macro", "Diff P micro", "Diff R micro", "CRM-цепочки", "Примечание"],
        [
            [
                row["slice"],
                str(row["cases"]),
                fmt_delta(row["task_success"]),
                fmt_delta(row["diff_recall_macro"]),
                fmt(row["diff_precision_micro"], sign=True),
                fmt(row["diff_recall_micro"], sign=True),
                fmt(row["crm_chains"], sign=True),
                small_note(row),
            ]
            for row in comparison["slices"]
        ],
    )
    lines += ["", f"### Регрессии: прошли в {comparison['base']}, не прошли в {comparison['run']}", ""]
    if comparison["regressions"]:
        lines += table(
            ["Кейс", "Класс", "Статус в прогоне"],
            [[item["id"], item["category"], item["status"]] for item in comparison["regressions"]],
        )
    else:
        lines.append("Нет")
    lines += ["", f"Исправлены: {', '.join(comparison['fixed']) or 'нет'}", ""]
    return lines


def render_markdown(report: dict) -> str:
    """report.json -> REPORT.md."""
    lines = [
        "# Отчет по прогонам",
        "",
        "Micro считает все факты вместе, поэтому длинная история весит больше короткого кейса. "
        "Macro усредняет значения по кейсам, где метрика определена, n - число таких кейсов. "
        "Успех кейса - все обязательные структурные проверки eval/score.py сошлись "
        "(fully_matches_structured_gold); кейс, где расхождений нет, но основания ждут судью, "
        'в успех не входит и посчитан в столбце "Ждут судью". '
        "Звездочка - часть проверок ждет судью, значение - нижняя граница. "
        f"На срезе меньше {MIN_CASES_FOR_DECISION} кейсов решение по классу не принимаем",
        "",
    ]
    for run in report["runs"]:
        lines += run_section(run)
    for comparison in report["comparisons"]:
        lines += comparison_section(comparison)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Аргументы CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="папки прогонов, первая - база сравнения")
    parser.add_argument("--out", type=Path, default=Path("runs/REPORT.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Чтение прогонов, REPORT.md и report.json."""
    args = parse_args(argv)
    try:
        report = build_report([load_run(path) for path in args.runs])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report), encoding="utf-8")
    args.out.with_name("report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.out}, {args.out.with_name('report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

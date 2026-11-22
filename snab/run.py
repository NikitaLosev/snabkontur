"""Единый раннер eval: python -m snab.run --system rules --split dev --out runs/<name>.

Состояние CRM создается заново для каждого кейса, эпизоды идут в порядке timeline,
система видит только источники, поступившие к шагу. Событие помечается source_complete,
если это последний кусок своего источника: так конвейер загрузки сообщает, что запись
встречи закончилась и частичные кандидаты можно собрать в один дифф. Действия
пользователя выполняет код через CRM: подтверждается только предложение системы, сопоставленное с gold по фактам
кодом, без судьи. Операции gold не исполняются.
"""

import argparse
import hashlib
import json
import platform
import re
import resource
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from math import ceil
from pathlib import Path

from pydantic import ValidationError

from snab.contracts import REPO_ROOT, SCORER, SCORER_PATH, EpisodeContext, Proposal, StopReason, SystemOutput
from snab.crm import MockCrm
from snab.systems import SYSTEMS, System, build_system
from snab.tools import DEFAULT_BUDGET, FaultInjector, ModelTools, StepBudget
from snab.trace import Tracer

DATASETS = {"dev": REPO_ROOT / "eval" / "dev.jsonl", "test": REPO_ROOT / "eval" / "test.jsonl"}
REQUIREMENTS = REPO_ROOT / "requirements.txt"
CODE_DIR = REPO_ROOT / "snab"
REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9_.-]+")
REPORT_METRICS = (
    "diff_precision",
    "diff_recall",
    "crm_checkpoints",
    "crm_chains",
    "literal_citations",
    "supported_citations",
    "conflict_recall",
    "answers_accuracy",
    "summary_precision",
    "summary_recall",
    "purchase_link_accuracy",
)


def same_atom(reference: dict, actual: dict) -> bool:
    """Факт системы совпадает с фактом gold для подтверждения.

    Как atom_compare из eval/score.py, но без судьи: задачи сравниваются по закупке,
    исполнителю, сроку и статусу без названия, остальное только точным значением.
    """
    if any(reference[key] != actual[key] for key in ("phase", "kind", "purchase", "field")):
        return False
    if reference["kind"] == "task":
        return SCORER.equal(
            {key: value for key, value in reference["value"].items() if key != "title"},
            {key: value for key, value in actual["value"].items() if key != "title"},
        )
    return SCORER.equal(reference["value"], actual["value"], reference["field"])


def map_gold_proposals(case: dict, proposals: list[dict]) -> dict[str, list[str]]:
    """ID предложения gold -> ID предложений системы по сопоставлению фактов."""
    predicted_case = {**case, "gold": {**case["gold"], "proposals": proposals}}
    expected = SCORER.atoms(case)
    task_map = {}
    for atom in SCORER.atoms(predicted_case):
        if atom["kind"] != "task":
            continue
        for reference in expected:
            if (
                reference["kind"] == "task"
                and reference["task_id"] not in task_map.values()
                and same_atom(reference, atom)
            ):
                task_map[atom["task_id"]] = reference["task_id"]
                break
    predicted = SCORER.atoms(predicted_case, task_map)
    pairs, _ = SCORER.match_items(expected, predicted, same_atom)
    mapping = {}
    for expected_index, predicted_index in pairs:
        system_ids = mapping.setdefault(expected[expected_index]["proposal_id"], [])
        if predicted[predicted_index]["proposal_id"] not in system_ids:
            system_ids.append(predicted[predicted_index]["proposal_id"])
    return mapping


def proposal_error(proposal: dict, sources: dict[str, dict], known_ids: set[str]) -> tuple[str, str] | None:
    """Проверка кодом: схема, уникальный ID и дословные цитаты -> (код, сообщение) или None."""
    try:
        Proposal.model_validate(proposal)
    except ValidationError as error:
        return "invalid_schema", error.errors()[0]["msg"]
    if proposal["id"] in known_ids:
        return "duplicate_id", f"proposal {proposal['id']} already exists"
    for operation in proposal["operations"]:
        for evidence in operation["evidence"]:
            source = sources.get(evidence.get("event_id") or evidence.get("action_id"))
            if source is None or evidence["quote"] not in source.get("text", source.get("reason", "")):
                return "unverified_quote", "quote is not a verbatim substring of an available source"
    return None


def apply_corrections(operations: list[dict], corrections: list[dict]) -> list[dict]:
    """Поправка пользователя заменяет операцию с той же целью или добавляется к пакету."""

    def target(operation: dict) -> tuple[str, ...]:
        """Что меняет операция: поле карточки или задачу."""
        if operation["op"] == "set_field":
            return operation["op"], operation["purchase_id"], operation["field"]
        return operation["op"], operation.get("task_id") or operation["task"]["id"]

    replaced = {target(correction): correction for correction in corrections}
    merged = [replaced.pop(target(operation), operation) for operation in operations]
    return merged + list(replaced.values())


class CaseReplay:
    """Прогон одного кейса: CRM, инструменты, трейс и результаты в формате gold."""

    def __init__(
        self, case: dict, system: System, run_id: str, repeat: int = 1, budget: StepBudget = DEFAULT_BUDGET
    ) -> None:
        self.case = case
        self.system = system
        self.budget = budget
        self.tracer = Tracer(run_id, case["id"], repeat)
        self.crm = MockCrm()
        self.faults = FaultInjector(case["input"].get("fault_injections", []))
        self.tools = ModelTools(self.crm, self.faults, self.tracer, budget)
        self.wanted_points = {check["after"] for check in case["gold"]["checks"]}
        self.all_events = {event["id"]: event for _, step in SCORER.timeline(case) for event in step["events"]}
        self.last_chunks = {event.get("source_id", event["id"]): event["id"] for event in self.all_events.values()}
        self.sources = {}
        self.proposals = {}
        self.blocked_proposals = {}
        self.invalid_proposals = Counter()
        self.applied = {}
        self.issues, self.answers, self.checks = [], [], []
        self.episode_latencies_ms = []
        self.stop_reasons = Counter()
        self.actions = {
            "approve": self.approve,
            "edit_then_approve": self.approve,
            "reject": self.reject,
            "manual_edit": self.manual_edit,
            "undo": self.undo,
            "reimport": self.reimport,
            "query": self.query,
            "rebuild_summary": self.rebuild_summary,
        }

    def run(self) -> dict:
        """Весь кейс -> строка predictions в формате набора.

        Предложения с недословной цитатой тоже попадают в predictions, чтобы оценщик их
        посчитал, но подтвердить их нельзя.
        """
        data = self.case["input"]
        for name in self.crm.load(data["initial_state"], data.get("derived_views")):
            self.issues.append({"code": "stale_derived_view", "view": name})
        for episode_id, step in SCORER.timeline(self.case):
            self.run_episode(episode_id, step)
        row = {key: value for key, value in self.case.items() if key != "gold"}
        row["gold"] = {
            "proposals": [*self.proposals.values(), *self.blocked_proposals.values()],
            "checks": self.checks,
            "expected_issues": self.issues,
            "forbidden_actions": [],
            "answers": self.answers,
        }
        return row

    def set_point(self, point: str) -> None:
        """Точка для трейса и инъекции сбоев: эпизод или действие."""
        self.tracer.point = point
        self.faults.point = point

    def run_episode(self, episode_id: str, step: dict) -> None:
        """Источники эпизода -> предложения, снимок до действий, затем действия пользователя."""
        self.set_point(episode_id)
        new_events = [event for event in step["events"] if self.register_source(event)]
        if new_events:
            self.ingest(episode_id, new_events)
        for point in [episode_id, *(event["id"] for event in step["events"])]:
            self.checkpoint(point)
        for action in step.get("user_actions", []):
            self.set_point(action["id"])
            self.sources[action["id"]] = action
            self.actions[action["kind"]](action)
            self.checkpoint(action["id"])

    def register_source(self, event: dict) -> bool:
        """Источник с тем же ID и текстом второй раз не обрабатывается."""
        digest = hashlib.sha256(event["text"].encode()).hexdigest()
        known = self.sources.get(event["id"])
        if known is not None and hashlib.sha256(known["text"].encode()).hexdigest() == digest:
            self.tracer.record("system", "duplicate_source", {"event_id": event["id"]}, result={"sha256": digest})
            return False
        self.sources[event["id"]] = event
        return True

    def ingest(self, episode_id: str, events: list[dict]) -> None:
        """Передает системе новые источники со свежим бюджетом шагов."""
        self.tools.add_sources(events)
        self.tools.start_episode()
        span = self.call_system("process_episode", self.context(episode_id, events))
        self.episode_latencies_ms.append(span["latency_ms"])

    def context(self, episode_id: str, events: list[dict]) -> EpisodeContext:
        """Допустимый контекст шага без gold, fault_injections и ID кейса."""
        data = self.case["input"]
        return EpisodeContext(
            episode_id=episode_id,
            events=[
                {**event, "source_complete": self.last_chunks[event.get("source_id", event["id"])] == event["id"]}
                for event in events
            ],
            seen_events=list(self.tools.sources),
            crm_state=self.crm.snapshot(),
            authority_registry=data.get("authority_registry", {}),
            now=data.get("now"),
            working_context=data.get("working_context", {}),
            derived_views={name: view for name, view in self.crm.derived_views().items() if view is not None},
        )

    def call_system(self, method: str, context: EpisodeContext, *args: dict) -> dict:
        """Вызов системы под дедлайном; ее шаги и инструменты пишутся дочерними записями.

        Возвращает запись шага системы в трейсе. Ответ системы проходит проверку кодом:
        принятые предложения, проблемы и ответы становятся результатом кейса.
        """
        state = context.crm_state
        span = self.tracer.record(
            "system",
            method,
            {
                "episode_id": context.episode_id,
                "event_ids": [event["id"] for event in context.events],
                "crm_revisions": {pid: card["revision"] for pid, card in state["purchases"].items()},
                "crm_state_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
            },
        )
        self.tracer.parent = span["step"]
        started = time.perf_counter()
        output, error = self.run_system(method, context, *args)
        latency_ms = (time.perf_counter() - started) * 1000
        for step in output.trace_steps:
            self.tracer.record(**step.model_dump())
        accepted = self.accept_proposals(output.proposals)
        self.tracer.parent = None
        if output.proposals and not accepted:
            output.stop_reason, error = StopReason.INVALID_OUTPUT, error or "invalid_output"
        span.update(
            latency_ms=round(latency_ms, 3),
            error=error,
            stop_reason=output.stop_reason.value,
            result={"proposals": len(accepted), "issues": len(output.issues), "answers": len(output.answers)},
        )
        self.stop_reasons[output.stop_reason.value] += 1
        self.issues.extend(output.issues)
        self.answers.extend(output.answers)
        return span

    def run_system(self, method: str, context: EpisodeContext, *args: dict) -> tuple[SystemOutput, str | None]:
        """Система в фоновом потоке с лимитом времени куска -> ответ и код ошибки.

        Ошибка или зависание системы не обрывают прогон: шаг получает статус, CRM не меняется.
        """
        outcome = {}

        def target() -> None:
            """Ответ или исключение системы сохраняются для основного потока."""
            try:
                outcome["output"] = SystemOutput.model_validate(
                    getattr(self.system, method)(context, *args, self.tools)
                )
            except Exception as error:  # граница системы: любая ее ошибка становится статусом шага
                outcome["error"] = error

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(self.budget.episode_timeout_s)
        if worker.is_alive():
            self.replace_tools()
            return SystemOutput(stop_reason=StopReason.TIMEOUT), "timeout"
        error = outcome.get("error")
        if isinstance(error, TimeoutError):
            return SystemOutput(stop_reason=StopReason.TIMEOUT), "timeout"
        if error is not None:
            return SystemOutput(stop_reason=StopReason.INVALID_OUTPUT), type(error).__name__
        return outcome["output"], None

    def replace_tools(self) -> None:
        """Зависший поток сохраняет закрытые инструменты, следующие шаги получают новые."""
        stale = self.tools
        stale.close()
        self.tools = ModelTools(self.crm, self.faults, self.tracer, self.budget)
        self.tools.add_sources(stale.sources)

    def accept_proposals(self, proposals: list[dict]) -> list[dict]:
        """Проверенные кодом предложения доступны для подтверждения; остальные считаются по причине."""
        accepted = []
        for proposal in proposals:
            error = proposal_error(proposal, self.sources, set(self.proposals) | set(self.blocked_proposals))
            if error is None:
                self.proposals[proposal["id"]] = proposal
                accepted.append(proposal)
                continue
            code, message = error
            self.invalid_proposals[code] += 1
            if code == "unverified_quote":
                self.blocked_proposals[proposal["id"]] = proposal
            self.tracer.record(
                "system",
                "validate_proposal",
                {"proposal_id": proposal.get("id")},
                error="invalid_output",
                result={"code": code, "message": message},
            )
        return accepted

    def checkpoint(self, point: str) -> None:
        """Снимок CRM в контрольной точке, которую задает протокол набора."""
        if point not in self.wanted_points or any(check["after"] == point for check in self.checks):
            return
        check = {"after": point, "expected_state": self.crm.snapshot()}
        if "derived_views" in self.case["input"]:
            check["expected_derived_views"] = self.crm.derived_views()
        self.checks.append(check)

    def commit(
        self, operations: list[dict], change_id: str, expected_revision: dict, kind: str, proposal_ids: list[str]
    ) -> None:
        """Одна попытка транзакции на действие пользователя.

        Сбой записи оставляет CRM прежней и показывается закупщику; повтор - его следующее
        действие с тем же change_id, которое применит пакет целиком.
        """
        started = time.perf_counter()
        result = self.crm.apply(
            operations,
            change_id,
            expected_revision,
            ",".join(proposal_ids),
            kind,
            fail_after_first_operation=self.faults.take("commit") is not None,
        )
        self.tracer.record(
            "crm",
            kind,
            {"proposal_ids": proposal_ids},
            result=result.data,
            error=result.error and result.error.code,
            latency_ms=(time.perf_counter() - started) * 1000,
            idempotency_key=change_id,
        )
        if result.ok:
            for proposal_id in proposal_ids:
                self.applied.setdefault(proposal_id, change_id)
            self.report_invalidated(result.data["invalidated_views"])
        elif result.error.code == "conflict":
            self.issues.append(
                {
                    "code": "stale_revision",
                    "after": self.tracer.point,
                    "purchase_id": result.error.details["purchase_id"],
                }
            )
        else:
            self.issues.append({"code": result.error.code, "after": self.tracer.point})

    def report_invalidated(self, view_names: list[str]) -> None:
        """Сброшенная после записи сводка требует пересборки."""
        for name in view_names:
            self.issues.append({"code": "derived_view_needs_rebuild", "after": self.tracer.point, "view": name})

    def approve(self, action: dict) -> None:
        """Подтверждение применяет соответствующие предложения системы одним пакетом."""
        change_id = action["change_id"]
        mapping = map_gold_proposals(self.case, list(self.proposals.values()))
        proposal_ids = [
            proposal_id
            for proposal_id in mapping.get(action["proposal_id"], [])
            if self.applied.get(proposal_id, change_id) == change_id
        ]
        if not proposal_ids:
            self.tracer.record(
                "crm",
                action["kind"],
                {"gold_proposal_id": action["proposal_id"]},
                error="not_found",
                idempotency_key=change_id,
                result={"message": "no matching system proposal"},
            )
            return
        operations = [
            operation for proposal_id in proposal_ids for operation in self.proposals[proposal_id]["operations"]
        ]
        operations = apply_corrections(operations, action.get("corrections", []))
        # Пользователь подтверждает дифф системы как есть: устаревшим его делает база самого диффа
        based_on = {}
        for proposal_id in proposal_ids:
            based_on.update(self.proposals[proposal_id]["based_on"])
        self.commit(operations, change_id, based_on, action["kind"], proposal_ids)

    def manual_edit(self, action: dict) -> None:
        """Ручная правка закупщика проходит ту же проверку ревизии."""
        expected = {action["purchase_id"]: action["expected_revision"]}
        self.commit(action["operations"], action["id"], expected, "manual_edit", [])

    def reject(self, action: dict) -> None:
        """Отклонение не меняет карточки."""
        mapping = map_gold_proposals(self.case, list(self.proposals.values()))
        for proposal_id in mapping.get(action["proposal_id"], []):
            result = self.crm.reject(proposal_id, action.get("reason"))
            self.tracer.record("crm", "reject", {"proposal_id": proposal_id}, result=result.data)

    def undo(self, action: dict) -> None:
        """Откат новой ревизией; конфликт с поздней правкой показывается закупщику."""
        started = time.perf_counter()
        result = self.crm.undo(action["change_id"], action["id"])
        self.tracer.record(
            "crm",
            "undo",
            {"change_id": action["change_id"]},
            result=result.data,
            error=result.error and result.error.code,
            idempotency_key=action["id"],
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if result.ok:
            self.report_invalidated(result.data["invalidated_views"])
        elif result.error.code == "conflict":
            self.issues.append({"code": "undo_conflict", "after": action["id"], **result.error.details})
        else:
            self.issues.append({"code": result.error.code, "after": action["id"]})

    def reimport(self, action: dict) -> None:
        """Повторная загрузка: тот же текст пропускается, измененный обрабатывается заново."""
        events = [self.all_events[event_id] for event_id in action["event_ids"]]
        new_events = [event for event in events if self.register_source(event)]
        if new_events:
            self.ingest(action["id"], new_events)

    def query(self, action: dict) -> None:
        """Контрольный запрос: ответ системы по текущему состоянию со своим бюджетом шагов."""
        self.tools.start_episode()
        self.call_system("answer_query", self.context(action["id"], []), action)

    def rebuild_summary(self, action: dict) -> None:
        """Пересборка сводки кодом, без модели."""
        result = self.crm.rebuild_summary(action["purchase_id"])
        self.tracer.record(
            "crm",
            "rebuild_summary",
            {"purchase_id": action["purchase_id"]},
            result=result.data,
            error=result.error and result.error.code,
        )


def percentile(values: list[float], share: float) -> float | None:
    """Перцентиль по ближайшему рангу."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, ceil(share * len(ordered)) - 1)]


def read_cases(path: Path) -> list[dict]:
    """JSONL набора -> список кейсов в порядке файла."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_cases(cases: list[dict], args: argparse.Namespace) -> list[dict]:
    """Фильтры CLI: файл с ID, категория, первые N."""
    if args.ids:
        wanted = {line.strip() for line in Path(args.ids).read_text(encoding="utf-8").splitlines() if line.strip()}
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"unknown case ids: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in wanted]
    if args.category:
        cases = [case for case in cases if case["category"] == args.category]
    return cases[: args.limit] if args.limit else cases


def git_output(*command: str) -> str:
    """Вывод git в корне репозитория."""
    return subprocess.run(["git", *command], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def hardware() -> dict:
    """Железо: на macOS через sysctl, иначе platform."""
    info = {"platform": platform.platform(), "machine": platform.machine()}
    if sys.platform == "darwin":
        keys = ("hw.model", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu")
        values = subprocess.run(["sysctl", "-n", *keys], capture_output=True, text=True, check=True).stdout
        info.update(zip(keys, values.splitlines(), strict=True))
    return info


def file_sha256(path: Path) -> str:
    """sha256 файла без учета CRLF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def requirement_versions() -> dict[str, str]:
    """Установленные версии всех пакетов из requirements.txt."""
    names = [
        REQUIREMENT_NAME.match(line).group(0)
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return {name: version(name) for name in names}


def dataset_path(args: argparse.Namespace) -> Path:
    """--cases или файл split -> путь к набору кейсов."""
    return args.cases or DATASETS[args.split]


def collect_meta(args: argparse.Namespace, system: System, cases: list[dict], started: str, finished: str) -> dict:
    """Все, что нужно для воспроизведения прогона, включая хэши кода при грязном дереве."""
    dataset = dataset_path(args)
    git_dirty = bool(git_output("status", "--porcelain").strip())
    return {
        "git_sha": git_output("rev-parse", "HEAD").strip(),
        "git_dirty": git_dirty,
        "git_diff_sha256": hashlib.sha256(git_output("diff", "HEAD", "--binary").encode()).hexdigest()
        if git_dirty
        else None,
        "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in sorted(CODE_DIR.rglob("*.py"))},
        "scorer_sha256": file_sha256(SCORER_PATH),
        "dataset": str(dataset.relative_to(REPO_ROOT)),
        "dataset_sha256_lf": file_sha256(dataset),
        "dataset_version": sorted({case["dataset_version"] for case in cases}),
        "selected_case_ids": [case["id"] for case in cases],
        "repeat": args.repeat,
        **system.describe(),
        "budgets": asdict(DEFAULT_BUDGET),
        "hardware": hardware(),
        "python": sys.version,
        "packages": requirement_versions(),
        "argv": sys.argv,
        "started_at": started,
        "finished_at": finished,
    }


def peak_rss_mb() -> float:
    """Пиковая память процесса: macOS отдает байты, Linux - килобайты."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1)


def summarize(records: list[dict], case_stats: list[dict], wall_time_s: float, cpu_time_s: float) -> dict:
    """Время, вызовы, токены, память и стоимость прогона.

    API не используется (раздел 3 design doc), поэтому стоимость - процессорное время
    и токены локальной модели, в том числе на одно событие.
    """
    episode_ms = [value for stats in case_stats for value in stats["episode_latencies_ms"]]
    case_ms = [stats["wall_time_ms"] for stats in case_stats]
    events = sum(stats["events"] for stats in case_stats)
    llm = [record for record in records if record["kind"] == "llm"]
    tokens = sum(record["tokens_in"] + record["tokens_out"] for record in llm)
    # prompt_eval_count Ollama на вызов: полный контекст, по нему проверяется окно 8192
    prompt_tokens = [record["tokens_in"] for record in llm]
    return {
        "wall_time_s": round(wall_time_s, 3),
        "cases": len(case_stats),
        "episodes": len(episode_ms),
        "events": events,
        "episode_latency_ms": {"p50": percentile(episode_ms, 0.5), "p95": percentile(episode_ms, 0.95)},
        "case_latency_ms": {"p50": percentile(case_ms, 0.5), "p95": percentile(case_ms, 0.95)},
        "llm_calls": len(llm),
        "tool_calls": sum(record["kind"] == "tool" for record in records),
        "tool_errors": dict(
            Counter(record["error"] for record in records if record["kind"] == "tool" and record["error"])
        ),
        "crm_writes": sum(
            record["kind"] == "crm" and (record["result"] or {}).get("status") == "applied" for record in records
        ),
        "tokens_in": sum(record["tokens_in"] for record in llm),
        "tokens_out": sum(record["tokens_out"] for record in llm),
        "prompt_tokens_per_call": {
            "p50": percentile(prompt_tokens, 0.5),
            "p95": percentile(prompt_tokens, 0.95),
            "max": max(prompt_tokens, default=None),
        },
        "stop_reasons": dict(sum((Counter(stats["stop_reasons"]) for stats in case_stats), Counter())),
        "invalid_proposals": dict(sum((Counter(stats["invalid_proposals"]) for stats in case_stats), Counter())),
        "unsupported_fault_injections": unsupported_cases(case_stats),
        "peak_rss_mb": peak_rss_mb(),
        "cost": {
            "api_usd": 0.0,
            "local_cpu_s": round(cpu_time_s, 3),
            "llm_tokens": tokens,
            "cpu_ms_per_event": round(cpu_time_s * 1000 / events, 3) if events else None,
            "llm_tokens_per_event": round(tokens / events, 1) if events else None,
        },
    }


def unsupported_cases(case_stats: list[dict]) -> dict[str, list[dict]]:
    """ID кейса -> инъекции, до которых система не дошла: такой кейс не считается успешным."""
    return {stats["case_id"]: stats["unconsumed_faults"] for stats in case_stats if stats["unconsumed_faults"]}


def replay_cases(cases: list[dict], system: System, run_id: str, repeat: int) -> tuple[list, list, list]:
    """Все кейсы одного повтора -> predictions, трейс и статистика по кейсам."""
    rows, records, case_stats = [], [], []
    for case in cases:
        started = time.perf_counter()
        replay = CaseReplay(case, system, run_id, repeat)
        rows.append(replay.run())
        records.extend(replay.tracer.records)
        case_stats.append(
            {
                "case_id": case["id"],
                "wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
                "events": len(replay.all_events),
                "episode_latencies_ms": replay.episode_latencies_ms,
                "stop_reasons": dict(replay.stop_reasons),
                "invalid_proposals": dict(replay.invalid_proposals),
                "unconsumed_faults": replay.faults.unconsumed(),
            }
        )
    return rows, records, case_stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Строки JSON по одной на линию."""
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def gold_file(args: argparse.Namespace, cases: list[dict], out_dir: Path) -> Path:
    """Эталон для оценщика: весь набор или только выбранные кейсы."""
    dataset = dataset_path(args)
    if len(cases) == len(read_cases(dataset)):
        return dataset
    path = out_dir / "gold-selection.jsonl"
    write_jsonl(path, cases)
    return path


def run_scorer(gold: Path, predictions: Path, report_path: Path, unsupported: dict[str, list[dict]]) -> dict:
    """eval/score.py --judge off -> отчет в папке прогона с пометкой unsupported.

    Кейс с инъекцией, до которой система не дошла, не проверял режим сбоя: он помечается
    unsupported и не засчитывается как полное совпадение.
    """
    subprocess.run(
        [
            sys.executable,
            str(SCORER_PATH),
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--judge",
            "off",
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["unsupported_fault_injections"] = unsupported
    for case in report["cases"]:
        case["unsupported"] = case["id"] in unsupported
        case["fully_matches_structured_gold"] &= not case["unsupported"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Аргументы CLI; test закрыт без явного флага."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--split", choices=sorted(DATASETS), default="dev")
    parser.add_argument("--cases", type=Path, help="свой набор кейсов вместо --split, например eval/dev_noisy.jsonl")
    parser.add_argument("--ids", help="file with one case id per line")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--i-know-this-is-heldout", action="store_true", dest="heldout_confirmed")
    args = parser.parse_args(argv)
    if args.cases is not None:
        args.cases = args.cases.resolve()
    heldout = args.split == "test" or args.cases == DATASETS["test"].resolve()
    if heldout and not args.heldout_confirmed:
        parser.error("test is held out until KT-4; pass --i-know-this-is-heldout to run it")
    if args.repeat < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--repeat and --limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Прогон системы, файлы прогона и оценка."""
    args = parse_args(argv)
    system = build_system(args.system)
    started_at = datetime.now(UTC)
    run_id = f"{started_at:%Y%m%dT%H%M%SZ}_{args.system}_{args.cases.stem if args.cases else args.split}"
    out_dir = args.out or REPO_ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(read_cases(dataset_path(args)), args)
    gold = gold_file(args, cases, out_dir)
    started, cpu_started = time.perf_counter(), time.process_time()
    all_records, all_stats, reports = [], [], []
    for repeat in range(1, args.repeat + 1):
        rows, records, case_stats = replay_cases(cases, system, run_id, repeat)
        suffix = "" if args.repeat == 1 else f"-r{repeat}"
        predictions = out_dir / f"predictions{suffix}.jsonl"
        write_jsonl(predictions, rows)
        report_path = out_dir / f"eval-report{suffix}.json"
        reports.append(run_scorer(gold, predictions, report_path, unsupported_cases(case_stats)))
        all_records.extend(records)
        all_stats.extend(case_stats)
    summary = summarize(all_records, all_stats, time.perf_counter() - started, time.process_time() - cpu_started)
    if args.repeat > 1:
        summary["pass_all_repeats"] = {
            case["id"]: all(
                next(item for item in report["cases"] if item["id"] == case["id"])["fully_matches_structured_gold"]
                for report in reports
            )
            for case in cases
        }
    write_jsonl(out_dir / "trace.jsonl", all_records)
    meta = collect_meta(args, system, cases, started_at.isoformat(), datetime.now(UTC).isoformat())
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {name: reports[0]["metrics"][name] for name in REPORT_METRICS}
    print(
        json.dumps(
            {
                "out": str(out_dir),
                "metrics": {
                    name: {key: value[key] for key in ("numerator", "denominator", "value", "status")}
                    for name, value in metrics.items()
                },
                "wall_time_s": summary["wall_time_s"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Замер времени закупщика: ручное заполнение диффа против проверки диффа системы.

python -m snab.timing --predictions runs/<run>/predictions.jsonl  запускает локальную страницу;
python -m snab.timing --summary  считает медианы времени и precision/recall по runs/timing/*.jsonl.

Режим кейса задается контрбалансом: группа A или B, выбранная на входе, решает, какие кейсы
участник заполняет вручную, а какие проверяет; у группы A и группы B режимы противоположны.
Четность хэша имени только подсказывает группу: двое могут получить одну четность, поэтому
группу выбирают явно. После первой сдачи группа участника закреплена. Время считается по меткам браузера без пауз,
правильность считает сервер по gold, участнику она не показывается.
"""

import argparse
import hashlib
import json
import logging
import re
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from snab.contracts import REPO_ROOT, SCORER

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CASES = REPO_ROOT / "eval" / "dev.jsonl"
FORBIDDEN_CASES = REPO_ROOT / "eval" / "test.jsonl"
DEFAULT_OUT = REPO_ROOT / "runs" / "timing"
PAGE_PATH = Path(__file__).with_name("timing_page.html")
# Встречи и переписки с 0-3 операциями: цена и срок, две закупки, пересчет суммы после
# предоплаты, поручение с ответственным и сроком, условное предложение, относительные даты
DEFAULT_IDS = ("E10", "E14", "D-WOR-01", "D-EXT-07", "D-DEC-01", "E12")
MANUAL, REVIEW = "manual", "review"
GROUPS = ("A", "B")
REVIEW_COUNTS = {("system", "accept"): "accepted", ("system", "reject"): "rejected", ("added", "accept"): "added"}
MAX_BODY_BYTES = 1_000_000
SECONDS_DIGITS = 3
TRUE_WORDS = {"да", "true", "1", "истина"}
FALSE_WORDS = {"нет", "false", "0", "ложь"}
BOOLEAN_FIELDS = {"price_includes_vat"}
RU_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
NUMBER_SPACES = re.compile(r"[\s  ]")
UNSAFE_NAME_CHARS = re.compile(r"[^\w-]+")
TASK_FIELDS = ("purchase_id", "title", "owner", "due_date")


@dataclass(frozen=True)
class TimingConfig:
    """Что нужно серверу: кейсы, предсказания системы, порядок кейсов и папка результатов."""

    cases: dict[str, dict]
    predictions: dict[str, dict]
    case_ids: tuple[str, ...]
    out_dir: Path
    predictions_path: str
    clock: Callable[[], datetime]


def participant_parity(participant: str) -> int:
    """Имя участника -> 0 или 1; регистр и пробелы по краям не влияют."""
    digest = hashlib.sha256(participant.strip().casefold().encode("utf-8")).digest()
    return digest[0] % 2


def suggested_group(participant: str) -> str:
    """Имя участника -> группа по умолчанию: A при четном хэше, B при нечетном."""
    return GROUPS[participant_parity(participant)]


def require_group(value: object) -> str:
    """Внешний вход: 'A' или 'B', иначе ValueError."""
    if value not in GROUPS:
        raise ValueError(f"group must be one of {', '.join(GROUPS)}")
    return value


def assign_cases(group: str, case_ids: tuple[str, ...]) -> list[dict]:
    """Группа и порядок кейсов -> [{case_id, mode}]; A начинает с ручного ввода, B с проверки."""
    parity = GROUPS.index(group)
    return [
        {"case_id": case_id, "mode": MANUAL if (index + parity) % 2 == 0 else REVIEW}
        for index, case_id in enumerate(case_ids)
    ]


def assigned_mode(group: str, case_ids: tuple[str, ...], case_id: str) -> str:
    """Режим кейса для группы; кейс вне списка — ошибка запроса."""
    modes = {item["case_id"]: item["mode"] for item in assign_cases(group, case_ids)}
    if case_id not in modes:
        raise ValueError(f"Case {case_id} is not assigned")
    return modes[case_id]


def shown_episodes(case: dict) -> list[str]:
    """Эпизоды кейса до первого подтверждения пользователя включительно."""
    episodes = []
    for episode_id, step in SCORER.timeline(case):
        episodes.append(episode_id)
        if any(action["kind"] == "approve" for action in step.get("user_actions", [])):
            break
    return episodes


def visible_events(case: dict) -> list[dict]:
    """Кейс -> источники, которые видит участник, без служебных полей."""
    episodes = set(shown_episodes(case))
    keys = ("id", "occurred_at", "author", "modality", "media_time", "text")
    return [
        {key: event[key] for key in keys if key in event}
        for episode_id, step in SCORER.timeline(case)
        if episode_id in episodes
        for event in step.get("events", [])
    ]


def proposal_operations(case: dict, proposals: list[dict]) -> list[dict]:
    """Операции set_field и create_task из предложений, доступных к показанному эпизоду."""
    episodes = set(shown_episodes(case))
    return [
        op
        for proposal in proposals
        if SCORER.phase_for(case, proposal) in episodes
        for op in proposal["operations"]
        if op["op"] in {"set_field", "create_task"}
    ]


def gold_operations(case: dict) -> list[dict]:
    """Ожидаемые операции gold для показанной части кейса."""
    return proposal_operations(case, case["gold"]["proposals"])


def predicted_operations(case: dict, prediction: dict | None) -> list[dict]:
    """Операции системы для показанной части кейса; кейса нет в предсказаниях -> пусто."""
    if prediction is None:
        return []
    return [strip_evidence(op) for op in proposal_operations(case, prediction["gold"]["proposals"])]


def strip_evidence(op: dict) -> dict:
    """Операция -> только то, что участник правит: закупка, поле и значение или задача."""
    if op["op"] == "set_field":
        return {"op": "set_field", "purchase_id": op["purchase_id"], "field": op["field"], "value": op["value"]}
    return {"op": "create_task", "task": {key: op["task"].get(key) for key in TASK_FIELDS}}


def parse_number(raw: str) -> str:
    """'12 000,50' -> '12000.50'; не число остается текстом и не совпадет с gold."""
    cleaned = NUMBER_SPACES.sub("", raw).replace(",", ".")
    try:
        Decimal(cleaned)
    except InvalidOperation:
        return raw
    return cleaned


def parse_date(raw: str) -> str:
    """'15.10.2026' или '2026-10-15' -> '2026-10-15'; иначе исходный текст."""
    match = RU_DATE.fullmatch(raw)
    if match:
        day, month, year = (int(part) for part in match.groups())
        return date(year, month, day).isoformat()
    return raw


def parse_boolean(raw: str) -> bool | str:
    """'да' -> True, 'нет' -> False, иное остается текстом."""
    word = raw.casefold()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    return raw


def parse_value(field: str, raw: object) -> object:
    """Значение из формы -> значение в типе поля карточки; пустая строка -> None."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return None
    if field in SCORER.NUMERIC:
        return parse_number(text)
    if field.endswith("_date"):
        return parse_date(text)
    if field in BOOLEAN_FIELDS:
        return parse_boolean(text)
    return text


def require_objects(value: object, name: str) -> list[dict]:
    """Внешний вход: список JSON-объектов, иначе ValueError."""
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def require_text(value: object, name: str) -> str:
    """Внешний вход: непустая строка, иначе ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def normalize_operation(op: dict) -> dict:
    """Операция из страницы -> set_field или create_task в формате gold без цитат."""
    if op.get("op") == "set_field":
        field = require_text(op.get("field"), "field")
        purchase_id = require_text(op.get("purchase_id"), "purchase_id")
        return {
            "op": "set_field",
            "purchase_id": purchase_id,
            "field": field,
            "value": parse_value(field, op.get("value")),
        }
    if op.get("op") == "create_task":
        task = op.get("task")
        if not isinstance(task, dict):
            raise ValueError("create_task needs a task object")
        normalized = {key: parse_value(key, task.get(key, "")) for key in TASK_FIELDS}
        normalized["purchase_id"] = require_text(task.get("purchase_id"), "task.purchase_id")
        return {"op": "create_task", "task": normalized}
    raise ValueError(f"Unsupported operation: {op.get('op')!r}")


def same_value(field: str, left: object, right: object) -> bool:
    """Равенство значений по нормализации eval/score.py: деньги через Decimal, текст по NFC."""
    return SCORER.normalized(left, field) == SCORER.normalized(right, field)


def drop_unchanged(operations: list[dict], initial_state: dict) -> list[dict]:
    """Убирает set_field, которые после нормализации не меняют карточку: '12500' при '12500.00'."""
    purchases = initial_state["purchases"]

    def changes_card(op: dict) -> bool:
        if op["op"] != "set_field" or op["purchase_id"] not in purchases:
            return True
        current = purchases[op["purchase_id"]]["fields"].get(op["field"])
        return not same_value(op["field"], current, op["value"])

    return [op for op in operations if changes_card(op)]


def operation_atom(op: dict) -> tuple[str, str, str]:
    """Операция -> (закупка, поле, нормализованное значение) для сравнения с gold.

    У задачи сравниваются ответственный и срок: название — свободный текст, его
    совпадение без судьи не проверить.
    """
    if op["op"] == "set_field":
        value = SCORER.normalized(op["value"], op["field"])
        return op["purchase_id"], op["field"], json.dumps(value, ensure_ascii=False, sort_keys=True)
    task = op["task"]
    value = [SCORER.normalized(task.get("owner"), "owner"), SCORER.normalized(task.get("due_date"), "due_date")]
    return task["purchase_id"], "create_task", json.dumps(value, ensure_ascii=False)


def ratio(numerator: int, denominator: int) -> float | None:
    """Доля или None при пустом знаменателе."""
    return numerator / denominator if denominator else None


def score_operations(submitted: list[dict], gold: list[dict]) -> dict:
    """Операции участника и gold -> tp/fp/fn по атомам с учетом повторов."""
    actual = Counter(operation_atom(op) for op in submitted)
    expected = Counter(operation_atom(op) for op in gold)
    matched = actual & expected
    tp = sum(matched.values())
    return {
        "tp": tp,
        "fp": sum((actual - matched).values()),
        "fn": sum((expected - matched).values()),
        "precision": ratio(tp, actual.total()),
        "recall": ratio(tp, expected.total()),
        "extra": [list(atom) for atom in (actual - matched).elements()],
        "missed": [list(atom) for atom in (expected - matched).elements()],
    }


def parse_time(value: object, name: str) -> datetime:
    """ISO-метка браузера с часовым поясом -> datetime."""
    moment = datetime.fromisoformat(require_text(value, name))
    if moment.tzinfo is None:
        raise ValueError(f"{name} needs a timezone")
    return moment


def parse_pauses(raw: object, started: datetime, finished: datetime) -> list[tuple[datetime, datetime]]:
    """Паузы -> упорядоченные непересекающиеся интервалы внутри [начало, конец]."""
    pauses = [
        (parse_time(p.get("paused_at"), "paused_at"), parse_time(p.get("resumed_at"), "resumed_at"))
        for p in require_objects(raw, "pauses")
    ]
    previous_end = started
    for paused, resumed in pauses:
        if not previous_end <= paused <= resumed <= finished:
            raise ValueError("Pauses must be ordered, non-overlapping and inside the attempt")
        previous_end = resumed
    return pauses


def active_seconds(started: datetime, finished: datetime, pauses: list[tuple[datetime, datetime]]) -> float:
    """Время от «Начать» до «Готово» без пауз, в секундах."""
    paused = sum((resumed - start).total_seconds() for start, resumed in pauses)
    return round((finished - started).total_seconds() - paused, SECONDS_DIGITS)


def count_review_rows(raw: object) -> dict:
    """Строки проверки -> сколько принято, отклонено, исправлено и добавлено."""
    counts = Counter()
    for row in require_objects(raw, "review_rows"):
        key = REVIEW_COUNTS.get((row.get("source"), row.get("decision")))
        if key is None:
            raise ValueError(f"Bad review row: {row!r}")
        counts[key] += 1
        counts["edited"] += key == "accepted" and bool(row.get("edited"))
    return {key: counts[key] for key in ("accepted", "rejected", "edited", "added")}


def build_record(payload: dict, case: dict, group: str, mode: str, config: TimingConfig) -> dict:
    """Ответ страницы -> запись замера с активным временем и правильностью по gold."""
    started = parse_time(payload.get("started_at"), "started_at")
    finished = parse_time(payload.get("finished_at"), "finished_at")
    if finished < started:
        raise ValueError("finished_at is before started_at")
    pauses = parse_pauses(payload.get("pauses"), started, finished)
    raw_operations = require_objects(payload.get("operations"), "operations")
    operations = drop_unchanged([normalize_operation(op) for op in raw_operations], case["input"]["initial_state"])
    record = {
        "participant": require_text(payload.get("participant"), "participant"),
        "case_id": case["id"],
        "group": group,
        "mode": mode,
        "dataset_version": case.get("dataset_version"),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "active_seconds": active_seconds(started, finished, pauses),
        "wall_seconds": round((finished - started).total_seconds(), SECONDS_DIGITS),
        "pauses": [{"paused_at": p.isoformat(), "resumed_at": r.isoformat()} for p, r in pauses],
        "operations": operations,
        "correctness": score_operations(operations, gold_operations(case)),
        "saved_at": config.clock().isoformat(),
    }
    if mode == REVIEW:
        record["predictions"] = config.predictions_path
        record["proposed_operations"] = len(predicted_operations(case, config.predictions.get(case["id"])))
        record["review"] = count_review_rows(payload.get("review_rows", []))
    return record


def record_path(out_dir: Path, participant: str) -> Path:
    """Имя участника -> файл его замеров; небуквенные символы заменяются на '_'."""
    return out_dir / f"{UNSAFE_NAME_CHARS.sub('_', participant.strip())}.jsonl"


def load_records(paths: list[Path]) -> list[dict]:
    """JSONL-файлы замеров -> записи подряд."""
    return [
        json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def participant_records(out_dir: Path, participant: str) -> list[dict]:
    """Уже сданные замеры участника; файла нет -> пусто."""
    path = record_path(out_dir, participant)
    return load_records([path]) if path.exists() else []


def record_group(record: dict) -> str:
    """Группа записи; записи без поля group сделаны до выбора группы, когда режим решала четность хэша."""
    return record.get("group", suggested_group(record["participant"]))


def locked_group(records: list[dict], participant: str) -> str | None:
    """Группа, в которой участник уже сдавал кейсы; ничего не сдал -> None."""
    if not records:
        return None
    return record_group(records[0])


def resolve_group(out_dir: Path, participant: str, requested: object) -> str:
    """Выбранная группа, если она совпадает с закрепленной; иначе ValueError.

    Без этой проверки участник мог бы сменить группу и пройти кейс во втором режиме.
    """
    group = require_group(requested)
    locked = locked_group(participant_records(out_dir, participant), participant)
    if locked is not None and locked != group:
        raise ValueError(f"{participant} already submitted cases in group {locked}")
    return group


def done_case_ids(out_dir: Path, participant: str) -> set[str]:
    """Кейсы, которые участник уже сдал."""
    return {record["case_id"] for record in participant_records(out_dir, participant)}


def append_record(out_dir: Path, record: dict) -> Path:
    """Дописывает запись в runs/timing/<participant>.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = record_path(out_dir, record["participant"])
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("saved %s %s for %s to %s", record["mode"], record["case_id"], record["participant"], path)
    return path


def summarize_mode(records: list[dict]) -> dict:
    """Записи одного режима -> медиана и среднее времени, micro precision/recall."""
    times = [record["active_seconds"] for record in records]
    totals = Counter()
    for record in records:
        totals.update({key: record["correctness"][key] for key in ("tp", "fp", "fn")})
    exact = sum(record["correctness"]["fp"] == record["correctness"]["fn"] == 0 for record in records)
    return {
        "n": len(records),
        "participants": len({record["participant"] for record in records}),
        "median_active_seconds": statistics.median(times),
        "mean_active_seconds": round(statistics.fmean(times), SECONDS_DIGITS),
        **dict(totals),
        "precision": ratio(totals["tp"], totals["tp"] + totals["fp"]),
        "recall": ratio(totals["tp"], totals["tp"] + totals["fn"]),
        "exact_share": exact / len(records),
    }


def summarize_records(records: list[dict]) -> dict:
    """Все замеры -> сводка по режимам, группам, медианы по кейсам и отношение review/manual.

    Отношение медиан считается только при балансе: обе группы есть и каждый кейс пройден
    в обоих режимах. Иначе режим смешан с трудностью кейсов, ради чего и нужен контрбаланс.
    """
    by_mode = {mode: [r for r in records if r["mode"] == mode] for mode in (MANUAL, REVIEW)}
    modes = {mode: summarize_mode(rows) for mode, rows in by_mode.items() if rows}
    cases = {}
    for record in records:
        cases.setdefault(record["case_id"], {}).setdefault(record["mode"], []).append(record["active_seconds"])
    groups = {group: len({r["participant"] for r in records if record_group(r) == group}) for group in GROUPS}
    one_mode = sorted(case_id for case_id, per_mode in cases.items() if set(per_mode) != {MANUAL, REVIEW})
    summary = {
        "records": len(records),
        "modes": modes,
        "groups": groups,
        "cases_in_one_mode": one_mode,
        "balanced": all(groups.values()) and not one_mode,
        "cases": {
            case_id: {mode: statistics.median(times) for mode, times in sorted(per_mode.items())}
            for case_id, per_mode in sorted(cases.items())
        },
    }
    if summary["balanced"]:
        summary["review_to_manual_median"] = round(
            modes[REVIEW]["median_active_seconds"] / modes[MANUAL]["median_active_seconds"], SECONDS_DIGITS
        )
    return summary


def format_rate(value: float | None) -> str:
    """Доля -> '0.875' или '—', если считать не из чего."""
    return "—" if value is None else f"{value:.3f}"


def print_summary(summary: dict, path: Path) -> None:
    """Сводка -> таблица в терминал."""
    print(f"Замеров: {summary['records']}; сводка записана в {path}")
    print(f"{'режим':<8} {'n':>3} {'людей':>5} {'медиана, с':>11} {'среднее, с':>11} {'precision':>9} {'recall':>7}")
    for mode, row in summary["modes"].items():
        print(
            f"{mode:<8} {row['n']:>3} {row['participants']:>5} {row['median_active_seconds']:>11.1f} "
            f"{row['mean_active_seconds']:>11.1f} {format_rate(row['precision']):>9} {format_rate(row['recall']):>7}"
        )
    print("Участников по группам: " + ", ".join(f"{group} {n}" for group, n in summary["groups"].items()))
    if summary["balanced"]:
        print(f"Медиана review / медиана manual: {summary['review_to_manual_median']:.3f}")
    else:
        only = ", ".join(summary["cases_in_one_mode"]) or "нет"
        print(f"Нет баланса групп, отношение медиан не считается; кейсы в одном режиме: {only}")
    print("Название задачи не сравнивается: у задачи проверяются закупка, ответственный и срок")


def write_summary(out_dir: Path) -> tuple[dict, Path]:
    """runs/timing/*.jsonl -> summary.json рядом с замерами."""
    paths = sorted(out_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No timing records in {out_dir}")
    summary = summarize_records(load_records(paths))
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary, path


def case_view(case: dict, mode: str, prediction: dict | None) -> dict:
    """Кейс -> то, что видит участник: карточки, задачи, источники и дифф системы без gold."""
    view = {
        "case_id": case["id"],
        "mode": mode,
        "initial_state": case["input"]["initial_state"],
        "events": visible_events(case),
    }
    if mode == REVIEW:
        view["proposed_operations"] = predicted_operations(case, prediction)
        view["has_proposal"] = bool(view["proposed_operations"])
    return view


def group_view(config: TimingConfig, participant: str) -> dict:
    """Участник -> подсказка группы по хэшу и закрепленная группа, если он уже сдавал."""
    records = participant_records(config.out_dir, participant)
    return {"suggested": suggested_group(participant), "locked": locked_group(records, participant)}


def assignment_view(config: TimingConfig, participant: str, group: str) -> dict:
    """Участник и группа -> его кейсы с режимом и отметкой о сдаче."""
    done = done_case_ids(config.out_dir, participant)
    cases = [dict(item, done=item["case_id"] in done) for item in assign_cases(group, config.case_ids)]
    return {"participant": participant, "group": group, "cases": cases}


class TimingHandler(BaseHTTPRequestHandler):
    """HTTP-обработчик страницы замера; конфиг лежит в self.server.timing."""

    def do_GET(self) -> None:  # noqa: N802 - имя задает http.server
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        config = self.server.timing
        try:
            if url.path == "/":
                self.send_body(HTTPStatus.OK, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/group":
                self.send_json(HTTPStatus.OK, group_view(config, require_text(query.get("participant"), "participant")))
            elif url.path == "/api/cases":
                participant = require_text(query.get("participant"), "participant")
                group = resolve_group(config.out_dir, participant, query.get("group"))
                self.send_json(HTTPStatus.OK, assignment_view(config, participant, group))
            elif url.path == "/api/case":
                participant = require_text(query.get("participant"), "participant")
                group = resolve_group(config.out_dir, participant, query.get("group"))
                case_id = require_text(query.get("id"), "id")
                mode = assigned_mode(group, config.case_ids, case_id)
                self.send_json(HTTPStatus.OK, case_view(config.cases[case_id], mode, config.predictions.get(case_id)))
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - имя задает http.server
        if urlparse(self.path).path != "/result":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        config = self.server.timing
        try:
            payload = self.read_json()
            participant = require_text(payload.get("participant"), "participant")
            case_id = require_text(payload.get("case_id"), "case_id")
            group = resolve_group(config.out_dir, participant, payload.get("group"))
            mode = assigned_mode(group, config.case_ids, case_id)
            if case_id in done_case_ids(config.out_dir, participant):
                self.send_json(HTTPStatus.CONFLICT, {"error": f"{case_id} already submitted by {participant}"})
                return
            record = build_record(payload, config.cases[case_id], group, mode, config)
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        append_record(config.out_dir, record)
        self.send_json(HTTPStatus.OK, {"status": "saved"})

    def read_json(self) -> dict:
        """Тело POST -> JSON-объект; слишком большое или не объект — ValueError."""
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            raise ValueError("Request body is too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def send_json(self, status: HTTPStatus, body: dict) -> None:
        self.send_body(status, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def send_body(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug(fmt, *args)


def build_server(config: TimingConfig, port: int) -> HTTPServer:
    """Конфиг -> HTTP-сервер на 127.0.0.1; порт 0 выбирает свободный."""
    server = HTTPServer((HOST, port), TimingHandler)
    server.timing = config
    return server


def load_rows(path: Path) -> dict[str, dict]:
    """JSONL кейсов или предсказаний -> {id: строка}; ошибки разбора — ValueError."""
    rows, errors = SCORER.load_jsonl(path)
    if errors:
        raise ValueError(f"{path}: {errors[:3]}")
    return rows


def load_config(args: argparse.Namespace, clock: Callable[[], datetime]) -> TimingConfig:
    """Аргументы CLI -> конфиг сервера; кейсы без gold-операций и вне файла — ошибка."""
    cases = load_rows(args.cases)
    case_ids = tuple(args.ids.split(",")) if args.ids else DEFAULT_IDS
    missing = [case_id for case_id in case_ids if case_id not in cases]
    if missing:
        raise ValueError(f"Cases not found in {args.cases}: {missing}")
    predictions = load_rows(args.predictions)
    absent = [case_id for case_id in case_ids if case_id not in predictions]
    if absent:
        logger.warning("no system predictions for %s: review shows an empty diff", absent)
    return TimingConfig(
        cases={case_id: cases[case_id] for case_id in case_ids},
        predictions=predictions,
        case_ids=case_ids,
        out_dir=args.out_dir,
        predictions_path=display_path(args.predictions),
        clock=clock,
    )


def display_path(path: Path) -> str:
    """Путь -> относительный от корня репозитория, если файл внутри него."""
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(resolved)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--ids", help="ID кейсов через запятую; по умолчанию " + ",".join(DEFAULT_IDS))
    parser.add_argument("--predictions", type=Path, help="predictions.jsonl прогона системы")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", action="store_true", help="посчитать сводку по --out-dir и выйти")
    args = parser.parse_args(argv)
    if args.cases.resolve() == FORBIDDEN_CASES:
        parser.error("eval/test.jsonl is held out; time participants on dev cases")
    if not args.summary and args.predictions is None:
        parser.error("--predictions is required to serve the page")
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.summary:
        print_summary(*write_summary(args.out_dir))
        return
    server = build_server(load_config(args, lambda: datetime.now(UTC)), args.port)
    print(f"Страница замера: http://{HOST}:{server.server_port}/  (результаты в {args.out_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

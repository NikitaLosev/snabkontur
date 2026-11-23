"""Учебная mock-CRM на SQLite: карточки, задачи, журнал изменений и производные сводки.

Запись идет только через apply и undo. Каждое изменение повышает ревизию
затронутых карточек на единицу, повтор того же change_id ничего не меняет, откат
создает новую ревизию с обратными значениями.
"""

import json
import sqlite3
from datetime import date, datetime, timedelta

from snab.contracts import ToolResult

SCHEMA = """
CREATE TABLE purchases (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, fields TEXT NOT NULL);
CREATE TABLE tasks (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE changes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    change_id TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,
    proposal_id TEXT,
    target_change_id TEXT,
    status TEXT NOT NULL,
    effects TEXT NOT NULL
);
CREATE TABLE rejections (proposal_id TEXT PRIMARY KEY, reason TEXT);
CREATE TABLE derived_views (
    name TEXT PRIMARY KEY,
    purchase_id TEXT NOT NULL,
    fact_keys TEXT NOT NULL,
    view TEXT
);
"""
SUMMARY_WINDOW_DAYS = 2


class CommitFaultError(Exception):
    """Сбой записи, подставленный fault injection после первой операции пакета."""


def morning_summary(state: dict, now: str) -> dict:
    """Утренняя сводка по сохраненным карточкам: сегодня и два следующих дня."""
    window_start = datetime.fromisoformat(now).date()
    window_end = window_start + timedelta(days=SUMMARY_WINDOW_DAYS)
    open_tasks = [task for task in state["tasks"].values() if task["status"] == "open"]
    deliveries = []
    for purchase_id, card in sorted(state["purchases"].items()):
        delivery_date = card["fields"].get("delivery_date")
        if delivery_date and window_start <= date.fromisoformat(delivery_date) <= window_end:
            deliveries.append({"purchase_id": purchase_id, "delivery_date": delivery_date})
    return {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "overdue_task_ids": sorted(
            task["id"]
            for task in open_tasks
            if task["due_date"] and date.fromisoformat(task["due_date"]) < window_start
        ),
        "upcoming_task_ids": sorted(
            task["id"]
            for task in open_tasks
            if task["due_date"] and window_start <= date.fromisoformat(task["due_date"]) <= window_end
        ),
        "unscheduled_task_ids": sorted(task["id"] for task in open_tasks if not task["due_date"]),
        "deliveries": deliveries,
    }


def targets_intersect(left: dict, right: dict) -> bool:
    """Две цели пересекаются, если это одно поле или одна задача целиком."""
    same_object = left["kind"] == right["kind"] and left["id"] == right["id"]
    return same_object and (left["field"] is None or right["field"] is None or left["field"] == right["field"])


class MockCrm:
    """CRM одного кейса в памяти процесса."""

    def __init__(self) -> None:
        # Система с инструментами работает в потоке харнесса под дедлайном, чтения идут оттуда
        self.connection = sqlite3.connect(":memory:", autocommit=False, check_same_thread=False)
        self.connection.executescript(SCHEMA)

    def load(self, initial_state: dict, derived_views: dict | None = None) -> list[str]:
        """Загружает подтвержденное состояние; возвращает имена устаревших сводок."""
        stale_views = []
        with self.connection:
            for purchase_id, card in initial_state["purchases"].items():
                self.connection.execute(
                    "INSERT INTO purchases VALUES (?, ?, ?)",
                    (purchase_id, card["revision"], json.dumps(card["fields"], ensure_ascii=False)),
                )
            for task_id, task in initial_state["tasks"].items():
                self.connection.execute(
                    "INSERT INTO tasks VALUES (?, ?)", (task_id, json.dumps(task, ensure_ascii=False))
                )
            for name, view in (derived_views or {}).items():
                card = self.read_card(view["purchase_id"])
                is_current = card is not None and card["revision"] == view["based_on_revision"]
                if not is_current:
                    stale_views.append(name)
                self.connection.execute(
                    "INSERT INTO derived_views VALUES (?, ?, ?, ?)",
                    (
                        name,
                        view["purchase_id"],
                        json.dumps(list(view["facts"])),
                        json.dumps(view, ensure_ascii=False) if is_current else None,
                    ),
                )
        return stale_views

    def read_card(self, purchase_id: str) -> dict | None:
        """Карточка с ревизией или None."""
        row = self.connection.execute("SELECT revision, fields FROM purchases WHERE id = ?", (purchase_id,)).fetchone()
        return None if row is None else {"revision": row[0], "fields": json.loads(row[1])}

    def read_task(self, task_id: str) -> dict | None:
        """Задача или None."""
        row = self.connection.execute("SELECT data FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return None if row is None else json.loads(row[0])

    def snapshot(self) -> dict:
        """Полное состояние в формате gold.checks[].expected_state."""
        purchases = {
            purchase_id: {"revision": revision, "fields": json.loads(fields)}
            for purchase_id, revision, fields in self.connection.execute(
                "SELECT id, revision, fields FROM purchases ORDER BY id"
            )
        }
        tasks = {
            task_id: json.loads(data)
            for task_id, data in self.connection.execute("SELECT id, data FROM tasks ORDER BY id")
        }
        return {"purchases": purchases, "tasks": tasks}

    def derived_views(self) -> dict:
        """Производные представления; устаревшее представление хранится как None."""
        return {
            name: None if view is None else json.loads(view)
            for name, view in self.connection.execute("SELECT name, view FROM derived_views ORDER BY name")
        }

    def find_change(self, change_id: str) -> dict | None:
        """Запись журнала по change_id или None."""
        row = self.connection.execute(
            "SELECT seq, kind, status, effects FROM changes WHERE change_id = ?", (change_id,)
        ).fetchone()
        if row is None:
            return None
        return {"seq": row[0], "kind": row[1], "status": row[2], "effects": json.loads(row[3])}

    def apply(
        self,
        operations: list[dict],
        change_id: str,
        expected_revision: dict[str, int],
        proposal_id: str | None = None,
        kind: str = "approve",
        fail_after_first_operation: bool = False,
    ) -> ToolResult:
        """Применяет пакет целиком или никак. Повтор change_id не пишет второй раз."""
        existing = self.find_change(change_id)
        if existing is not None:
            return ToolResult.success(
                status="already_applied", change_id=change_id, change_status=existing["status"], invalidated_views=[]
            )
        touched = self.touched_purchases(operations)
        if isinstance(touched, ToolResult):
            return touched
        for purchase_id in sorted(touched):
            current = self.read_card(purchase_id)["revision"]
            if expected_revision.get(purchase_id) != current:
                return ToolResult.failure(
                    "conflict",
                    f"{purchase_id}: expected revision {expected_revision.get(purchase_id)}, current {current}",
                    purchase_id=purchase_id,
                    current_revision=current,
                )
        try:
            with self.connection:
                effects = []
                for index, operation in enumerate(operations):
                    effects.extend(self.write_operation(operation))
                    if fail_after_first_operation and index == 0:
                        raise CommitFaultError(change_id)
                invalidated = self.finish_change(change_id, kind, proposal_id, None, effects, touched)
        except CommitFaultError:
            return ToolResult.failure(
                "write_failed", "commit failed after the first operation; rolled back", change_id=change_id
            )
        return ToolResult.success(
            status="applied",
            change_id=change_id,
            invalidated_views=invalidated,
            revisions={pid: self.read_card(pid)["revision"] for pid in sorted(touched)},
        )

    def touched_purchases(self, operations: list[dict]) -> set[str] | ToolResult:
        """Карточки, ревизия которых вырастет; ошибка, если объект не найден."""
        touched = set()
        for operation in operations:
            if operation["op"] == "set_field":
                purchase_id = operation["purchase_id"]
            elif operation["op"] == "create_task":
                task_id, purchase_id = operation["task"]["id"], operation["task"]["purchase_id"]
                if self.read_task(task_id) is not None:
                    return ToolResult.failure("invalid_args", f"task {task_id} already exists")
            else:
                task = self.read_task(operation["task_id"])
                if task is None:
                    return ToolResult.failure("not_found", f"task {operation['task_id']} not found")
                purchase_id = task["purchase_id"]
            if self.read_card(purchase_id) is None:
                return ToolResult.failure("not_found", f"purchase {purchase_id} not found")
            touched.add(purchase_id)
        if not touched:
            return ToolResult.failure("invalid_args", "empty change")
        return touched

    def write_operation(self, operation: dict) -> list[dict]:
        """Пишет одну операцию и возвращает ее эффекты для журнала и отката."""
        evidence = operation.get("evidence", [])
        if operation["op"] == "set_field":
            purchase_id, field = operation["purchase_id"], operation["field"]
            fields = self.read_card(purchase_id)["fields"]
            effect = {
                "kind": "field",
                "id": purchase_id,
                "field": field,
                "purchase_id": purchase_id,
                "had_before": field in fields,
                "before": fields.get(field),
                "after": operation["value"],
                "evidence": evidence,
            }
            fields[field] = operation["value"]
            self.save_fields(purchase_id, fields)
            return [effect]
        if operation["op"] == "create_task":
            task = operation["task"]
            self.connection.execute(
                "INSERT INTO tasks VALUES (?, ?)", (task["id"], json.dumps(task, ensure_ascii=False))
            )
            return [
                {
                    "kind": "task",
                    "id": task["id"],
                    "field": None,
                    "purchase_id": task["purchase_id"],
                    "had_before": False,
                    "before": None,
                    "after": task,
                    "evidence": evidence,
                }
            ]
        task = self.read_task(operation["task_id"])
        effects = []
        for field, value in operation["fields"].items():
            effects.append(
                {
                    "kind": "task",
                    "id": task["id"],
                    "field": field,
                    "purchase_id": task["purchase_id"],
                    "had_before": field in task,
                    "before": task.get(field),
                    "after": value,
                    "evidence": evidence,
                }
            )
            task[field] = value
        self.connection.execute(
            "UPDATE tasks SET data = ? WHERE id = ?", (json.dumps(task, ensure_ascii=False), task["id"])
        )
        return effects

    def save_fields(self, purchase_id: str, fields: dict) -> None:
        """Сохраняет поля карточки без изменения ревизии."""
        self.connection.execute(
            "UPDATE purchases SET fields = ? WHERE id = ?", (json.dumps(fields, ensure_ascii=False), purchase_id)
        )

    def finish_change(
        self,
        change_id: str,
        kind: str,
        proposal_id: str | None,
        target_change_id: str | None,
        effects: list[dict],
        touched: set[str],
    ) -> list[str]:
        """Повышает ревизии, пишет журнал и сбрасывает сводки по затронутым карточкам."""
        for purchase_id in touched:
            self.connection.execute("UPDATE purchases SET revision = revision + 1 WHERE id = ?", (purchase_id,))
        self.connection.execute(
            "INSERT INTO changes (change_id, kind, proposal_id, target_change_id, status, effects) "
            "VALUES (?, ?, ?, ?, 'applied', ?)",
            (change_id, kind, proposal_id, target_change_id, json.dumps(effects, ensure_ascii=False)),
        )
        invalidated = []
        for purchase_id in sorted(touched):
            rows = self.connection.execute(
                "SELECT name FROM derived_views WHERE purchase_id = ? AND view IS NOT NULL", (purchase_id,)
            ).fetchall()
            invalidated.extend(name for (name,) in rows)
            self.connection.execute("UPDATE derived_views SET view = NULL WHERE purchase_id = ?", (purchase_id,))
        return invalidated

    def reject(self, proposal_id: str, reason: str | None = None) -> ToolResult:
        """Отклонение записывается в журнал и не меняет карточки."""
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO rejections VALUES (?, ?)", (proposal_id, reason))
        return ToolResult.success(status="rejected", proposal_id=proposal_id)

    def undo(self, change_id: str, undo_change_id: str) -> ToolResult:
        """Откат новой ревизией; останавливается, если поля позже правили."""
        target = self.find_change(change_id)
        if target is None:
            return ToolResult.failure("not_found", f"change {change_id} not found")
        if target["status"] == "undone" or self.find_change(undo_change_id) is not None:
            return ToolResult.success(status="already_undone", change_id=change_id, invalidated_views=[])
        # Отмененная позже правка и запись ее отката возвращают поле к значению цели: не блокируют
        later_rows = self.connection.execute(
            "SELECT effects FROM changes WHERE seq > ? AND status = 'applied' AND kind != 'undo'", (target["seq"],)
        ).fetchall()
        later_effects = [effect for (effects,) in later_rows for effect in json.loads(effects)]
        blocked = [
            effect for effect in target["effects"] if any(targets_intersect(effect, later) for later in later_effects)
        ]
        if blocked:
            return ToolResult.failure(
                "conflict",
                f"fields changed after {change_id}",
                purchase_id=blocked[0]["purchase_id"],
                fields=sorted({effect["field"] or effect["id"] for effect in blocked}),
            )
        with self.connection:
            for effect in reversed(target["effects"]):
                self.revert_effect(effect)
            self.connection.execute("UPDATE changes SET status = 'undone' WHERE change_id = ?", (change_id,))
            inverse = [dict(effect, before=effect["after"], after=effect["before"]) for effect in target["effects"]]
            touched = {effect["purchase_id"] for effect in target["effects"]}
            invalidated = self.finish_change(undo_change_id, "undo", None, change_id, inverse, touched)
        return ToolResult.success(status="undone", change_id=change_id, invalidated_views=invalidated)

    def revert_effect(self, effect: dict) -> None:
        """Возвращает значение до изменения, включая отсутствие поля или задачи."""
        if effect["kind"] == "field":
            fields = self.read_card(effect["id"])["fields"]
            if effect["had_before"]:
                fields[effect["field"]] = effect["before"]
            else:
                fields.pop(effect["field"], None)
            self.save_fields(effect["id"], fields)
        elif effect["field"] is None:
            self.connection.execute("DELETE FROM tasks WHERE id = ?", (effect["id"],))
        else:
            task = self.read_task(effect["id"])
            task[effect["field"]] = effect["before"]
            self.connection.execute(
                "UPDATE tasks SET data = ? WHERE id = ?", (json.dumps(task, ensure_ascii=False), effect["id"])
            )

    def field_evidence(self, purchase_id: str, field: str) -> list[dict]:
        """Цитаты последнего действующего изменения поля из журнала."""
        evidence = []
        for kind, effects in self.connection.execute(
            "SELECT kind, effects FROM changes WHERE status = 'applied' ORDER BY seq"
        ):
            if kind == "undo":
                continue
            for effect in json.loads(effects):
                if effect["kind"] == "field" and effect["id"] == purchase_id and effect["field"] == field:
                    evidence = effect["evidence"]
        return evidence

    def rebuild_summary(self, purchase_id: str) -> ToolResult:
        """Пересобирает сводку карточки по текущей ревизии и журналу оснований."""
        card = self.read_card(purchase_id)
        if card is None:
            return ToolResult.failure("not_found", f"purchase {purchase_id} not found")
        rows = self.connection.execute(
            "SELECT name, fact_keys FROM derived_views WHERE purchase_id = ?", (purchase_id,)
        ).fetchall() or [("summary", json.dumps(list(card["fields"])))]
        with self.connection:
            for name, fact_keys in rows:
                keys = json.loads(fact_keys)
                evidence = []
                for key in keys:
                    evidence.extend(item for item in self.field_evidence(purchase_id, key) if item not in evidence)
                view = {
                    "purchase_id": purchase_id,
                    "based_on_revision": card["revision"],
                    "facts": {key: card["fields"].get(key) for key in keys},
                    "evidence": evidence,
                }
                self.connection.execute(
                    "INSERT OR REPLACE INTO derived_views VALUES (?, ?, ?, ?)",
                    (name, purchase_id, fact_keys, json.dumps(view, ensure_ascii=False)),
                )
        return ToolResult.success(status="rebuilt", views=[name for name, _ in rows])

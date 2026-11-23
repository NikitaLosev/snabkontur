"""Трейс прогона: одна запись на шаг системы, вызов модели или инструмента, операцию CRM.

Записи внутри вызова системы ссылаются на него через parent_step, поэтому по трейсу видно,
на каком шаге эпизода остановился прогон.
"""

from datetime import UTC, datetime


class Tracer:
    """Собирает записи трейса одного кейса."""

    def __init__(self, run_id: str, case_id: str, repeat: int) -> None:
        self.run_id = run_id
        self.case_id = case_id
        self.repeat = repeat
        self.point = None
        self.parent = None
        self.records = []

    def record(
        self,
        kind: str,
        name: str,
        args: dict | None = None,
        result: dict | None = None,
        error: str | None = None,
        latency_ms: float = 0.0,
        idempotency_key: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        stop_reason: str | None = None,
        model: str | None = None,
        prompt_hash: str | None = None,
        prompt: str | None = None,
        output: str | None = None,
        ttft_ms: float | None = None,
    ) -> dict:
        """Добавляет шаг и возвращает его запись; error - код ошибки ToolResult или None."""
        record = {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "repeat": self.repeat,
            "episode": self.point,
            "step": len(self.records) + 1,
            "parent_step": self.parent,
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "kind": kind,
            "name": name,
            "args": args or {},
            "result": result,
            "error": error,
            "idempotency_key": idempotency_key,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 3),
            "stop_reason": stop_reason,
            "model": model,
            "prompt_hash": prompt_hash,
            "prompt": prompt,
            "output": output,
            "ttft_ms": ttft_ms,
        }
        self.records.append(record)
        return record

"""Контракты харнесса: результаты инструментов, причины остановки, входы и выходы систем.

Схемы предложений и снимков CRM берем из eval/score.py, чтобы раннер и оценщик
проверяли один и тот же формат.
"""

import importlib.util
import sys
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORER_PATH = REPO_ROOT / "eval" / "score.py"

ErrorCode = Literal[
    "not_found",
    "ambiguous",
    "conflict",
    "timeout",
    "budget_exhausted",
    "invalid_args",
    "invalid_output",
    "write_failed",
]
RETRYABLE_CODES = frozenset({"timeout", "write_failed"})


def load_scorer() -> ModuleType:
    """Загружает eval/score.py как модуль: в eval нет пакета, а схемы нужны раннеру."""
    spec = importlib.util.spec_from_file_location("snab_eval_score", SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCORER = load_scorer()
Proposal = SCORER.Proposal


class StopReason(StrEnum):
    """Почему система закончила шаг."""

    DONE = "done"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    INVALID_OUTPUT = "invalid_output"


class ToolError(BaseModel):
    """Типизированная ошибка инструмента или CRM."""

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    retryable: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Результат вызова: либо данные, либо ошибка, но не оба сразу."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def check_shape(self) -> "ToolResult":
        """Успех без ошибки, неуспех только с ошибкой."""
        if self.ok == (self.error is not None):
            raise ValueError("ok=True requires no error, ok=False requires an error")
        return self

    @classmethod
    def success(cls, **data: object) -> "ToolResult":
        """Успешный результат с данными."""
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, code: ErrorCode, message: str, **details: object) -> "ToolResult":
        """Ошибка; повторяемость определяется кодом."""
        error = ToolError(code=code, retryable=code in RETRYABLE_CODES, message=message, details=details)
        return cls(ok=False, error=error)


class EpisodeContext(BaseModel):
    """Все, что система видит на шаге. Gold, fault_injections и метаданные кейса, включая ID, сюда не попадают."""

    episode_id: str
    events: list[dict[str, Any]]
    seen_events: list[dict[str, Any]]
    crm_state: dict[str, Any]
    authority_registry: dict[str, Any] = Field(default_factory=dict)
    now: str | None = None
    working_context: dict[str, Any] = Field(default_factory=dict)
    derived_views: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """Шаг внутри системы для трейса: вызов модели или собственная операция системы."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["llm", "system"]
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str | None = None
    prompt_hash: str | None = None
    prompt: str | None = None
    output: str | None = None
    ttft_ms: float | None = None


class SystemOutput(BaseModel):
    """Ответ системы на эпизод или запрос пользователя."""

    proposals: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    answers: list[dict[str, Any]] = Field(default_factory=list)
    trace_steps: list[TraceStep] = Field(default_factory=list)
    stop_reason: StopReason = StopReason.DONE

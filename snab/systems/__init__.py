"""Системы, которые раннер сравнивает на одном наборе и одном контракте.

rules - B0 на правилах, llm_single (B1) - одно извлечение моделью Ollama на кусок,
agent (S1) - то же извлечение с ограниченной развилкой инструментов только для чтения,
hybrid (S1.2) - развилку решает код, а модель получает его привязку и кандидатов фактов правил.
B1, S1 и S1.2 идут одним путем кода и отличаются только инструментами и подсказками.
"""

from functools import partial
from typing import Protocol

from snab.contracts import EpisodeContext, SystemOutput
from snab.systems.llm import LlmSystem
from snab.systems.rules import RuleSystem
from snab.tools import ModelTools


class System(Protocol):
    """Система получает только допустимый контекст шага и инструменты только для чтения."""

    name: str

    def describe(self) -> dict:
        """Модель, параметры генерации, тексты и хэши промптов для meta.json."""

    def process_episode(self, context: EpisodeContext, tools: ModelTools) -> SystemOutput:
        """Новые источники эпизода -> предложения изменений и проблемы."""

    def answer_query(self, context: EpisodeContext, query: dict, tools: ModelTools) -> SystemOutput:
        """Контрольный запрос пользователя -> ответ по текущему состоянию."""


SYSTEMS = {
    "rules": RuleSystem,
    "llm_single": partial(LlmSystem, "llm_single", use_tools=False),
    "agent": partial(LlmSystem, "agent", use_tools=True),
    "hybrid": partial(LlmSystem, "hybrid", use_tools=False, code_routing=True),
}


def build_system(name: str) -> System:
    """Имя из CLI -> экземпляр системы."""
    return SYSTEMS[name]()

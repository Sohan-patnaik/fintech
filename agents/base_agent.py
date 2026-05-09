from abc import ABC, abstractmethod
from typing import Any
from core.logger import get_logger


class BaseAgent(ABC):
    """All agents inherit this. Enforces a common run(state) interface."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def run(self, state: dict) -> dict:
        """Execute agent logic. Receives shared LangGraph state, returns updated state."""

        def _safe_float(self, val: Any, default: float = 0.0) -> float:
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

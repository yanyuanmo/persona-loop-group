from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    def __init__(self, llm: Any, memory: Optional[Any] = None, checker: Optional[Any] = None):
        self.llm = llm
        self.memory = memory
        self.checker = checker

    @abstractmethod
    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        raise NotImplementedError

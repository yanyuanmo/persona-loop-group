from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    def __init__(self, llm: Any, memory: Optional[Any] = None, checker: Optional[Any] = None):
        self.llm = llm
        self.memory = memory
        self.checker = checker

    @staticmethod
    def _extract_prefixed_lines(context: str, prefix: str) -> List[str]:
        out: List[str] = []
        for line in context.splitlines():
            line = line.strip()
            if line.startswith(prefix):
                out.append(line[len(prefix):].strip())
        return out

    @abstractmethod
    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        raise NotImplementedError

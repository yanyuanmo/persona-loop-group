from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseMemory(ABC):
    @abstractmethod
    def add(self, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[str]:
        raise NotImplementedError

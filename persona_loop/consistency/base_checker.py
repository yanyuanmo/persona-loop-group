from __future__ import annotations

from abc import ABC, abstractmethod


class BaseChecker(ABC):
    @abstractmethod
    def score(self, premise: str, hypothesis: str) -> float:
        raise NotImplementedError

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, context: str) -> Optional[dict]:
        return None

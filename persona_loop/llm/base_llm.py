from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        raise NotImplementedError

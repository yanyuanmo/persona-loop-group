from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @staticmethod
    def build_message(prompt: str, context: str) -> str:
        return (
            "You are an assistant in a persona consistency benchmark.\n"
            "Answer the question based on the context below. Be concise.\n\n"
            "Priority rule: [PERSONA] and [CORRECTION] tags are authoritative ground truth. "
            "If they contradict anything in [HISTORY], trust [PERSONA]/[CORRECTION] first.\n\n"
            "Context tags you may see:\n"
            "- [PERSONA] Stable facts about a speaker (identity, preferences, background).\n"
            "- [MEMORY]  Retrieved snippets from earlier in the conversation.\n"
            "- [RECENT]  Most recent conversation turns.\n"
            "- [HISTORY] Conversation history.\n"
            "- [SUMMARY] Condensed summary of older conversation.\n"
            "- [CORRECTION] A detected inconsistency — prioritize factual accuracy.\n"
            "- [STYLE]   Soft style preferences (secondary to correctness).\n"
            "- [RAG]     Retrieved augmentation from memory.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{prompt}"
        )

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, context: str) -> Optional[dict]:
        return None

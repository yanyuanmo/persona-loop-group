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

    @staticmethod
    def build_roleplay_message(
        speaker_name: str,
        partner_name: str,
        partner_text: str,
        persona_summary: str,
        context_extra: str = "",
    ) -> tuple[str, str]:
        """Build (system_prompt, user_message) for roleplay/character generation.

        Returns a (system, user) tuple so that subclasses can pass them to
        a multi-turn chat API (system: describes the character; user: partner's line).
        """
        system = (
            f"You are {speaker_name}. "
            f"Stay fully in character at all times.\n\n"
            f"Your character description:\n{persona_summary}\n\n"
            "Priority rule: [PERSONA] and [CORRECTION] are authoritative ground truth. "
            "If they contradict anything in [HISTORY] or [MEMORY], trust [PERSONA]/[CORRECTION] first.\n\n"
            "Context tags you may see:\n"
            "- [PERSONA]    Fixed character description (identity, traits, background).\n"
            "- [MEMORY]     Retrieved snippets from past conversations.\n"
            "- [HISTORY]    Recent verbatim conversation turns.\n"
            "- [CORRECTION] A detected out-of-character statement — correct course.\n"
        )
        user_parts = []
        if context_extra.strip():
            user_parts.append(context_extra.strip())
        user_parts.append(f"{partner_name} says: \"{partner_text}\"")
        user_parts.append(f"Respond naturally as {speaker_name} in 1-3 sentences.")
        return system, "\n\n".join(user_parts)

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        raise NotImplementedError

    def generate_roleplay(
        self,
        speaker_name: str,
        partner_name: str,
        partner_text: str,
        persona_summary: str,
        context_extra: str = "",
    ) -> str:
        """Generate a roleplay response as speaker_name.

        Default implementation falls back to generate() using the formatted
        context so subclasses that don't override this still work.
        """
        system, user = self.build_roleplay_message(
            speaker_name=speaker_name,
            partner_name=partner_name,
            partner_text=partner_text,
            persona_summary=persona_summary,
            context_extra=context_extra,
        )
        # Combine into a single prompt for the generic generate() path.
        combined_context = system
        combined_prompt = user
        return self.generate(prompt=combined_prompt, context=combined_context)

    def generate_json(self, prompt: str, context: str) -> Optional[dict]:
        return None

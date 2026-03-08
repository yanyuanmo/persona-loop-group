from __future__ import annotations

from persona_loop.llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def generate(self, prompt: str, context: str) -> str:
        # Placeholder response for skeleton mode.
        return f"[openai:{self.model_name}] {prompt} | ctx={context[:80]}"

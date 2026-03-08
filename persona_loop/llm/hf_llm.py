from __future__ import annotations

from persona_loop.llm.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    def generate(self, prompt: str, context: str) -> str:
        # Placeholder response for skeleton mode.
        return f"[hf:{self.model_name}] {prompt} | ctx={context[:80]}"

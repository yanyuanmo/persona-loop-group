from __future__ import annotations

import os

from persona_loop.llm.base_llm import BaseLLM


class KimiLLM(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Kimi backend requires openai package. Install with: pip install openai"
            ) from exc

        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            raise RuntimeError("KIMI_API_KEY is not set.")

        base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, context: str) -> str:
        message = self.build_message(prompt=prompt, context=context)
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": message}],
            temperature=0,
            max_tokens=256,
        )
        return (resp.choices[0].message.content or "").strip()

from __future__ import annotations

import os
from typing import Any

from persona_loop.llm.base_llm import BaseLLM

_UNSET: Any = object()  # sentinel: distinguish "not passed" from explicit None/""


class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, base_url: Any = _UNSET, api_key: Any = _UNSET):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI backend requires openai package. Install with: pip install openai"
            ) from exc

        resolved_api_key = api_key if api_key is not _UNSET else os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        if base_url is _UNSET:
            # Not explicitly passed: inherit OPENAI_BASE_URL env var (may be None).
            resolved_base_url = os.getenv("OPENAI_BASE_URL") or None
        else:
            # Explicitly passed: '' means real OpenAI (no override); any URL means use it.
            resolved_base_url = base_url or None

        self._client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    def generate(self, prompt: str, context: str) -> str:
        message = self.build_message(prompt=prompt, context=context)
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": message}],
            temperature=0,
            max_tokens=256,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate_roleplay(
        self,
        speaker_name: str,
        partner_name: str,
        partner_text: str,
        persona_summary: str,
        context_extra: str = "",
    ) -> str:
        """Override: use system+user message structure for roleplay."""
        system, user = self.build_roleplay_message(
            speaker_name=speaker_name,
            partner_name=partner_name,
            partner_text=partner_text,
            persona_summary=persona_summary,
            context_extra=context_extra,
        )
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        return (resp.choices[0].message.content or "").strip()

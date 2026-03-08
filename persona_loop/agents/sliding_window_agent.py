from __future__ import annotations

from typing import Any, Dict

from persona_loop.agents.base_agent import BaseAgent


class SlidingWindowAgent(BaseAgent):
    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        trimmed_context = context[-300:]
        response = self.llm.generate(prompt=prompt, context=trimmed_context)
        return {
            "agent": "sliding_window",
            "prompt": prompt,
            "context": trimmed_context,
            "response": response,
            "consistency": None,
        }

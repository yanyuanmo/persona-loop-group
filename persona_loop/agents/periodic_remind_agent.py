from __future__ import annotations

from typing import Any, Dict

from persona_loop.agents.base_agent import BaseAgent


class PeriodicRemindAgent(BaseAgent):
    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        response = self.llm.generate(prompt=prompt, context=context)
        return {
            "agent": "periodic_remind",
            "prompt": prompt,
            "context": context,
            "response": response,
            "consistency": None,
        }

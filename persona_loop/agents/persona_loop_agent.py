from __future__ import annotations

from typing import Any, Dict

from persona_loop.agents.base_agent import BaseAgent


class PersonaLoopAgent(BaseAgent):
    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        response = self.llm.generate(prompt=prompt, context=context)

        consistency_score = None
        if self.checker is not None:
            consistency_score = self.checker.score(premise=context, hypothesis=response)

        if self.memory is not None:
            self.memory.add(text=response)

        return {
            "agent": "persona_loop",
            "prompt": prompt,
            "context": context,
            "response": response,
            "consistency": consistency_score,
        }

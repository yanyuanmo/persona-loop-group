"""ContinuousAgent — baseline: accumulate full conversation history in context.

Context structure passed by caller (QA mode):
    [PERSONA] <fixed persona description text>
    [HISTORY] turn 1
    [HISTORY] turn 2
    ...

The agent concatenates all history into a single growing context window and
generates a response. No memory database, no context reset.
"""
from __future__ import annotations

from typing import Any, Dict, List

from persona_loop.agents.base_agent import BaseAgent


class ContinuousAgent(BaseAgent):
    def __init__(self, llm: Any, max_history: int = 20):
        super().__init__(llm=llm)
        self.max_history = max_history
        self._history: List[str] = []

    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        response = self.llm.generate(prompt=prompt, context=context)
        return {
            "agent": "continuous",
            "response": response,
            "context_used": context,
        }

    def run_roleplay_turn(
        self,
        speaker_name: str,
        partner_name: str,
        partner_text: str,
        persona_summary: str,
    ) -> Dict[str, Any]:
        """Generate a roleplay response as speaker_name, with windowed history."""
        hist = self._history[-self.max_history:] if self.max_history > 0 else self._history
        context_extra = "\n".join(f"[HISTORY] {h}" for h in hist)
        response = self.llm.generate_roleplay(
            speaker_name=speaker_name,
            partner_name=partner_name,
            partner_text=partner_text,
            persona_summary=persona_summary,
            context_extra=context_extra,
        )
        self._history.append(f"{partner_name}: {partner_text}\n{speaker_name}: {response}")
        return {
            "agent": "continuous",
            "response": response,
            "loop_reset": False,
            "loop_corrections_count": 0,
            "loop_retrieved_count": 0,
        }

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

from typing import Any, Dict

from persona_loop.agents.base_agent import BaseAgent


class ContinuousAgent(BaseAgent):
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
        """Generate a roleplay response as speaker_name (no loop, no memory)."""
        response = self.llm.generate_roleplay(
            speaker_name=speaker_name,
            partner_name=partner_name,
            partner_text=partner_text,
            persona_summary=persona_summary,
            context_extra="",
        )
        return {
            "agent": "continuous",
            "response": response,
            "loop_reset": False,
            "loop_corrections_count": 0,
            "loop_retrieved_count": 0,
        }

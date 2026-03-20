from __future__ import annotations

from typing import Any, Dict, List

from persona_loop.agents.base_agent import BaseAgent


class PeriodicRemindAgent(BaseAgent):
    def __init__(
        self,
        llm: Any,
        memory: Any = None,
        checker: Any = None,
        reminder_interval: int = 4,
        reminder_max_persona_lines: int = 4,
    ):
        super().__init__(llm=llm, memory=memory, checker=checker)
        self.reminder_interval = max(1, int(reminder_interval))
        self.reminder_max_persona_lines = max(1, int(reminder_max_persona_lines))
        self._turn_count = 0
        self._persona_memory: List[str] = []

    @staticmethod
    def _strip_prefix_block(context: str, prefix: str) -> str:
        kept: List[str] = []
        for line in context.splitlines():
            if not line.strip().startswith(prefix):
                kept.append(line)
        return "\n".join(kept).strip()

    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        self._turn_count += 1

        persona_lines = self._extract_prefixed_lines(context=context, prefix="[PERSONA]")
        if persona_lines:
            self._persona_memory = persona_lines

        # Base context excludes per-turn persona injection; persona reminders are periodic.
        base_context = self._strip_prefix_block(context=context, prefix="[PERSONA]")
        use_context = base_context
        if self._persona_memory and self._turn_count % self.reminder_interval == 0:
            reminder_lines = self._persona_memory[: self.reminder_max_persona_lines]
            reminder_block = "\n".join(f"[PERSONA_REMINDER] {x}" for x in reminder_lines)
            use_context = f"{reminder_block}\n{base_context}".strip()

        response = self.llm.generate(prompt=prompt, context=use_context)
        return {
            "agent": "periodic_remind",
            "prompt": prompt,
            "context": use_context,
            "response": response,
            "consistency": None,
        }

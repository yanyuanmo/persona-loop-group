from __future__ import annotations

from typing import Any, Dict, List

from persona_loop.agents.base_agent import BaseAgent


class SlidingWindowAgent(BaseAgent):
    def __init__(
        self,
        llm: Any,
        memory: Any = None,
        checker: Any = None,
        window_size: int = 4,
        summary_tail_items: int = 2,
    ):
        super().__init__(llm=llm, memory=memory, checker=checker)
        self.window_size = max(1, int(window_size))
        self.summary_tail_items = max(1, int(summary_tail_items))
        self._rolling_summary = ""

    @staticmethod
    def _extract_non_history_lines(context: str) -> List[str]:
        kept: List[str] = []
        for line in context.splitlines():
            s = line.strip()
            if s.startswith("[HISTORY]") or s.startswith("[SUMMARY]"):
                continue
            if s:
                kept.append(s)
        return kept

    @staticmethod
    def _shorten(text: str, max_len: int = 220) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."

    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        history_lines = self._extract_prefixed_lines(context=context, prefix="[HISTORY]")
        anchor_lines = self._extract_non_history_lines(context=context)

        dropped = history_lines[:-self.window_size] if len(history_lines) > self.window_size else []
        recent = history_lines[-self.window_size :] if history_lines else []

        if dropped:
            tail = dropped[-self.summary_tail_items :]
            tail_text = " | ".join(self._shorten(x, max_len=120) for x in tail)
            if self._rolling_summary:
                self._rolling_summary = self._shorten(f"{self._rolling_summary} | {tail_text}")
            else:
                self._rolling_summary = self._shorten(tail_text)

        blocks: List[str] = []
        blocks.extend(anchor_lines)
        if self._rolling_summary:
            blocks.append(f"[SUMMARY] {self._rolling_summary}")
        blocks.extend(f"[HISTORY] {x}" for x in recent)

        use_context = "\n".join(blocks).strip()
        response = self.llm.generate(prompt=prompt, context=use_context)

        return {
            "agent": "sliding_window",
            "prompt": prompt,
            "context": use_context,
            "response": response,
            "consistency": None,
        }

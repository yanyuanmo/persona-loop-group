from __future__ import annotations

from typing import List


def build_priority_context(
    persona_facts: List[str],
    corrections: List[str],
    history: List[str],
    recent_turn: str,
    max_items: int,
) -> str:
    if max_items <= 0:
        return ""

    persona_blocks = [f"[PERSONA] {x}" for x in persona_facts]
    correction_blocks = [f"[CORRECTION] {x}" for x in corrections]
    recent_block = f"[RECENT] {recent_turn}"

    # Always preserve the current user turn. Fill remaining budget with
    # persona/correction blocks first, then tail history.
    if max_items == 1:
        return recent_block

    budget_before_recent = max_items - 1
    prefix = (persona_blocks + correction_blocks)[:budget_before_recent]
    remaining = budget_before_recent - len(prefix)
    history_tail = [f"[HISTORY] {x}" for x in history[-remaining:]] if remaining > 0 else []

    return "\n".join(prefix + history_tail + [recent_block])

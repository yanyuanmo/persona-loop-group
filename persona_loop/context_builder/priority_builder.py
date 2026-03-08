from __future__ import annotations

from typing import List


def build_priority_context(
    persona_facts: List[str],
    corrections: List[str],
    history: List[str],
    recent_turn: str,
    max_items: int,
) -> str:
    blocks: List[str] = []

    # Priority order: persona -> corrections -> history -> recent turn
    blocks.extend([f"[PERSONA] {x}" for x in persona_facts])
    blocks.extend([f"[CORRECTION] {x}" for x in corrections])
    blocks.extend([f"[HISTORY] {x}" for x in history])
    blocks.append(f"[RECENT] {recent_turn}")

    return "\n".join(blocks[:max_items])

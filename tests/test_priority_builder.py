from persona_loop.context_builder.priority_builder import build_priority_context


def test_priority_builder_order():
    context = build_priority_context(
        persona_facts=["p1"],
        corrections=["c1"],
        history=["h1"],
        recent_turn="r1",
        max_items=4,
    )

    lines = context.splitlines()
    assert lines[0].startswith("[PERSONA]")
    assert lines[1].startswith("[CORRECTION]")
    assert lines[2].startswith("[HISTORY]")
    assert lines[3].startswith("[RECENT]")


def test_priority_builder_keeps_recent_when_truncated():
    context = build_priority_context(
        persona_facts=["p1", "p2"],
        corrections=["c1"],
        history=["h1", "h2", "h3"],
        recent_turn="r1",
        max_items=2,
    )

    lines = context.splitlines()
    assert len(lines) == 2
    assert lines[-1].startswith("[RECENT]")

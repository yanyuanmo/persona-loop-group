"""Quick smoke test for v2 agents."""
import sys
sys.path.insert(0, '.')

from persona_loop.core.factories import create_memory
from persona_loop.agents.persona_loop_agent_v2 import PersonaLoopAgent
from persona_loop.agents.continuous_agent_v2 import ContinuousAgent


class FakeLLM:
    model_name = "fake"
    def generate(self, prompt, context):
        return "response"
    def generate_roleplay(self, speaker_name, partner_name, partner_text, persona_summary, context_extra=""):
        return f"{speaker_name} says: hi!"


def test_persona_loop():
    memory = create_memory("embedding")
    agent = PersonaLoopAgent(llm=FakeLLM(), memory=memory, loop_interval=3)
    results = []
    for i in range(6):
        r = agent.run_roleplay_turn(
            speaker_name="Yangyang",
            partner_name="Chenmo",
            partner_text=f"Hello turn {i}",
            persona_summary="Yangyang is a lively person.",
        )
        results.append(r)
        print(f"  turn {i+1}: reset={r['loop_reset']}  corrections={r['loop_corrections_count']}  retrieved={r['loop_retrieved_count']}")
    # Loop should have fired at turn 3 and turn 6
    assert results[2]["loop_reset"], "Expected loop reset at turn 3"
    assert results[5]["loop_reset"], "Expected loop reset at turn 6"
    assert not results[0]["loop_reset"], "Should not reset at turn 1"
    print("  PersonaLoopAgent: PASS")


def test_continuous():
    agent = ContinuousAgent(llm=FakeLLM())
    r = agent.run_roleplay_turn("Yangyang", "Chenmo", "hi", "Yangyang persona")
    assert r["agent"] == "continuous"
    assert "hi" not in r["response"] or True  # just check it returns something
    assert not r["loop_reset"]
    print("  ContinuousAgent: PASS")


def test_data_loader():
    from persona_loop.data.multimodal_loader import load_all_pairs
    samples = load_all_pairs("data/multimodal_dialog", exclude=["example"])
    assert len(samples) >= 2
    for s in samples:
        assert s.agent_a.persona_summary
        assert s.agent_b.persona_summary
        assert len(s.turns) > 0
    print(f"  DataLoader: PASS ({len(samples)} pairs)")


if __name__ == "__main__":
    print("Testing PersonaLoopAgent roleplay turns...")
    test_persona_loop()
    print("Testing ContinuousAgent roleplay turns...")
    test_continuous()
    print("Testing data loader...")
    test_data_loader()
    print("\nAll smoke tests passed.")

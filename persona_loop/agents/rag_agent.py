from __future__ import annotations

from typing import Any, Dict

from persona_loop.agents.base_agent import BaseAgent


class RAGAgent(BaseAgent):
    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        retrieved = ""
        if self.memory is not None:
            hits = self.memory.search(query=prompt, top_k=2)
            retrieved = " | ".join(hits)

        merged_context = f"{context}\n[RAG]{retrieved}".strip()
        response = self.llm.generate(prompt=prompt, context=merged_context)
        if self.memory is not None:
            self.memory.add(text=response)
        return {
            "agent": "rag",
            "prompt": prompt,
            "context": merged_context,
            "response": response,
            "consistency": None,
        }

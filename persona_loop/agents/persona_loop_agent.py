from __future__ import annotations

import re
from typing import Any, Dict, List

from persona_loop.agents.base_agent import BaseAgent


class PersonaLoopAgent(BaseAgent):
    def __init__(
        self,
        llm: Any,
        memory: Any = None,
        checker: Any = None,
        loop_interval: int = 8,
        retrieval_top_k: int = 3,
        recent_turns: int = 3,
        nli_threshold: float = 0.2,
        max_corrections: int = 2,
        min_history_for_reset: int = 0,
        disable_persona_persist: bool = False,
        disable_nli_rerank: bool = False,
        disable_corrections: bool = False,
        reset_require_low_consistency: bool = False,
        rerank_relevance_weight: float = 0.45,
        rerank_support_weight: float = 0.55,
        summary_max_items: int = 0,
    ):
        super().__init__(llm=llm, memory=memory, checker=checker)
        self.loop_interval = max(1, int(loop_interval))
        self.retrieval_top_k = max(1, int(retrieval_top_k))
        self.recent_turns = max(1, int(recent_turns))
        self.nli_threshold = float(nli_threshold)
        self.max_corrections = max(0, int(max_corrections))
        self.min_history_for_reset = max(0, int(min_history_for_reset))
        self.disable_persona_persist = bool(disable_persona_persist)
        self.disable_nli_rerank = bool(disable_nli_rerank)
        self.disable_corrections = bool(disable_corrections)
        self.reset_require_low_consistency = bool(reset_require_low_consistency)
        self.rerank_relevance_weight = max(0.0, float(rerank_relevance_weight))
        self.rerank_support_weight = max(0.0, float(rerank_support_weight))
        self.summary_max_items = max(0, int(summary_max_items))

        # Fallback to default split if both weights are zero.
        if self.rerank_relevance_weight + self.rerank_support_weight <= 1e-9:
            self.rerank_relevance_weight = 0.45
            self.rerank_support_weight = 0.55

        self._turn_count = 0
        self._dialogue_buffer: List[str] = []

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']+", text.lower()))

    def _keyword_relevance(self, query: str, candidate: str) -> float:
        query_tokens = self._tokenize(query)
        candidate_tokens = self._tokenize(candidate)
        if not query_tokens or not candidate_tokens:
            return 0.0
        return len(query_tokens.intersection(candidate_tokens)) / max(1, len(query_tokens))

    def _select_persona_relevant_snippets(self, snippets: List[str], persona_text: str) -> List[str]:
        if not snippets:
            return []
        if not persona_text:
            return snippets

        selected: List[str] = []
        for snippet in snippets:
            relevance = self._keyword_relevance(persona_text, snippet)
            support = float(self.checker.score(premise=persona_text, hypothesis=snippet)) if self.checker is not None else 0.0
            snippet_tokens = self._tokenize(snippet)
            self_reference = bool({"i", "i'm", "im", "me", "my", "mine"}.intersection(snippet_tokens))
            if relevance > 0.0 or support >= max(-0.05, self.nli_threshold - 0.15) or self_reference:
                selected.append(snippet)

        if selected:
            return selected
        return snippets[-1:]

    def _rerank_retrieved_snippets(self, query: str, persona_text: str, candidates: List[str]) -> List[str]:
        if not candidates:
            return []

        ranked = []
        for idx, candidate in enumerate(candidates):
            relevance = self._keyword_relevance(query, candidate)
            support = 0.0
            if self.checker is not None and persona_text:
                support = float(self.checker.score(premise=persona_text, hypothesis=candidate))
            score = (self.rerank_relevance_weight * relevance) + (self.rerank_support_weight * support)
            ranked.append((score, idx, candidate))

        ranked.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        out: List[str] = []
        seen = set()
        for _, _, candidate in ranked:
            if candidate in seen:
                continue
            out.append(candidate)
            seen.add(candidate)
            if len(out) >= self.retrieval_top_k:
                break
        return out

    @staticmethod
    def _extract_prefixed_lines(context: str, prefix: str) -> List[str]:
        out: List[str] = []
        for line in context.splitlines():
            line = line.strip()
            if line.startswith(prefix):
                out.append(line[len(prefix) :].strip())
        return out

    def _history_count(self, context: str) -> int:
        return len(self._extract_prefixed_lines(context, "[HISTORY]"))

    def _should_reset_now(self, context: str) -> bool:
        history_count = self._history_count(context)
        cadence_ok = self._turn_count % self.loop_interval == 0
        if not cadence_ok or history_count < self.min_history_for_reset:
            return False

        if not self.reset_require_low_consistency:
            return True
        if self.checker is None:
            return True

        persona_facts = self._extract_prefixed_lines(context, "[PERSONA]")
        persona_text = " ".join(persona_facts).strip()
        if not persona_text:
            return True

        history_lines = self._extract_prefixed_lines(context, "[HISTORY]")
        probe = history_lines[-self.loop_interval :] if history_lines else self._dialogue_buffer[-self.loop_interval :]
        if not probe:
            return False

        # Skip reset only when all probed snippets are neutral-to-positive (no
        # contradiction detected). NLI scores are in [-1, +1] where negative means
        # contradiction. Using -threshold guards against spurious neutral lines
        # (score≈0) triggering resets when consistency is actually fine.
        contradiction_threshold = -max(0.0, self.nli_threshold)
        has_contradiction = False
        for snippet in probe:
            score = float(self.checker.score(premise=persona_text, hypothesis=snippet))
            if score < contradiction_threshold:
                has_contradiction = True
                break
        return has_contradiction

    def _build_mid_summary(self, prompt: str, persona_text: str, history_lines: List[str]) -> List[str]:
        if self.summary_max_items <= 0 or len(history_lines) <= self.recent_turns:
            return []

        middle = history_lines[: -self.recent_turns]
        if not middle:
            return []

        query = f"{prompt} {persona_text}".strip()
        ranked: List[tuple[float, int, str]] = []
        for idx, line in enumerate(middle):
            rel = self._keyword_relevance(query, line)
            support = 0.0
            if self.checker is not None and persona_text:
                support = float(self.checker.score(premise=persona_text, hypothesis=line))
            score = (0.7 * rel) + (0.3 * support)
            ranked.append((score, idx, line))

        ranked.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        out: List[str] = []
        seen = set()
        for _, _, line in ranked:
            short = line.replace("\n", " ").strip()
            if not short:
                continue
            short = short[:180]
            if short in seen:
                continue
            out.append(short)
            seen.add(short)
            if len(out) >= self.summary_max_items:
                break
        return out

    def _build_reinitialized_context(self, prompt: str, context: str) -> Dict[str, Any]:
        persona_facts = self._extract_prefixed_lines(context, "[PERSONA]")
        history_lines = self._extract_prefixed_lines(context, "[HISTORY]")
        persona_text = " ".join(persona_facts)

        # Stage A: persist recent K rounds into external memory.
        recent_k = history_lines[-self.loop_interval :] if history_lines else self._dialogue_buffer[-self.loop_interval :]
        persist_candidates = self._select_persona_relevant_snippets(recent_k, persona_text)
        if self.disable_persona_persist:
            persist_candidates = []
        if self.memory is not None:
            for snippet in persist_candidates:
                self.memory.add(text=snippet)

        # Stage B: detect low-consistency responses and convert them into compact repair hints.
        corrections: List[str] = []
        if (not self.disable_corrections) and self.checker is not None and persona_text:
            for snippet in recent_k:
                score = float(self.checker.score(premise=persona_text, hypothesis=snippet))
                if score < self.nli_threshold:
                    short = snippet[:120].replace("\n", " ").strip()
                    corrections.append(
                        (
                            "Potential mismatch with stable persona facts in recent context: "
                            f"'{short}'. Keep the next answer focused on the user's question and avoid contradictions."
                        )
                    )
                    if len(corrections) >= self.max_corrections:
                        break
        # Keep the instruction block compact to avoid flooding the reset context.
        corrections = list(dict.fromkeys(corrections))

        # Stage C: retrieve related long-term memory snippets.
        retrieved: List[str] = []
        if self.memory is not None:
            query = f"{prompt} {persona_text}".strip()
            candidate_pool = self.memory.search(query=query, top_k=max(self.retrieval_top_k * 3, self.retrieval_top_k))
            filtered_candidates = [x for x in candidate_pool if str(x).strip()]
            if self.disable_nli_rerank:
                retrieved = filtered_candidates[: self.retrieval_top_k]
            else:
                retrieved = self._rerank_retrieved_snippets(
                    query=query,
                    persona_text=persona_text,
                    candidates=filtered_candidates,
                )

        # Stage D: rebuild a fresh prompt context with clear priority ordering.
        rebuilt_blocks: List[str] = []
        summary_lines = self._build_mid_summary(prompt=prompt, persona_text=persona_text, history_lines=history_lines)
        rebuilt_blocks.extend([f"[PERSONA] {x}" for x in persona_facts])
        rebuilt_blocks.extend([f"[SUMMARY] {x}" for x in summary_lines])
        rebuilt_blocks.extend([f"[CORRECTION] {x}" for x in corrections])
        rebuilt_blocks.extend([f"[MEMORY] {x}" for x in retrieved])
        recent_blocks = history_lines[-self.recent_turns :] if history_lines else self._dialogue_buffer[-self.recent_turns :]
        rebuilt_blocks.extend([f"[RECENT] {x}" for x in recent_blocks])
        rebuilt_blocks.append(f"[USER] {prompt}")

        return {
            "context": "\n".join(rebuilt_blocks),
            "corrections": corrections,
            "retrieved": retrieved,
            "summary": summary_lines,
            "recent_persisted": len(persist_candidates),
        }

    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        self._turn_count += 1
        do_reset = self._should_reset_now(context)

        used_context = context
        corrections: List[str] = []
        retrieved: List[str] = []
        summary: List[str] = []
        recent_persisted = 0

        if do_reset:
            rebuilt = self._build_reinitialized_context(prompt=prompt, context=context)
            used_context = str(rebuilt.get("context", context))
            corrections = list(rebuilt.get("corrections", []))
            retrieved = list(rebuilt.get("retrieved", []))
            summary = list(rebuilt.get("summary", []))
            recent_persisted = int(rebuilt.get("recent_persisted", 0))

        response = self.llm.generate(prompt=prompt, context=used_context)

        # Keep a rolling dialogue buffer for the next reset cycle.
        self._dialogue_buffer.append(f"[USER] {prompt} [ASSISTANT] {response}")
        if len(self._dialogue_buffer) > self.loop_interval * 4:
            self._dialogue_buffer = self._dialogue_buffer[-self.loop_interval * 4 :]

        consistency_score = None
        # Score only on reset turns: non-reset turns mirror continuous behavior and
        # this avoids extra checker overhead on short contexts.
        if self.checker is not None and do_reset:
            consistency_score = self.checker.score(premise=used_context, hypothesis=response)

        return {
            "agent": "persona_loop",
            "prompt": prompt,
            "context": used_context,
            "response": response,
            "consistency": consistency_score,
            "loop_reset": do_reset,
            "loop_recent_persisted": recent_persisted,
            "loop_retrieved_count": len(retrieved),
            "loop_corrections_count": len(corrections),
            "loop_summary_count": len(summary),
        }

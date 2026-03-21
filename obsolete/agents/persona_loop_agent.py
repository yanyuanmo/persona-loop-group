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
        self.retrieval_top_k = max(0, int(retrieval_top_k))
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
        # Set to True during QA evaluation to prevent loop resets from injecting
        # accumulated (potentially erroneous) corrections into QA answers.
        self._eval_mode: bool = False

    @staticmethod
    def _extract_persona_owners(persona_facts: List[str]) -> List[str]:
        owners: List[str] = []
        seen = set()
        for fact in persona_facts:
            m = re.match(r"\s*\[([^\]]+)\]", str(fact))
            if not m:
                continue
            owner = m.group(1).strip()
            if not owner:
                continue
            key = owner.lower()
            if key in seen:
                continue
            owners.append(owner)
            seen.add(key)
        return owners

    @staticmethod
    def _owners_mentioned_in_text(text: str, owners: List[str]) -> List[str]:
        out: List[str] = []
        if not text:
            return out
        for owner in owners:
            if re.search(rf"\b{re.escape(owner)}\b", text, flags=re.IGNORECASE):
                out.append(owner)
        return out

    def _owner_aware_bonus(self, candidate: str, target_owners: List[str], all_owners: List[str]) -> float:
        if not all_owners:
            return 0.0

        mentioned = {
            owner.lower()
            for owner in self._owners_mentioned_in_text(candidate, all_owners)
        }
        if not mentioned:
            return 0.0

        target = {owner.lower() for owner in target_owners}
        if not target:
            return 0.0

        matched_target = len(mentioned.intersection(target))
        matched_non_target = len(mentioned - target)
        return (0.4 * matched_target) - (0.25 * matched_non_target)

    def _persona_premise_for_owners(self, persona_facts: List[str], target_owners: List[str]) -> str:
        if not persona_facts:
            return ""
        if not target_owners:
            return " ".join(persona_facts).strip()

        target = {owner.lower() for owner in target_owners}
        selected = []
        for fact in persona_facts:
            m = re.match(r"\s*\[([^\]]+)\]", str(fact))
            if not m:
                continue
            owner = m.group(1).strip().lower()
            if owner in target:
                selected.append(fact)
        if selected:
            return " ".join(selected).strip()
        return " ".join(persona_facts).strip()

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

    def _rerank_retrieved_snippets(
        self,
        query: str,
        prompt: str,
        persona_text: str,
        persona_facts: List[str],
        candidates: List[str],
    ) -> List[str]:
        if not candidates:
            return []

        all_owners = self._extract_persona_owners(persona_facts)
        target_owners = self._owners_mentioned_in_text(prompt, all_owners)
        premise_for_support = self._persona_premise_for_owners(persona_facts, target_owners)
        contradiction_threshold = -max(0.0, self.nli_threshold)

        owner_weight = 0.2
        contradiction_penalty_weight = 0.65

        ranked = []
        for idx, candidate in enumerate(candidates):
            relevance = self._keyword_relevance(query, candidate)
            support = 0.0
            if self.checker is not None and premise_for_support:
                support = float(self.checker.score(premise=premise_for_support, hypothesis=candidate))
            elif self.checker is not None and persona_text:
                support = float(self.checker.score(premise=persona_text, hypothesis=candidate))

            owner_bonus = self._owner_aware_bonus(
                candidate=candidate,
                target_owners=target_owners,
                all_owners=all_owners,
            )
            contradiction_penalty = 0.0
            if support < contradiction_threshold:
                contradiction_penalty = contradiction_penalty_weight * abs(support - contradiction_threshold)

            score = (
                (self.rerank_relevance_weight * relevance)
                + (self.rerank_support_weight * max(0.0, support))
                + (owner_weight * owner_bonus)
                - contradiction_penalty
            )
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

    def _history_count(self, context: str) -> int:
        return len(self._extract_prefixed_lines(context, "[HISTORY]"))

    def set_eval_mode(self, enabled: bool) -> None:
        """In eval mode, do memory retrieval on every QA turn (not just every
        loop_interval turns) while suppressing correction injection (Stage B).
        This maximises the benefit of embedded conversation memory at eval time
        without contaminating QA prompts with potentially-wrong corrections."""
        self._eval_mode = enabled

    def _should_reset_now(self, context: str) -> bool:
        # In eval mode always retrieve from memory; Stage B is still suppressed.
        if self._eval_mode and self.memory is not None:
            return True
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
        # In eval mode persist ALL turns (not just persona-relevant ones) so that
        # temporal/event answer turns are also available for Stage C retrieval.
        # In production mode keep the original persona-relevance filter to avoid
        # polluting long-running memory with unrelated dialogue.
        if self._eval_mode:
            persist_candidates = [s for s in recent_k if str(s).strip()]
        else:
            persist_candidates = self._select_persona_relevant_snippets(recent_k, persona_text)
        if self.disable_persona_persist:
            persist_candidates = []
        if self.memory is not None:
            for snippet in persist_candidates:
                self.memory.add(text=snippet)

        # Stage B: detect low-consistency responses and convert them into compact repair hints.
        # Skipped in eval mode — QA evaluation history is conversation turns, not agent
        # answers, so correction hints based on them are noisy and inflate contradiction.
        contradiction_threshold = -max(0.0, self.nli_threshold)
        corrections: List[str] = []
        if (not self.disable_corrections) and (not self._eval_mode) and self.checker is not None and persona_text:
            for snippet in recent_k:
                score = float(self.checker.score(premise=persona_text, hypothesis=snippet))
                if score < contradiction_threshold:
                    short = snippet[:120].replace("\n", " ").strip()
                    corrections.append(f"Mismatch: '{short}'")
                    if len(corrections) >= self.max_corrections:
                        break
        corrections = list(dict.fromkeys(corrections))

        # Stage C: retrieve related long-term memory snippets.
        retrieved: List[str] = []
        if self.memory is not None:
            # In eval mode use only the question as the retrieval query so that
            # we fetch the conversation turns most likely to contain the answer,
            # rather than generic persona-relevant snippets that may mislead.
            query = prompt if self._eval_mode else f"{prompt} {persona_text}".strip()
            candidate_pool = self.memory.search(query=query, top_k=max(self.retrieval_top_k * 3, self.retrieval_top_k))
            filtered_candidates = [x for x in candidate_pool if str(x).strip()]
            if self.disable_nli_rerank:
                retrieved = filtered_candidates[: self.retrieval_top_k]
            else:
                retrieved = self._rerank_retrieved_snippets(
                    query=query,
                    prompt=prompt,
                    persona_text=persona_text,
                    persona_facts=persona_facts,
                    candidates=filtered_candidates,
                )

        # Stage D: rebuild context with clear priority ordering.
        # Keep only the most recent `recent_turns` history lines verbatim; older
        # history has already been persisted to memory in Stage A and can be
        # recalled via Stage C retrieval.  This is the design from the proposal:
        # context = [PERSONA] + [CORRECTION] + [MEMORY retrieved] + [HISTORY tail].
        recent_history = history_lines[-self.recent_turns:] if history_lines else []
        # Any history beyond recent_turns was already stored in Stage A; Stage C
        # retrieves relevant snippets back as [MEMORY] so evidence is not lost.
        rebuilt_blocks: List[str] = []
        summary_lines = self._build_mid_summary(prompt=prompt, persona_text=persona_text, history_lines=history_lines)
        rebuilt_blocks.extend([f"[PERSONA] {x}" for x in persona_facts])
        rebuilt_blocks.extend([f"[SUMMARY] {x}" for x in summary_lines])
        rebuilt_blocks.extend([f"[CORRECTION] {x}" for x in corrections])
        rebuilt_blocks.extend([f"[MEMORY] {x}" for x in retrieved])
        rebuilt_blocks.extend([f"[HISTORY] {x}" for x in recent_history])
        # Note: prompt is passed separately to llm.generate(); do NOT add [USER] here.

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
        if self.checker is not None:
            # Score against persona facts instead of full rebuilt context. This avoids
            # long-premise truncation in NLI and gives a fair per-turn consistency signal.
            persona_lines = self._extract_prefixed_lines(used_context, "[PERSONA]")
            if not persona_lines:
                persona_lines = self._extract_prefixed_lines(context, "[PERSONA]")
            premise = " ".join(persona_lines).strip()
            if premise:
                consistency_score = self.checker.score(premise=premise, hypothesis=response)

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

"""PersonaLoopAgent — 周期性上下文重建的人格一致性 agent。

架构设计（严格按 proposal 图）：

    常规轮次（第 1..K-1 轮）：
        context = [PERSONA] + [HISTORY 最近若干轮]
        直接生成回复，不与外部记忆库交互。

    第 K / 2K / 3K 轮触发 Loop 重置：
        Stage A — 将最近 K 轮对话存入向量数据库（memory）
        Stage B — NLI 检查 agent 自己的回复是否与人格描述矛盾，生成修正提示
        Stage C — 从向量数据库检索与当前话题相关的历史片段
        Stage D — 重建 context:
                  [PERSONA] + [CORRECTION] + [MEMORY 检索片段] + [HISTORY 最近几轮]

关键设计原则：
- [PERSONA] = 固定的角色身份描述文本（由调用方在 session 开始时一次性传入，
              整个对话不变）。它 **不是** 从对话中提取的 slot-value facts。
- [MEMORY]  = 外部向量数据库召回的历史对话片段（动态，每次 loop 更新）。
- [HISTORY] = 最近 K 轮原始对话（verbatim），loop 后只保留 recent_turns 条。
- Stage B NLI 检测对象 = agent 自己说的话（response），而非 [HISTORY] 对话片段。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from persona_loop.agents.base_agent import BaseAgent


class PersonaLoopAgent(BaseAgent):
    """Persona Loop agent.

    Parameters
    ----------
    llm:
        语言模型实例（BaseLLM 子类）。
    memory:
        向量记忆库实例（BaseMemory 子类）。loop 重置时读写。
    checker:
        NLI 一致性检查器（BaseChecker 子类）。None 则跳过 Stage B。
    loop_interval : int
        每隔多少轮触发一次 loop 重置（即图中的 K）。默认 8。
    retrieval_top_k : int
        Stage C 从记忆库检索的片段数。默认 3。
    recent_turns : int
        Stage D 保留多少最近轮次的原始 [HISTORY]。默认 3。
    nli_threshold : float
        Stage B 矛盾判定阈值（NLI score < -threshold 视为矛盾）。默认 0.1。
    max_corrections : int
        Stage B 最多生成多少条修正提示。默认 2。
    """

    def __init__(
        self,
        llm: Any,
        memory: Any = None,
        checker: Any = None,
        loop_interval: int = 8,
        retrieval_top_k: int = 3,
        recent_turns: int = 3,
        nli_threshold: float = 0.1,
        max_corrections: int = 2,
        disable_persona_persist: bool = False,
        disable_corrections: bool = False,
    ):
        super().__init__(llm=llm, memory=memory, checker=checker)
        self.loop_interval = max(1, int(loop_interval))
        self.retrieval_top_k = max(0, int(retrieval_top_k))
        self.recent_turns = max(1, int(recent_turns))
        self.nli_threshold = float(nli_threshold)
        self.max_corrections = max(0, int(max_corrections))
        self.disable_persona_persist = bool(disable_persona_persist)
        self.disable_corrections = bool(disable_corrections)

        self._turn_count: int = 0
        # Rolling buffer: stores "[SPEAKER] text" strings for the current K-window.
        # Used by Stage A (persist to memory) and Stage B (NLI check on agent responses).
        self._recent_buffer: List[str] = []  # raw turn texts (speaker tagged)
        self._agent_responses: List[str] = []  # only agent's own responses (for Stage B)

    # ------------------------------------------------------------------
    # Core loop logic
    # ------------------------------------------------------------------

    def _should_reset(self) -> bool:
        return self._turn_count > 0 and (self._turn_count % self.loop_interval == 0)

    def _build_reset_context(self, persona_text: str, user_prompt: str) -> Dict[str, Any]:
        """Execute Stage A-D and return rebuilt context string + diagnostics."""

        # --- Stage A: persist recent K turns into external memory ---
        if self.memory is not None and not self.disable_persona_persist:
            for snippet in self._recent_buffer:
                self.memory.add(text=snippet)

        # --- Stage B: detect contradictions in agent's own responses ---
        corrections: List[str] = []
        if not self.disable_corrections and self.checker is not None and persona_text.strip() and self.max_corrections > 0:
            threshold = -max(0.0, self.nli_threshold)
            # Collect all contradicting responses with their scores first,
            # then sort by severity (most contradictory first) and cap at
            # min(loop_interval, 5) to avoid context bloat.
            flagged: List[tuple] = []
            for response in self._agent_responses:
                score = float(self.checker.score(premise=persona_text, hypothesis=response))
                if score < threshold:
                    flagged.append((score, response))
            flagged.sort(key=lambda x: x[0])  # lowest score = most contradictory
            cap = min(self.loop_interval, 5)
            for score, response in flagged[:cap]:
                short = response[:120].replace("\n", " ").strip()
                corrections.append(f"CORRECTION: You said '{short}' — this contradicts your persona. Please maintain consistency.")

        # --- Stage C: retrieve relevant history from memory ---
        retrieved: List[str] = []
        if self.memory is not None and self.retrieval_top_k > 0:
            retrieved = self.memory.search(query=user_prompt, top_k=self.retrieval_top_k)

        # --- Stage D: rebuild context ---
        # Priority (high → low): [PERSONA] > [CORRECTION] > [MEMORY] > [HISTORY]
        blocks: List[str] = []
        blocks.append(f"[PERSONA] {persona_text}")
        for c in corrections:
            blocks.append(f"[CORRECTION] {c}")
        for m in retrieved:
            blocks.append(f"[MEMORY] {m}")
        # Keep only the most recent `recent_turns` verbatim turns.
        recent_history = self._recent_buffer[-self.recent_turns:]
        for h in recent_history:
            blocks.append(f"[HISTORY] {h}")

        # Reset the buffer for the next K-window.
        self._recent_buffer = []
        self._agent_responses = []

        return {
            "context": "\n".join(blocks),
            "corrections": corrections,
            "retrieved": retrieved,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_turn(self, prompt: str, context: str) -> Dict[str, Any]:
        """Process one dialogue turn (QA / benchmark mode).

        Parameters
        ----------
        prompt : str
            The user's current message.
        context : str
            Current context string.  Must contain a ``[PERSONA] ...`` line at
            the top — this is the fixed persona description that never changes.
            May also contain ``[HISTORY] ...`` lines for the current window.

        Returns
        -------
        dict with keys: agent, response, context_used, loop_reset,
                        loop_corrections_count, loop_retrieved_count.
        """
        self._turn_count += 1

        # Extract the fixed persona text from context (passed by caller).
        persona_lines = self._extract_prefixed_lines(context, "[PERSONA]")
        persona_text = " ".join(persona_lines).strip()

        loop_reset = False
        corrections: List[str] = []
        retrieved: List[str] = []

        if self._should_reset():
            rebuilt = self._build_reset_context(persona_text=persona_text, user_prompt=prompt)
            context = rebuilt["context"]
            corrections = rebuilt["corrections"]
            retrieved = rebuilt["retrieved"]
            loop_reset = True

        response = self.llm.generate(prompt=prompt, context=context)

        # Update rolling buffers for this turn.
        self._recent_buffer.append(f"{prompt} | {response}")
        self._agent_responses.append(response)
        # Cap buffer to avoid unbounded growth between resets.
        max_buf = self.loop_interval * 2
        if len(self._recent_buffer) > max_buf:
            self._recent_buffer = self._recent_buffer[-max_buf:]
        if len(self._agent_responses) > max_buf:
            self._agent_responses = self._agent_responses[-max_buf:]

        return {
            "agent": "persona_loop",
            "response": response,
            "context_used": context,
            "loop_reset": loop_reset,
            "loop_corrections_count": len(corrections),
            "loop_retrieved_count": len(retrieved),
        }

    def run_roleplay_turn(
        self,
        speaker_name: str,
        partner_name: str,
        partner_text: str,
        persona_summary: str,
    ) -> Dict[str, Any]:
        """Process one dialogue turn in roleplay (character-generation) mode.

        Unlike run_turn (QA mode), this method:
        - Calls ``llm.generate_roleplay`` so the LLM responds *in character*.
        - Stores history as "{name}: {text}" lines suitable for roleplay context.
        - Still runs Stage A-D loop logic at every ``loop_interval`` turn.

        Parameters
        ----------
        speaker_name : str   Name of the agent whose turn it is (e.g. "Yangyang").
        partner_name : str   Name of the other speaker (e.g. "Chenmo").
        partner_text : str   The partner's latest utterance.
        persona_summary : str   Fixed persona description for speaker_name.

        Returns
        -------
        dict with keys: agent, response, loop_reset, loop_corrections_count,
                        loop_retrieved_count.
        """
        self._turn_count += 1

        loop_reset = False
        corrections: List[str] = []
        retrieved: List[str] = []

        if self._should_reset():
            rebuilt = self._build_reset_context(
                persona_text=persona_summary, user_prompt=partner_text
            )
            corrections = rebuilt["corrections"]
            retrieved = rebuilt["retrieved"]
            loop_reset = True
            # _recent_buffer and _agent_responses are cleared inside _build_reset_context

        # Build context_extra from corrections + retrieved memory + recent history.
        context_parts: List[str] = []
        for c in corrections:
            context_parts.append(f"[CORRECTION] {c}")
        for m in retrieved:
            context_parts.append(f"[MEMORY] {m}")
        # Append recent history window (may be empty right after a reset).
        for h in self._recent_buffer[-self.recent_turns:]:
            context_parts.append(f"[HISTORY] {h}")
        context_extra = "\n".join(context_parts)

        response = self.llm.generate_roleplay(
            speaker_name=speaker_name,
            partner_name=partner_name,
            partner_text=partner_text,
            persona_summary=persona_summary,
            context_extra=context_extra,
        )

        # Update roleplay-mode rolling buffers.
        turn_str = f"{partner_name}: {partner_text}\n{speaker_name}: {response}"
        self._recent_buffer.append(turn_str)
        self._agent_responses.append(response)
        max_buf = self.loop_interval * 2
        if len(self._recent_buffer) > max_buf:
            self._recent_buffer = self._recent_buffer[-max_buf:]
        if len(self._agent_responses) > max_buf:
            self._agent_responses = self._agent_responses[-max_buf:]

        return {
            "agent": "persona_loop",
            "response": response,
            "loop_reset": loop_reset,
            "loop_corrections_count": len(corrections),
            "loop_corrections_texts": corrections,
            "loop_retrieved_count": len(retrieved),
        }

    def reset(self) -> None:
        """Reset agent state between conversations / samples."""
        self._turn_count = 0
        self._recent_buffer = []
        self._agent_responses = []
        if self.memory is not None:
            self.memory.reset()

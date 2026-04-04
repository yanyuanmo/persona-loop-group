"""run_multimodal_eval.py â€" Persona consistency evaluation on multimodal_dialog dataset.

Evaluates two agents (continuous vs persona_loop) on the new multimodal_dialog data.

What this script does:
    1. Load agent data (persona_summary, sessions, facts) from pair directories.
    2. For each agent role (agent_a / agent_b) in each pair:
       a. Set persona = agent.persona_summary  (fixed, never changes)
       b. Replay the full conversation turn by turn.
          - On each turn that belongs to the agent, call agent.run_roleplay_turn().
          - Partner turns use the ground-truth text (not generated).
       c. After replay, run PCS evaluation:
          - For each of the agent's responses, run NLI(premise=persona_summary, hypothesis=response)
          - Compute: entailment_avg, contradiction_avg, PCS = entailment - contradiction

Evaluation metrics:
    - persona_pcs          : mean(entailment - contradiction) across all response turns
    - persona_entailment   : mean entailment score
    - persona_contradiction: mean contradiction score
    - persona_any_contra_ratio: fraction of turns where contradiction > threshold
    - loop_resets          : number of times the loop fired (persona_loop only)
    - loop_corrections     : total correction hints generated (persona_loop only)

Usage:
    python scripts/run_multimodal_eval.py \\
        --data data/multimodal_dialog \\
        --agent persona_loop \\
        --llm-provider openai \\
        --llm-model qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \\
        --memory-backend embedding \\
        --loop-interval 8 \\
        --output artifacts/multimodal_pl

    python scripts/run_multimodal_eval.py \\
        --data data/multimodal_dialog \\
        --agent continuous \\
        --llm-provider openai \\
        --llm-model qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \\
        --output artifacts/multimodal_cont
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows terminals that default to GBK/cp936.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_loop.data.multimodal_loader import AgentData, DialogSample, Turn, load_all_pairs
from persona_loop.agents.continuous_agent_v2 import ContinuousAgent
from persona_loop.agents.persona_loop_agent_v2 import PersonaLoopAgent
from persona_loop.core.factories import create_llm, create_memory, create_checker
from persona_loop.eval.nli_scorer import NLIScorer


# ---------------------------------------------------------------------------
# PCS computation (using NLIScorer directly)
# ---------------------------------------------------------------------------

def compute_pcs(
    responses: List[str],
    persona_summary: str,
    nli: NLIScorer,
    contradiction_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute Persona Consistency Score for a list of responses."""
    if not responses or not persona_summary.strip():
        return {
            "persona_pcs": 0.0,
            "persona_entailment": 0.0,
            "persona_contradiction": 0.0,
            "persona_contradiction_max": 0.0,
            "persona_any_contra_ratio": 0.0,
            "n_turns": 0,
        }

    entailments: List[float] = []
    contradictions: List[float] = []

    for resp in responses:
        if not resp.strip():
            continue
        scores = nli.score(premise=persona_summary, hypothesis=resp)
        entailments.append(float(scores.get("entailment", 0.0)))
        contradictions.append(float(scores.get("contradiction", 0.0)))

    if not entailments:
        return {
            "persona_pcs": 0.0,
            "persona_entailment": 0.0,
            "persona_contradiction": 0.0,
            "persona_contradiction_max": 0.0,
            "persona_any_contra_ratio": 0.0,
            "n_turns": 0,
        }

    pcs = mean(e - c for e, c in zip(entailments, contradictions))
    return {
        "persona_pcs": round(pcs, 4),
        "persona_entailment": round(mean(entailments), 4),
        "persona_contradiction": round(mean(contradictions), 4),
        "persona_contradiction_max": round(max(contradictions), 4),
        "persona_any_contra_ratio": round(
            sum(1 for c in contradictions if c >= contradiction_threshold) / len(contradictions), 4
        ),
        "n_turns": len(entailments),
    }


# ---------------------------------------------------------------------------
# Method B: LLM-as-Judge persona consistency scoring
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
Below is a character persona description and a single conversational response from that character.

Persona:
{persona}

Response:
{response}

Rate how consistent this response is with the persona on a scale of 1 to {scale}:
  1 = clearly contradicts or conflicts with the persona
  {mid} = neutral / no persona-relevant information
  {scale} = strongly consistent with and reflective of the persona

Reply with ONLY a single integer ({scale_range}). No explanation."""


def compute_pcs_judge(
    responses: List[str],
    persona_summary: str,
    llm: Any,
    scale: int = 5,
) -> Dict[str, Any]:
    """Use LLM-as-judge to score persona consistency (1-scale) for each response.

    Returns:
        judge_pcs      : normalized score in [-1, 1] (mirrors NLI PCS sign convention)
        judge_pcs_avg  : raw 1-N average (more interpretable for humans)
        judge_scores   : per-turn integer scores
    """
    if not responses or not persona_summary.strip():
        return {"judge_pcs": 0.0, "judge_pcs_avg": 0.0, "judge_scores": [], "n_turns": 0}

    neutral = (scale + 1) / 2  # e.g. 3.0 for scale=5
    half = (scale - 1) / 2     # e.g. 2.0 for scale=5
    scores: List[int] = []

    for resp in responses:
        if not resp.strip():
            scores.append(int(neutral))
            continue
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            persona=persona_summary,
            response=resp,
            scale=scale,
            mid=int(neutral),
            scale_range=f"1-{scale}",
        )
        try:
            reply = llm.generate(prompt=prompt, context="").strip()
            m = re.search(r'\b([1-9])\b', reply)
            score = int(m.group(1)) if m else int(neutral)
            score = max(1, min(scale, score))
        except Exception:
            score = int(neutral)
        scores.append(score)

    if not scores:
        return {"judge_pcs": 0.0, "judge_pcs_avg": 0.0, "judge_scores": [], "n_turns": 0}

    avg_raw = mean(scores)
    normalized = [(s - neutral) / half for s in scores]
    pcs = mean(normalized)
    return {
        "judge_pcs": round(pcs, 4),
        "judge_pcs_avg": round(avg_raw, 4),
        "judge_scores": scores,
        "n_turns": len(scores),
    }


# ---------------------------------------------------------------------------
# Single agent evaluation on one AgentData + turns
# ---------------------------------------------------------------------------

def run_agent_on_sample(
    agent_data: AgentData,
    partner_name: str,
    turns: List[Turn],
    agent_name: str,
    llm: Any,
    memory_backend: Optional[str],
    checker_model: Optional[str],
    loop_interval: int,
    retrieval_top_k: int,
    recent_turns: int,
    nli_threshold: float,
    max_history_window: int,
    skip_nli: bool,
    nli: Optional[NLIScorer],
    judge_llm: Optional[Any] = None,
    judge_scale: int = 5,
    disable_persona_persist: bool = False,
    disable_corrections: bool = False,
) -> Dict[str, Any]:
    """Replay turns for one agent role and collect responses."""

    # Build the agent
    memory = create_memory(memory_backend) if memory_backend else None

    checker = None
    if agent_name == "persona_loop" and not skip_nli and checker_model:
        checker = create_checker(enabled=True, checker_type="deberta", model_name=checker_model)

    if agent_name == "persona_loop":
        agent = PersonaLoopAgent(
            llm=llm,
            memory=memory,
            checker=checker,
            loop_interval=loop_interval,
            retrieval_top_k=retrieval_top_k,
            recent_turns=recent_turns,
            nli_threshold=nli_threshold,
            disable_persona_persist=disable_persona_persist,
            disable_corrections=disable_corrections,
        )
    else:
        agent = ContinuousAgent(llm=llm, max_history=max_history_window)

    persona_summary = agent_data.persona_summary
    agent_speaker = agent_data.name

    responses: List[str] = []
    turn_records: List[Dict[str, Any]] = []
    last_partner_text: str = ""

    loop_resets_total = 0
    loop_corrections_total = 0
    loop_retrieved_total = 0

    agent_turns = [t for t in turns if t.speaker == agent_speaker]
    total_agent_turns = len(agent_turns)
    agent_turn_idx = 0

    for turn in turns:
        if turn.speaker == agent_speaker:
            result = agent.run_roleplay_turn(
                speaker_name=agent_speaker,
                partner_name=partner_name,
                partner_text=last_partner_text,
                persona_summary=persona_summary,
            )
            response = result["response"]
            responses.append(response)
            agent_turn_idx += 1

            loop_resets_total += int(result.get("loop_reset", False))
            loop_corrections_total += int(result.get("loop_corrections_count", 0))
            loop_retrieved_total += int(result.get("loop_retrieved_count", 0))

            reset_tag = " [LOOP RESET]" if result.get("loop_reset") else ""
            print(f"    [{agent_turn_idx}/{total_agent_turns}] {agent_speaker} turn {turn.dia_id}{reset_tag}: {response[:60].replace(chr(10), ' ')!r}", flush=True)

            turn_records.append({
                "dia_id": turn.dia_id,
                "session": turn.session,
                "speaker": agent_speaker,
                "gold_text": turn.text,
                "response": response,
                "loop_reset": result.get("loop_reset", False),
            })
        else:
            # Partner's turn — record text for next agent turn
            last_partner_text = turn.text

    # Compute PCS (NLI)
    pcs_metrics: Dict[str, Any] = {}
    if not skip_nli and nli is not None:
        pcs_metrics = compute_pcs(
            responses=responses,
            persona_summary=persona_summary,
            nli=nli,
        )

    # Compute Judge PCS (Method B)
    judge_metrics: Dict[str, Any] = {}
    if judge_llm is not None:
        print(f"    Scoring {len(responses)} responses with LLM judge ...", flush=True)
        judge_out = compute_pcs_judge(
            responses=responses,
            persona_summary=persona_summary,
            llm=judge_llm,
            scale=judge_scale,
        )
        judge_metrics = {
            "judge_pcs": judge_out["judge_pcs"],
            "judge_pcs_avg": judge_out["judge_pcs_avg"],
        }
        # Attach per-turn scores back to turn_records
        for tr, jscore in zip(turn_records, judge_out["judge_scores"]):
            tr["judge_score"] = jscore

    return {
        "agent_name": agent_speaker,
        "agent_type": agent_name,
        "n_responses": len(responses),
        "loop_resets": loop_resets_total,
        "loop_corrections": loop_corrections_total,
        "loop_retrieved": loop_retrieved_total,
        **pcs_metrics,
        **judge_metrics,
        "turn_records": turn_records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Persona consistency eval on multimodal_dialog data.")
    parser.add_argument("--data", default="data/multimodal_dialog", help="Path to multimodal_dialog directory.")
    parser.add_argument("--agent", choices=["continuous", "persona_loop"], default="persona_loop")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--skip-nli", action="store_true", help="Skip NLI evaluation (fast mode).")
    # --- Method B: LLM-as-Judge ---
    parser.add_argument("--judge", action="store_true",
                        help="Enable LLM-as-judge persona consistency scoring (Method B).")
    parser.add_argument("--judge-model", default=None,
                        help="LLM model for judge. Defaults to --llm-model.")
    parser.add_argument("--judge-base-url", default=None, nargs='?', const='',
                        help="Base URL for judge LLM. Use without value to use real OpenAI. Defaults to --llm-base-url.")
    parser.add_argument("--judge-provider", default=None,
                        help="LLM provider for judge (e.g. openai). Defaults to --llm-provider.")
    parser.add_argument("--judge-scale", type=int, default=5,
                        help="Rating scale for judge (default: 5).")
    parser.add_argument("--memory-backend", choices=["bm25", "embedding"], default="embedding",
                        help="Memory backend for persona_loop (ignored for continuous).")
    parser.add_argument("--loop-interval", type=int, default=8, help="Loop reset interval K.")
    parser.add_argument("--loop-retrieval-topk", type=int, default=3)
    parser.add_argument("--loop-recent-turns", type=int, default=3)
    parser.add_argument("--loop-nli-threshold", type=float, default=0.3)
    parser.add_argument("--max-history-window", type=int, default=20,
                        help="Max history lines kept in context for continuous agent (0=unlimited).")
    parser.add_argument("--pairs", default="", help="Comma-separated pair names to run (e.g. pair1,pair2). Empty=all.")
    parser.add_argument("--loop-ablation", default="",
                        help="Comma-separated ablation tokens: disable_persona_persist,disable_corrections")
    parser.add_argument("--output", default="artifacts/multimodal_eval", help="Output directory.")
    args = parser.parse_args()

    ablation_tokens = {x.strip().lower() for x in args.loop_ablation.split(",") if x.strip()}
    allowed_ablation = {"disable_persona_persist", "disable_corrections"}
    invalid = sorted(ablation_tokens - allowed_ablation)
    if invalid:
        raise ValueError(f"Invalid --loop-ablation values: {', '.join(invalid)}. Allowed: {', '.join(sorted(allowed_ablation))}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build LLM
    llm_kwargs: Dict[str, Any] = {}
    if args.llm_base_url:
        llm_kwargs["base_url"] = args.llm_base_url
    llm = create_llm(provider=args.llm_provider, model_name=args.llm_model, **llm_kwargs)

    # Build judge LLM (Method B) -- may be same as main LLM
    judge_llm: Optional[Any] = None
    if args.judge:
        judge_model = args.judge_model or args.llm_model
        # Use is-not-None so that --judge-base-url "" explicitly means "real OpenAI"
        judge_base_url = args.judge_base_url if args.judge_base_url is not None else args.llm_base_url
        judge_provider = args.judge_provider or args.llm_provider
        judge_kwargs: Dict[str, Any] = {}
        if judge_base_url:
            judge_kwargs["base_url"] = judge_base_url
        if judge_model == args.llm_model and judge_base_url == args.llm_base_url:
            judge_llm = llm  # reuse same instance
        else:
            judge_llm = create_llm(provider=judge_provider, model_name=judge_model, **judge_kwargs)
        print(f"Judge LLM: {args.llm_provider}:{judge_model} (scale={args.judge_scale})", flush=True)

    # Build NLI scorer once (shared across all runs)
    nli: Optional[NLIScorer] = None
    if not args.skip_nli:
        print(f"Loading NLI model: {args.nli_model} ...")
        nli = NLIScorer(model_name=args.nli_model)

    # Load data
    pair_filter = {p.strip() for p in args.pairs.split(",") if p.strip()} if args.pairs else set()
    samples = load_all_pairs(args.data, exclude=["example"])
    if pair_filter:
        samples = [s for s in samples if s.pair_id in pair_filter]

    if not samples:
        print("No samples found. Check --data path and --pairs filter.")
        sys.exit(1)

    print(f"Loaded {len(samples)} pair(s). Agent={args.agent}, LLM={args.llm_provider}:{args.llm_model}", flush=True)

    # Memory backend alias
    memory_backend: Optional[str] = None
    if args.agent == "persona_loop":
        memory_backend = "embedding" if args.memory_backend == "embedding" else "chroma"

    all_results: List[Dict[str, Any]] = []
    start_time = time.time()

    for sample in samples:
        print(f"\n--- Pair: {sample.pair_id} ({len(sample.turns)} turns across {sample.session_count} sessions) ---", flush=True)

        # Evaluate both agent roles
        for role, agent_data, partner_data in [
            ("agent_a", sample.agent_a, sample.agent_b),
            ("agent_b", sample.agent_b, sample.agent_a),
        ]:
            print(f"  Running as {agent_data.name} (partner={partner_data.name}) ...", flush=True)
            result = run_agent_on_sample(
                agent_data=agent_data,
                partner_name=partner_data.name,
                turns=sample.turns,
                agent_name=args.agent,
                llm=llm,
                memory_backend=memory_backend,
                checker_model=args.nli_model if not args.skip_nli else None,
                loop_interval=args.loop_interval,
                retrieval_top_k=args.loop_retrieval_topk,
                recent_turns=args.loop_recent_turns,
                nli_threshold=args.loop_nli_threshold,
                max_history_window=args.max_history_window,
                skip_nli=args.skip_nli,
                nli=nli,
                judge_llm=judge_llm,
                judge_scale=args.judge_scale,
                disable_persona_persist="disable_persona_persist" in ablation_tokens,
                disable_corrections="disable_corrections" in ablation_tokens,
            )
            result["pair_id"] = sample.pair_id
            result["role"] = role
            all_results.append(result)

            pcs_str = f"  PCS={result.get('persona_pcs', 'N/A')}" if not args.skip_nli else ""
            judge_str = f"  judge_pcs={result.get('judge_pcs')}(avg={result.get('judge_pcs_avg')})" if args.judge else ""
            loop_str = f"  resets={result['loop_resets']} corrections={result['loop_corrections']}" if args.agent == "persona_loop" else ""
            print(f"    Done: responses={result['n_responses']}{pcs_str}{judge_str}{loop_str}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed:.1f}s.", flush=True)

    # Aggregate summary
    pcs_values = [r["persona_pcs"] for r in all_results if "persona_pcs" in r]
    ent_values = [r["persona_entailment"] for r in all_results if "persona_entailment" in r]
    con_values = [r["persona_contradiction"] for r in all_results if "persona_contradiction" in r]
    any_con_values = [r["persona_any_contra_ratio"] for r in all_results if "persona_any_contra_ratio" in r]

    summary: Dict[str, Any] = {
        "agent": args.agent,
        "llm": f"{args.llm_provider}:{args.llm_model}",
        "n_samples": len(all_results),
        "total_turns": sum(r["n_responses"] for r in all_results),
        "loop_interval": args.loop_interval if args.agent == "persona_loop" else None,
        "memory_backend": args.memory_backend if args.agent == "persona_loop" else None,
    }
    if not args.skip_nli and pcs_values:
        summary["persona_pcs"] = round(mean(pcs_values), 4)
        summary["persona_entailment"] = round(mean(ent_values), 4)
        summary["persona_contradiction"] = round(mean(con_values), 4)
        summary["persona_any_contra_ratio"] = round(mean(any_con_values), 4)
    # Method B judge aggregation
    judge_values = [r["judge_pcs"] for r in all_results if "judge_pcs" in r]
    judge_avg_values = [r["judge_pcs_avg"] for r in all_results if "judge_pcs_avg" in r]
    per_role_judge = {
        r["agent_name"]: {"judge_pcs": r.get("judge_pcs"), "judge_pcs_avg": r.get("judge_pcs_avg")}
        for r in all_results if "judge_pcs" in r
    }
    if judge_values:
        summary["judge_pcs"] = round(mean(judge_values), 4)
        summary["judge_pcs_avg"] = round(mean(judge_avg_values), 4)
        summary["judge_pcs_per_role"] = per_role_judge
    if args.agent == "persona_loop":
        summary["loop_resets_total"] = sum(r["loop_resets"] for r in all_results)
        summary["loop_corrections_total"] = sum(r["loop_corrections"] for r in all_results)

    print("\nSummary:", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    # Save outputs
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "per_role_results.json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "turn_records"} for r in all_results],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Save turn-level records separately (can be large)
    turn_rows: List[Dict[str, Any]] = []
    for r in all_results:
        for tr in r.get("turn_records", []):
            turn_rows.append({
                "pair_id": r["pair_id"],
                "role": r["role"],
                "agent_name": r["agent_name"],
                "agent_type": r["agent_type"],
                **tr,
            })
    (output_dir / "turn_records.json").write_text(
        json.dumps(turn_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save run manifest
    manifest = {
        "agent": args.agent,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "nli_model": args.nli_model,
        "skip_nli": args.skip_nli,
        "memory_backend": args.memory_backend,
        "loop_interval": args.loop_interval,
        "loop_retrieval_topk": args.loop_retrieval_topk,
        "loop_recent_turns": args.loop_recent_turns,
        "loop_nli_threshold": args.loop_nli_threshold,
        "max_history_window": args.max_history_window,
        "loop_ablation": args.loop_ablation,
        "data": str(args.data),
        "pairs": args.pairs,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

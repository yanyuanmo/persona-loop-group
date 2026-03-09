from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import time
from typing import Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_loop.core.factories import create_agent
from persona_loop.core.factories import create_llm
from persona_loop.core.factories import create_memory
from persona_loop.eval.nli_scorer import NLIScorer
from persona_loop.eval.qa_metrics import qa_scores


def load_locomo(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_conversation(conv: Dict[str, object]) -> List[Dict[str, str]]:
    flat: List[Dict[str, str]] = []
    for idx in range(1, 100):
        key = f"session_{idx}"
        if key not in conv:
            break
        turns = conv[key]
        for turn in turns:
            dia_id = str(turn.get("dia_id", ""))
            text = str(turn.get("text", "")).strip()
            if dia_id and text:
                flat.append({"dia_id": dia_id, "text": text})
    return flat


def build_history(turns: List[Dict[str, str]], max_turns: int) -> List[str]:
    items = [f"{t['dia_id']} {t['text']}" for t in turns]
    if max_turns <= 0:
        return items
    return items[-max_turns:]


def answer_question(agent, question: str, history: List[str]) -> str:
    context = "\n".join([f"[HISTORY] {x}" for x in history])
    result = agent.run_turn(prompt=question, context=context)
    return str(result["response"])


def print_progress_bar(current: int, total: int, start_time: float, width: int = 30) -> None:
    if total <= 0:
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else 0.0
    msg = f"\r[{bar}] {current}/{total} ({ratio * 100:5.1f}%) | ETA {remaining:6.1f}s"
    print(msg, end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo QA + NLI consistency evaluation.")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--agent", default="continuous")
    parser.add_argument("--llm-provider", default="hf")
    parser.add_argument("--llm-model", default="mistral-7b-instruct")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--output", default="artifacts/locomo_eval")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = create_llm(provider=args.llm_provider, model_name=args.llm_model)
    memory = create_memory(memory_type="chroma" if args.agent == "rag" else None)
    agent = create_agent(name=args.agent, llm=llm, memory=memory, checker=None)
    nli = None if args.skip_nli else NLIScorer(model_name=args.nli_model)

    rows: List[Dict[str, object]] = []
    samples = load_locomo(data_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    total_qa = sum(len(sample.get("qa", [])) for sample in samples)
    if args.max_qa > 0:
        total_qa = min(total_qa, args.max_qa)

    print(
        f"Starting LoCoMo eval: samples={len(samples)}, target_qa={total_qa}, "
        f"agent={args.agent}, llm={args.llm_provider}:{args.llm_model}, skip_nli={args.skip_nli}"
    )
    start_time = time.time()
    processed = 0
    stopped_early = False

    for sample in samples:
        conv = sample.get("conversation", {})
        turns = flatten_conversation(conv)
        dia2idx = {t["dia_id"]: i for i, t in enumerate(turns)}

        for qa in sample.get("qa", []):
            if args.max_qa > 0 and processed >= args.max_qa:
                stopped_early = True
                break

            question = str(qa.get("question", "")).strip()
            if not question:
                continue

            evidence = [str(x) for x in qa.get("evidence", [])]
            if evidence:
                max_idx = max(dia2idx.get(e, -1) for e in evidence)
                visible_turns = turns[: max_idx + 1] if max_idx >= 0 else turns
            else:
                visible_turns = turns

            history = build_history(visible_turns, max_turns=args.max_turns)
            prediction = answer_question(agent=agent, question=question, history=history)

            gold = qa.get("answer")
            adv = qa.get("adversarial_answer")
            row: Dict[str, object] = {
                "sample_id": sample.get("sample_id", ""),
                "category": int(qa.get("category", -1)),
                "question": question,
                "prediction": prediction,
                "gold_answer": gold,
                "adversarial_answer": adv,
            }

            if isinstance(gold, str) and gold.strip():
                row.update(qa_scores(prediction, gold))
                if nli is not None:
                    nli_gold = nli.score(premise=prediction, hypothesis=gold)
                    row["nli_entailment_gold"] = nli_gold["entailment"]
                    row["nli_contradiction_gold"] = nli_gold["contradiction"]

            if isinstance(adv, str) and adv.strip():
                if nli is not None:
                    nli_adv = nli.score(premise=prediction, hypothesis=adv)
                    row["nli_entailment_adv"] = nli_adv["entailment"]
                    row["nli_contradiction_adv"] = nli_adv["contradiction"]

            rows.append(row)
            processed += 1

            if not args.no_progress_bar:
                print_progress_bar(processed, total_qa, start_time)

            if args.progress_every > 0 and (
                processed % args.progress_every == 0 or processed == total_qa
            ):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"Progress: {processed}/{total_qa} QA, elapsed={elapsed:.1f}s, rate={rate:.2f} qa/s"
                )

        if stopped_early:
            break

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r]
        return round(mean(vals), 4) if vals else 0.0

    metrics = {
        "count": len(rows),
        "em": avg("em"),
        "f1": avg("f1"),
        "nli_entailment_gold": avg("nli_entailment_gold"),
        "nli_contradiction_gold": avg("nli_contradiction_gold"),
        "nli_entailment_adv": avg("nli_entailment_adv"),
        "nli_contradiction_adv": avg("nli_contradiction_adv"),
    }

    (output_dir / "qa_predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "qa_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not args.no_progress_bar:
        print()

    elapsed = time.time() - start_time
    print(f"Finished LoCoMo eval in {elapsed:.1f}s. Output: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

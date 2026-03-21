"""
Analyze qa_predictions.json to diagnose the source of persona_pcs < 0.
Prints:
  - field names in predictions
  - top-N worst pcs samples with injected facts + answer
  - inter-fact contradiction count in persona_cache
"""
import json
import argparse
import re
from pathlib import Path
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cache", default=None, help="persona_cache json (optional, for inter-fact conflict check)")
    ap.add_argument("--n", type=int, default=10, help="show N worst pcs samples")
    args = ap.parse_args()

    preds = json.load(open(args.pred, encoding="utf-8"))
    print(f"Total predictions: {len(preds)}")
    print(f"Keys: {sorted(preds[0].keys())}\n")

    # pcs per sample
    scored = []
    for p in preds:
        ent = p.get("persona_entailment", 0) or 0
        con = p.get("persona_contradiction", 0) or 0
        neu = 1.0 - ent - con
        pcs = ent - con
        facts_injected = p.get("persona_fact_texts", []) or []
        if isinstance(facts_injected, int):
            facts_injected = []
        conflict_ratio = p.get("persona_conflict_ratio", 0) or 0
        owner_match = p.get("persona_owner_match_ratio", None)
        scored.append((pcs, ent, con, neu, p.get("question",""), p.get("prediction",""), facts_injected, p.get("gold_answer",""), p.get("sample_id",""), conflict_ratio, owner_match))

    scored.sort(key=lambda x: x[0])

    print(f"=== Bottom {args.n} PCS samples ===")
    for row in scored[:args.n]:
        pcs, ent, con, neu, q, pred, facts, gt, sid, conflict_ratio, owner_match = row
        print(f"[{sid}] pcs={pcs:.3f} (ent={ent:.3f} con={con:.3f} neu={neu:.3f}) conflict={conflict_ratio:.2f} owner_match={owner_match}")
        print(f"  Q: {q}")
        print(f"  GT: {gt}")
        print(f"  Pred: {str(pred)[:200] if pred else '(none)'}")
        if facts:
            print(f"  Injected facts ({len(facts)}):")
            for f in facts:
                print(f"    - {f}")
        print()

    # Distribution
    pcs_vals = [s[0] for s in scored]
    neg = sum(1 for v in pcs_vals if v < 0)
    zero = sum(1 for v in pcs_vals if v == 0)
    pos = sum(1 for v in pcs_vals if v > 0)
    high_con = sum(1 for s in scored if s[2] > 0)  # contradiction > 0
    high_conflict = sum(1 for s in scored if s[9] > 0)  # conflict_ratio > 0
    low_owner = sum(1 for s in scored if s[10] is not None and s[10] < 0.5)
    print(f"PCS distribution: neg={neg} zero={zero} pos={pos}")
    print(f"  mean={sum(pcs_vals)/len(pcs_vals):.4f}  min={min(pcs_vals):.4f}  max={max(pcs_vals):.4f}")
    print(f"  has_contradiction={high_con}/{len(scored)}")
    print(f"  has_fact_conflict={high_conflict}/{len(scored)}")
    print(f"  low_owner_match(<0.5)={low_owner}/{len(scored)}")

    # Check if contradiction is mainly in facts-vs-answer or facts-are-empty
    no_facts = sum(1 for s in scored if not s[6])
    print(f"\nSamples with no injected facts recorded: {no_facts}/{len(scored)}")
    
    # Owner match distribution
    owner_vals = [s[10] for s in scored if s[10] is not None]
    if owner_vals:
        print(f"\nOwner match ratio: mean={sum(owner_vals)/len(owner_vals):.3f}  min={min(owner_vals):.3f}  max={max(owner_vals):.3f}")
    
    # Conflict ratio distribution
    conflict_vals = [s[9] for s in scored]
    print(f"Conflict ratio:    mean={sum(conflict_vals)/len(conflict_vals):.3f}  max={max(conflict_vals):.3f}")

    # cache inter-fact conflict check
    if args.cache:
        cache = json.load(open(args.cache, encoding="utf-8"))
        print(f"\n=== Persona cache: {len(cache)} conversations ===")
        for conv_id, facts_list in list(cache.items())[:5]:
            print(f"\n  conv={conv_id} facts={len(facts_list)}")
            for i, f in enumerate(facts_list):
                # show text if dict or str
                txt = f.get("text", f) if isinstance(f, dict) else str(f)
                print(f"    [{i}] {txt[:120]}")

if __name__ == "__main__":
    main()

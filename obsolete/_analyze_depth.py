import json
from collections import Counter

data = json.load(open("data/locomo10.json"))
conv = next(s for s in data if s["sample_id"] == "conv-47")
conversation = conv["conversation"]  # has keys: speaker_a, speaker_b, session_1, session_2, ...

# flatten session_1 / session_2 / ... into a flat turn list
turns = []
for idx in range(1, 100):
    key = f"session_{idx}"
    if key not in conversation:
        break
    for t in conversation[key]:
        dia_id = str(t.get("dia_id", ""))
        text = str(t.get("text", "")).strip()
        if dia_id and text:
            turns.append({"dia_id": dia_id, "text": text})

dia2idx = {t["dia_id"]: i for i, t in enumerate(turns)}
qas = conv["qa"]

print(f"Total turns: {len(turns)}, Total QAs: {len(qas)}")

rows = []
for i, qa in enumerate(qas):
    evidence = [str(e) for e in qa.get("evidence", [])]
    qtype = qa.get("category", "?")
    if not evidence:
        rows.append((i, qtype, -1, -1, -1))
        continue
    ev_indices = [dia2idx.get(e, -1) for e in evidence if dia2idx.get(e, -1) >= 0]
    if not ev_indices:
        rows.append((i, qtype, -1, -1, -1))
        continue
    cutoff_idx = max(ev_indices)
    min_ev_idx = min(ev_indices)
    depth = cutoff_idx - min_ev_idx
    rows.append((i, qtype, min_ev_idx, cutoff_idx, depth))

valid = [r for r in rows if r[4] >= 0]
depths = [r[4] for r in valid]

print(f"\nEvidence depth distribution (depth = cutoff_idx - earliest_evidence_idx):")
print(f"  depth=0  -> evidence IS the last visible turn (trivially easy)")
print(f"  depth>=3 -> evidence is outside the 3-turn recency window (memory needed)")
print()
for d, cnt in sorted(Counter(depths).items()):
    bar = "#" * cnt
    tag = "<-- OUTSIDE 3-window" if d >= 3 else ""
    print(f"  depth={d:3d}: {cnt:3d} QAs  {bar} {tag}")

inside  = sum(1 for d in depths if d < 3)
outside = sum(1 for d in depths if d >= 3)
deep    = sum(1 for d in depths if d >= 20)
print(f"\nSummary:")
print(f"  IN 3-turn window  (depth < 3) : {inside}")
print(f"  OUTSIDE window    (depth >= 3): {outside}")
print(f"  Deeply buried     (depth >=20): {deep}")

print(f"\nOutside-window QAs by type:")
tc = Counter(r[1] for r in valid if r[4] >= 3)
for t, c in sorted(tc.items()):
    print(f"  type={t}: {c}")

print(f"\nAll deeply-buried QAs (depth>=10):")
for r in sorted(valid, key=lambda x: -x[4]):
    if r[4] < 10:
        break
    qa = qas[r[0]]
    print(f"  QA#{r[0]:3d} depth={r[4]:4d} cat={r[1]}  Q: {qa.get('question','')[:90]}")

# Emit hard-slice index files for locomo eval
import os, json as _json
os.makedirs("configs/benchmark/slices", exist_ok=True)

def write_slice(name, indices, label):
    path = f"configs/benchmark/slices/{name}.json"
    entries = [{"sample_id": "conv-47", "qa_index": i} for i in sorted(indices)]
    _json.dump(entries, open(path, "w"), indent=2)
    print(f"\nWrote {path}: {len(entries)} QAs -> indices {sorted(indices)}")

hard_indices = [r[0] for r in valid if r[4] >= 3]
deep_indices = [r[0] for r in valid if r[4] >= 20]
write_slice("hard_depth3_conv47", hard_indices, "depth>=3")
write_slice("hard_depth20_conv47", deep_indices, "depth>=20")


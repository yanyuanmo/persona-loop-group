"""
Depth-vs-F1 analysis: compare continuous vs persona_loop across evidence burial depth.

burial_depth = max_turn_idx - evidence_turn_pos
  0      : evidence at very last turn (trivially easy)
  1-5    : within recent window, very easy
  6-19   : mild depth
  20-49  : moderate depth
  50-99  : deep
  100+   : very deep (evidence from early sessions)
"""

import json
from collections import defaultdict
from statistics import mean

def load(path):
    return json.load(open(f"{path}/qa_predictions.json"))

def bucket_label(burial):
    if burial is None:
        return "no_ev"
    if burial == 0:
        return "b0_last"
    if burial <= 5:
        return "b1-5"
    if burial <= 19:
        return "b6-19"
    if burial <= 49:
        return "b20-49"
    if burial <= 99:
        return "b50-99"
    return "b100+"

cont_rows = load("artifacts/depth_cont_conv47")
pl_rows   = load("artifacts/depth_pl_conv47")

# compute max_pos from combined data
all_positions = [r["evidence_turn_pos"] for r in cont_rows + pl_rows
                 if r.get("evidence_turn_pos", -1) >= 0]
max_pos = max(all_positions)

# align by qa_index
cont_by_idx = {r["qa_index"]: r for r in cont_rows}
pl_by_idx   = {r["qa_index"]: r for r in pl_rows}
shared = sorted(set(cont_by_idx) & set(pl_by_idx))

cont_aligned = [cont_by_idx[i] for i in shared]
pl_aligned   = [pl_by_idx[i]   for i in shared]

# bucket both
cont_buckets = defaultdict(list)
pl_buckets   = defaultdict(list)

for c_row, p_row in zip(cont_aligned, pl_aligned):
    pos = c_row.get("evidence_turn_pos", -1)
    burial = (max_pos - pos) if pos >= 0 else None
    label = bucket_label(burial)
    cont_buckets[label].append(float(c_row.get("f1", 0.0)))
    pl_buckets[label].append(float(p_row.get("f1", 0.0)))

all_labels = ["b0_last", "b1-5", "b6-19", "b20-49", "b50-99", "b100+", "no_ev"]

print(f"\nConversation length: {max_pos + 1} turns,  QAs analyzed: {len(shared)}\n")
print(f"{'Bucket':<12}  {'N':>4}  {'burial':>12}  {'cont_F1':>8}  {'pl_F1':>8}  {'delta':>7}  {'winner':>8}")
print("-" * 70)
for label in all_labels:
    c = cont_buckets.get(label, [])
    p = pl_buckets.get(label, [])
    n = len(c)
    if n == 0:
        continue
    burial_desc = label.replace("b", "").replace("_last", "=0")
    cf = round(mean(c), 4)
    pf = round(mean(p), 4)
    delta = round(pf - cf, 4)
    winner = "PL+" if delta > 0.005 else ("CONT+" if delta < -0.005 else "tie")
    print(f"{label:<12}  {n:>4}  {burial_desc:>12}  {cf:>8.4f}  {pf:>8.4f}  {delta:>+7.4f}  {winner:>8}")

all_c = [float(r.get("f1", 0)) for r in cont_aligned]
all_p = [float(r.get("f1", 0)) for r in pl_aligned]
print(f"\n{'OVERALL':<12}  {len(shared):>4}  {'':>12}  {mean(all_c):>8.4f}  {mean(all_p):>8.4f}  {mean(all_p)-mean(all_c):>+7.4f}")

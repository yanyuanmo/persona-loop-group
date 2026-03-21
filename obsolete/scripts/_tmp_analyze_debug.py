import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/formal20_gpt4omini_continuous_hybrid/persona_facts_debug.json"
with open(path) as f:
    data = json.load(f)

total = len(data)
json_ok = sum(1 for d in data if d.get("persona_llm_json_parsed", False))
struct_ok = sum(1 for d in data if d.get("persona_llm_structured_success", False))
both_fail = sum(1 for d in data if not d.get("persona_llm_json_parsed") and not d.get("persona_llm_structured_success"))
fallback = sum(1 for d in data if d.get("persona_llm_fallback_used", False))
repair_ok = sum(1 for d in data if d.get("persona_llm_repair_success", False))
cache_hit = sum(1 for d in data if d.get("persona_cache_hit", False))

print(f"total={total}  cache_hit={cache_hit}")
print(f"structured_ok={struct_ok} ({100*struct_ok/total:.0f}%)  json_parsed_ok={json_ok} ({100*json_ok/total:.0f}%)  both_fail={both_fail} ({100*both_fail/total:.0f}%)")
print(f"fallback_used={fallback}  repair_ok={repair_ok}")

# show a few failure samples with non-zero raw
fails = [d for d in data if not d.get("persona_llm_json_parsed") and not d.get("persona_llm_structured_success") and d.get("persona_llm_raw_len", 0) > 0]
print(f"\nfails_with_raw={len(fails)}")
for d in fails[:3]:
    print(f"\n--- {d.get('sample_id')} q={d.get('question','')[:60]}")
    print(f"  raw_len={d.get('persona_llm_raw_len')}  facts_used={d.get('persona_facts_used')}  facts_total={d.get('persona_facts_total')}")
    print(f"  fact_texts={d.get('persona_fact_texts', [])}")

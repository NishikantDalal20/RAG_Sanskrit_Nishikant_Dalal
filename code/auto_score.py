import csv
import statistics as st

from config import load_config

path = load_config()["paths"]["eval_file"].with_name("eval_answers.csv")
rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))

ok, failures = 0, []
for r in rows:
    hit = r["gold"] in (r["answer"] or "")
    ok += hit
    if not hit:
        failures.append(r)

print(f"Evidence match: {ok}/{len(rows)} = {ok / len(rows):.0%}")
print(f"avg retrieval_s: {st.mean(float(r['retrieval_s']) for r in rows):.2f}")
print(f"avg generation_s: {st.mean(float(r['generation_s']) for r in rows):.1f}")
print(f"avg tokens_per_s: {st.mean(float(r['tokens_per_s']) for r in rows):.1f}")
print("\nFAILURES:")
for r in failures:
    print("-", r["question"], "\n  gold:", r["gold"], "\n  answer:", (r["answer"] or "")[:200].replace("\n", " | "))
"""Step 6: measure retrieval quality, speed and memory.

  python code/evaluate.py              retrieval metrics for dense / bm25 / hybrid
  python code/evaluate.py --with-llm   also run the LLM and save answers to a CSV
"""
import argparse
import csv
import json
import statistics
import time

import psutil

from config import load_config
from retriever import Retriever


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def evaluate_retrieval(retriever, questions, mode, k=3):
    hits1 = hitsk = 0
    rr, times = [], []
    for item in questions:
        start = time.perf_counter()
        results = retriever.search(item["question"], top_k=k, mode=mode)
        times.append(time.perf_counter() - start)
        rank = next(
            (i for i, c in enumerate(results, start=1) if item["gold"] in c["text"]), None
        )
        hits1 += rank == 1
        hitsk += rank is not None
        rr.append(1 / rank if rank else 0.0)
    n = len(questions)
    return {
        "mode": mode,
        "Hit@1": round(hits1 / n, 2),
        f"Hit@{k}": round(hitsk / n, 2),
        "MRR": round(sum(rr) / n, 2),
        "avg_ms": round(1000 * statistics.mean(times), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid"])
    args = parser.parse_args()

    cfg = load_config()
    questions = json.loads(cfg["paths"]["eval_file"].read_text(encoding="utf-8"))

    #cfg["retrieval"]["mode"] = "hybrid" if "dense" in args.modes else "bm25"  # loads what is needed
    cfg["retrieval"]["mode"] = "hybrid" if any(m in ("dense", "hybrid") for m in args.modes) else "bm25"
    retriever = Retriever(cfg)

    print(f"{'mode':8} {'Hit@1':>6} {'Hit@3':>6} {'MRR':>5} {'avg ms':>8}")
    for mode in args.modes:
        r = evaluate_retrieval(retriever, questions, mode)
        print(f"{r['mode']:8} {r['Hit@1']:>6} {r['Hit@3']:>6} {r['MRR']:>5} {r['avg_ms']:>8}")
    print(f"\nProcess memory after retrieval: {rss_mb():.0f} MB")

    if args.with_llm:
        from rag_pipeline import SanskritRAG

        rag = SanskritRAG(cfg, use_llm=True)
        rows = []
        for item in questions:
            res = rag.ask(item["question"])
            tps = res["completion_tokens"] / max(res["timings"]["generation_s"], 1e-6)
            rows.append(
                {
                    "question": item["question"],
                    "gold": item["gold"],
                    "answer": res["answer"],
                    "retrieval_s": res["timings"]["retrieval_s"],
                    "generation_s": res["timings"]["generation_s"],
                    "tokens_per_s": round(tps, 1),
                    "manual_score_1_to_5": "",  # you fill this in
                }
            )
            print(f"done: {item['question']}")
        out = cfg["paths"]["eval_file"].with_name("eval_answers.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {out}. Open it in Excel and fill the manual_score column.")
        print(f"Peak process memory: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
"""Command-line interface.

  python code/main.py index                 build the index (once)
  python code/main.py ask "your question"   one question
  python code/main.py chat                  interactive loop
  add --no-llm to see retrieval only (fast, good for debugging)
"""
import argparse

from config import load_config


def show(result: dict):
    print(f"\nQuery (Devanagari): {result['query_devanagari']}")
    print("\n--- Retrieved passages ---")
    for i, c in enumerate(result["chunks"], start=1):
        print(f"[{i}] ({c['chunk_id']}, score {c['score']}) {c['text']}")
    if result["answer"] is not None:
        print("\n--- Answer ---")
        print(result["answer"])
    t = result["timings"]
    print(
        f"\n(retrieval {t['retrieval_s']}s | generation {t['generation_s']}s | "
        f"total {t['total_s']}s | {result['completion_tokens']} tokens)"
    )


def main():
    parser = argparse.ArgumentParser(description="Sanskrit RAG")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="build the search index")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    chat = sub.add_parser("chat", help="interactive mode")
    for p in (ask, chat):
        p.add_argument("--no-llm", action="store_true", help="retrieval only")
    args = parser.parse_args()

    cfg = load_config()

    if args.command == "index":
        from indexer import build_index

        build_index(cfg)
        return

    from rag_pipeline import SanskritRAG

    rag = SanskritRAG(cfg, use_llm=not args.no_llm)
    if args.command == "ask":
        show(rag.ask(args.question))
    else:
        print("Type a question in Sanskrit or Roman letters. Type 'exit' to quit.")
        while True:
            q = input("\n? ").strip()
            if q.lower() in {"exit", "quit", "q"}:
                break
            if q:
                show(rag.ask(q))


if __name__ == "__main__":
    main()
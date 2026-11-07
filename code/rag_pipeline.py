import time

from config import load_config
from extractive import best_sentence
from preprocess import to_devanagari
from retriever import Retriever


class SanskritRAG:
    def __init__(self, cfg=None, use_llm=True):
        self.cfg = cfg or load_config()
        self.retriever = Retriever(self.cfg)
        self.generator = None
        if use_llm:
            from generator import Generator

            self.generator = Generator(self.cfg)

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        query = to_devanagari(question, self.cfg["query"]["default_scheme"])

        t1 = time.perf_counter()
        chunks = self.retriever.search(query)
        best = best_sentence(query, chunks, self.cfg["retrieval"]["ngram"])
        t2 = time.perf_counter()

        result = {
            "question": question,
            "query_devanagari": query,
            "chunks": chunks,
            "answer": f"Best sentence: {best}",
            "completion_tokens": 0,
        }
        if self.generator:
            gen = self.generator.generate(query, [{"text": best}])
            result["answer"] = f"Best sentence: {best}\n{gen['text']}"
            result["completion_tokens"] = gen["completion_tokens"]
        t3 = time.perf_counter()

        result["timings"] = {
            "transliteration_s": round(t1 - t0, 4),
            "retrieval_s": round(t2 - t1, 4),
            "generation_s": round(t3 - t2, 4),
            "total_s": round(t3 - t0, 4),
        }
        return result
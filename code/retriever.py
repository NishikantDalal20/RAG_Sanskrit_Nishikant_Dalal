"""Step 3b: find the chunks most relevant to a question.

Three modes (set in config.yaml):
  dense  - meaning-based search with embeddings + FAISS
  bm25   - keyword search on character n-grams
  hybrid - both, merged with Reciprocal Rank Fusion (RRF)
"""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

# Devanagari letters, but NOT the danda marks (U+0964, U+0965)
WORD = re.compile(r"[ऀ-ॣ०-ॿ]+")


def tokenize(text: str, n: int = 3) -> list[str]:
    """Each word plus its character n-grams, so 'शर्कराम्' still matches 'शर्करा'."""
    tokens = []
    for word in WORD.findall(text):
        tokens.append(word)
        if len(word) > n:
            tokens += [word[i : i + n] for i in range(len(word) - n + 1)]
    return tokens


class Retriever:
    def __init__(self, cfg):
        self.cfg = cfg["retrieval"]
        index_dir = cfg["paths"]["index_dir"]
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))

        self.bm25 = BM25Okapi([tokenize(c["text"], self.cfg["ngram"]) for c in self.chunks])

        self.index = self.model = None
        if self.cfg["mode"] != "bm25":
            self._load_dense(index_dir)

    def _load_dense(self, index_dir):
        import faiss  # imported here so bm25-only runs need no model
        from sentence_transformers import SentenceTransformer

        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.model = SentenceTransformer(self.cfg["embedding_model"], device="cpu")

    # --- the two rankers: each returns chunk positions, best first -------------
    def _dense_rank(self, query: str, pool: int) -> list[int]:
        q = self.model.encode(
            [self.cfg["query_prefix"] + query], normalize_embeddings=True
        ).astype("float32")
        _, ids = self.index.search(q, pool)
        return [int(i) for i in ids[0] if i >= 0]

    def _bm25_rank(self, query: str, pool: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query, self.cfg["ngram"]))
        order = np.argsort(-scores)[:pool]
        return [int(i) for i in order if scores[i] > 0]  # ignore chunks with no overlap

    # --- public ----------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None, mode: str | None = None) -> list[dict]:
        top_k = top_k or self.cfg["top_k"]
        mode = mode or self.cfg["mode"]
        pool = self.cfg["candidate_pool"]

        rankings = []  # (weight, ranked list)
        if mode in ("dense", "hybrid"):
            rankings.append((self.cfg["dense_weight"], self._dense_rank(query, pool)))
        if mode in ("bm25", "hybrid"):
            rankings.append((self.cfg["bm25_weight"], self._bm25_rank(query, pool)))

        scores: dict[int, float] = {}
        for weight, ranked in rankings:
            for rank, pos in enumerate(ranked):
                scores[pos] = scores.get(pos, 0.0) + weight / (self.cfg["rrf_k"] + rank + 1)

        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [{**self.chunks[pos], "score": round(scores[pos], 5)} for pos in best]
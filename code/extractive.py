from rank_bm25 import BM25Okapi

from preprocess import split_sentences
from retriever import tokenize


def best_sentence(query, chunks, n=3):
    """Pick the one sentence from the retrieved chunks that best matches the question."""
    sentences = list(dict.fromkeys(s for c in chunks for s in split_sentences(c["text"])))
    if not sentences:
        return ""
    bm25 = BM25Okapi([tokenize(s, n) for s in sentences])
    scores = bm25.get_scores(tokenize(query, n))
    return sentences[int(scores.argmax())]
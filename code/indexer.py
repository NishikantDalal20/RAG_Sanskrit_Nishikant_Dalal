"""Step 3a: turn the documents into a searchable index (run once)."""
import json

import faiss
from sentence_transformers import SentenceTransformer

from preprocess import build_chunks, chunks_to_dicts


def build_index(cfg):
    index_dir = cfg["paths"]["index_dir"]
    index_dir.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["processed_dir"].mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(cfg)
    print(f"Built {len(chunks)} chunks from {cfg['paths']['raw_dir']}")
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks_to_dicts(chunks), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    r = cfg["retrieval"]
    model = SentenceTransformer(r["embedding_model"], device="cpu")  # CPU only
    texts = [r["passage_prefix"] + c.text for c in chunks]
    vectors = model.encode(
        texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine (vectors are normalised)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    print(f"Saved index with {index.ntotal} vectors to {index_dir}")
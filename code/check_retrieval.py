from config import load_config
from indexer import build_index
from preprocess import to_devanagari
from retriever import Retriever

cfg = load_config()
if not (cfg["paths"]["index_dir"] / "faiss.index").exists():
    build_index(cfg)

retriever = Retriever(cfg)
for q in ["ghaNTAkarNaH kaH", "शंखनादः मुखं केन लिम्पति?", "bhojarAjaH kiM ghoShitavAn"]:
    print("\nQ:", q)
    for mode in ["dense", "bm25", "hybrid"]:
        top = retriever.search(to_devanagari(q), top_k=1, mode=mode)[0]
        print(f"  {mode:7} -> {top['chunk_id']}: {top['text'][:70]}...")
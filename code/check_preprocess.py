from config import load_config
from preprocess import build_chunks, to_devanagari

cfg = load_config()
chunks = build_chunks(cfg)
print(len(chunks), "chunks")
for c in chunks[:3]:
    print(c.chunk_id, "|", c.text, "\n")

for q in ["ghaNTAkarNaH kaH", "bhojarAjaH kiM ghoShitavAn", "bhāskaraḥ kaḥ", "भोजराजः"]:
    print(q, "->", to_devanagari(q))
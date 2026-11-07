"""Step 2: clean Sanskrit text, split into sentences, build overlapping chunks."""
import re
import unicodedata
from dataclasses import asdict, dataclass

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# --- cleaning ---------------------------------------------------------------
PAREN = re.compile(r"\([^)]*\)")  # "(English translation)" blocks
# a run of Latin text: starts with a letter, continues through English punctuation/spaces
LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’\"“”.,:;!?\-\s()/@_]*")
QUOTES = re.compile(r"[\"“”‘’()\[\]]")
INVISIBLE = re.compile(r"[​﻿]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
SENTENCE_END = re.compile(r"(?<=[।॥])\s*")  # split AFTER a danda or double danda


def clean_text(text: str) -> str:
    """Keep the Sanskrit, drop English translations and stray symbols."""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE.sub("", text)
    text = PAREN.sub(" ", text)
    text = LATIN_RUN.sub(" ", text)
    text = QUOTES.sub(" ", text)
    text = text.replace("//", "।")          # verse line break used in the source
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_END.split(text)
    # drop fragments with fewer than 3 Devanagari characters
    return [p.strip() for p in parts if len(DEVANAGARI.findall(p)) >= 3]


# --- chunking ---------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc: str
    text: str
    start_sentence: int


def make_chunks(doc_name: str, sentences: list[str], size: int, overlap: int) -> list[Chunk]:
    step = max(1, size - overlap)
    chunks, i = [], 0
    while i < len(sentences):
        window = sentences[i : i + size]
        chunks.append(Chunk(f"{doc_name}#{len(chunks)}", doc_name, " ".join(window), i))
        if i + size >= len(sentences):
            break
        i += step
    return chunks


def build_chunks(cfg) -> list[Chunk]:
    p = cfg["preprocess"]
    chunks = []
    for path in sorted(cfg["paths"]["raw_dir"].glob("*.txt")):
        text = clean_text(path.read_text(encoding="utf-8"))
        sentences = split_sentences(text)
        # save the cleaned version so you can inspect it
        (cfg["paths"]["processed_dir"] / path.name).write_text("\n".join(sentences), encoding="utf-8")
        chunks += make_chunks(path.stem, sentences, p["chunk_sentences"], p["overlap_sentences"])
    return chunks


def chunks_to_dicts(chunks):
    return [asdict(c) for c in chunks]


# --- query transliteration --------------------------------------------------
IAST_MARKS = re.compile(r"[āīūṛṝḷḹṃḥṅñṭḍṇśṣ]")


def to_devanagari(text: str, default_scheme: str = "itrans") -> str:
    """Return the query in Devanagari. Roman input is converted; Devanagari is kept."""
    if DEVANAGARI.search(text):
        return text
    scheme = "iast" if IAST_MARKS.search(text.lower()) else default_scheme
    return transliterate(text, getattr(sanscript, scheme.upper()), sanscript.DEVANAGARI)
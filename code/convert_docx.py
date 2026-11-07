"""Step 1: split Rag-docs.docx into one UTF-8 .txt file per story.

Run once:  python code/convert_docx.py
"""
from docx import Document

from config import load_config

# (output file name, text the story's first paragraph STARTS with, keep that paragraph?)
# Title paragraphs are dropped (False); the devotee story has no title, so its
# first paragraph is real story text (True).
STORIES = [
    ("01_murkhabhritya", "मूर्खभृत्यस्य", False),
    ("02_kalidasa_lakshmudra", "चतुरस्य कालीदासस्य", False),
    ("03_vriddha_chaturya", "वृद्धायाः चार्तुयम्", False),
    ("04_devabhakta", "एकः परमः देवभक्तः अस्ति", True),
    ("05_shitam_bahu_badhati", "शीतं बहु बाधति", False),
]


def main():
    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    doc = Document(raw_dir / "Rag-docs.docx")
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    stories = {}      # file name -> list of paragraphs
    current = None
    for para in paragraphs:
        for name, marker, keep in STORIES:
            if para.startswith(marker):
                current = name
                stories[current] = []
                if keep:
                    stories[current].append(para)
                break
        else:  # no marker matched: ordinary paragraph of the current story
            if current is not None:
                stories[current].append(para)

    for name, paras in stories.items():
        out = raw_dir / f"{name}.txt"
        out.write_text("\n".join(paras), encoding="utf-8")
        print(f"wrote {out.name}: {len(paras)} paragraphs")


if __name__ == "__main__":
    main()
# Sanskrit Document Retrieval-Augmented Generation (RAG) System

**Author:** Nishikant Dalal  
**Execution:** CPU-Only Local Pipeline (No GPU / No External APIs)

A fully local Retrieval-Augmented Generation (RAG) system capable of processing and answering queries based on Sanskrit documents, operating strictly on CPU-based inference. The system accepts queries in native Devanagari script as well as Romanized transliterations (ITRANS and IAST).

---

## Highlights

- **Local-only execution:** Runs 100% locally on CPU without GPUs, CUDA, or external paid APIs.
- **Transliteration support:** Seamlessly parses Devanagari, ITRANS, and IAST input with automatic script detection.
- **Hybrid retrieval:** Combines dense semantic vectors (`multilingual-e5-small` + FAISS) with lexical BM25 character 3-grams via Reciprocal Rank Fusion (RRF).
- **CPU-optimized generation:** 4-bit quantized GGUF inference (`Qwen2.5-1.5B-Instruct`) via `llama.cpp` (`n_gpu_layers=0`).
- **Extractive-augmented reranking:** Sentence-level extractive reranking doubles evidence accuracy and halves generation latency.
- **Dual user interfaces:** Includes a feature-complete Command-Line Interface (CLI) and an interactive Gradio web application.

---

## Project Overview

The pipeline ingests source Sanskrit stories from `data/raw/`, cleans inline English translations, performs danda-aware sentence splitting, constructs an overlapping chunk index, and retrieves relevant context using hybrid ranking. A quantized local language model then produces a grounded response supported by exact Sanskrit textual evidence.

---

## Architecture

```text
User Query (Devanagari / ITRANS / IAST)
                 |
                 v
      Transliteration & Normalization (NFC)
                 |
                 v
   Dense Retrieval (multilingual-e5-small)
      + BM25 Lexical Retrieval (char 3-grams)
                 |
                 v
       Reciprocal Rank Fusion (RRF)
                 |
                 v
       Best-Matching Evidence Sentence Selection
                 |
                 v
    Local GGUF Model via llama.cpp (CPU-only)
                 |
                 v
            Grounded Answer + Sanskrit Evidence
```

### Key Technical Choices

- **Danda-aware sentence segmentation:** Segments specifically on Sanskrit sentence delimiters (`।`, `॥`) instead of English periods.
- **Character n-gram tokenization:** Pairs dense embeddings with BM25 character 3-grams to handle Sanskrit's complex inflections, nominal compounds, and case endings (e.g., matching *शर्कराम्* with *शर्करा*).
- **Decoupled design:** The retriever and generator are completely modular and decoupled, coordinated exclusively through `code/rag_pipeline.py`.

---

## Quick Start

### 1) Clone the repository

```bash
git clone https://github.com/NishikantDalal20/RAG_Sanskrit_Nishikant_Dalal.git
cd RAG_Sanskrit_Nishikant_Dalal
```

### 2) Create and activate a virtual environment

#### Windows (PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

#### Windows (PowerShell)
```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install -r requirements.txt
```

#### Linux / macOS
```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install llama-cpp-python
pip install -r requirements.txt
```

> **Windows Configuration Tips:**
> - If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> - To ensure proper Devanagari Unicode display in PowerShell, run: `$env:PYTHONUTF8 = "1"`

### 4) Build the search index

```bash
python code/main.py index
```
*Processes files in `data/raw/`, generates overlapping chunks, and creates the FAISS vector index (downloads the embedding model on first run).*

### 5) Download the local language model

```bash
python code/download_model.py
```
*Downloads the 4-bit quantized `qwen2.5-1.5b-instruct-q4_k_m.gguf` (approx. 1.1 GB) into `models/`.*

---

## Usage

Run commands from the project root with your virtual environment activated.

### Ask a single question

```bash
# Devanagari query
python code/main.py ask "भोजराजः किं घोषितवान्?"

# Romanized transliteration (ITRANS)
python code/main.py ask "ghaNTAkarNaH kaH"

# Romanized transliteration with diacritics (IAST)
python code/main.py ask "bhojarājaḥ kim"
```

### Retrieval-only mode (fast debugging without loading LLM)

```bash
python code/main.py ask "ghaNTAkarNaH kaH" --no-llm
```

### Interactive chat mode

```bash
python code/main.py chat
```

### Optional Gradio web UI

```bash
python code/app_gradio.py
```
Open your browser and navigate to **http://127.0.0.1:7860**.

---

## Supported Input Styles

| Format | Example | Notes |
| --- | --- | --- |
| **Devanagari** | `घण्टाकर्णः कः?` | Native Sanskrit script |
| **ITRANS** | `ghaNTAkarNaH kaH` | Case-sensitive (`A`, `I`, `U` represent long vowels) |
| **IAST** | `bhojarājaḥ kim` | Automatically detected via diacritics |

---

## Configuration

All pipeline parameters are configured via `config.yaml`:
- Document chunk size and overlap sentences
- Retrieval mode (`dense`, `bm25`, or `hybrid`)
- Reciprocal Rank Fusion (RRF) constants and weights
- Model context window (`n_ctx`), max tokens, temperature, and CPU thread count

To rebuild the index after modifying settings:
```powershell
# PowerShell:
Remove-Item -Recurse -Force index
python code/main.py index

# Linux / macOS:
rm -rf index
python code/main.py index
```

---

## Repository Structure

```text
RAG_Sanskrit_Nishikant_Dalal/
├── README.md
├── requirements.txt
├── config.yaml
├── code/
│   ├── config.py             # Configuration loader
│   ├── convert_docx.py       # DOCX source splitter
│   ├── preprocess.py         # Sanskrit cleaning, sentence splitting & transliteration
│   ├── indexer.py            # FAISS vector indexing
│   ├── retriever.py          # Dense, BM25, and Hybrid RRF search
│   ├── extractive.py         # Best-matching sentence reranker
│   ├── download_model.py     # GGUF model downloader
│   ├── generator.py          # CPU LLM inference via llama.cpp
│   ├── rag_pipeline.py       # Decoupled RAG pipeline orchestrator
│   ├── main.py               # CLI entrypoint (index, ask, chat)
│   ├── app_gradio.py         # Interactive web interface
│   ├── evaluate.py           # Retrieval and generation evaluation script
│   └── auto_score.py         # Automated evaluation scoring utility
├── data/
│   ├── raw/                  # Source Sanskrit texts
│   ├── processed/            # Cleaned sentence-segmented texts
│   ├── eval_questions.json   # 19 benchmark test questions with gold answers
│   ├── eval_answers.csv      # Optimized pipeline evaluation results
│   └── eval_answers_before.csv # Baseline evaluation results (ablation)
├── index/                    # Generated FAISS index & chunk mappings (local)
├── models/                   # Downloaded GGUF weights (local)
└── report/
    └── Technical_Report.pdf  # Comprehensive technical report
```

---

## Evaluation and Benchmarks

Measured on an **Intel Core i3-1125G4 (8 GB RAM, CPU-only)** on the 19 benchmark questions in `data/eval_questions.json`.

### 1. Retrieval Performance

```bash
python code/evaluate.py
```

| Retrieval Mode | Hit@1 | Hit@3 | MRR | Avg Latency |
| --- | :---: | :---: | :---: | :---: |
| Dense only (`multilingual-e5-small`) | 0.53 | 0.84 | 0.66 | 93 ms |
| BM25 only (character 3-grams) | 0.74 | 0.89 | 0.79 | 1 ms |
| **Hybrid (RRF, dense weight 0.3)** | **0.74** | **1.00** | **0.85** | **98 ms** |

### 2. Generation & Architecture Ablation

```bash
python code/evaluate.py --with-llm
python code/auto_score.py
```

| Generation Strategy | Evidence Match | Avg Latency | Throughput |
| --- | :---: | :---: | :---: |
| Baseline (LLM reads 3 full chunks) | 32% (6/19) | 16.0s | 4.8 tokens/s |
| **Optimized (Best sentence + LLM)** | **63% (12/19)** | **7.1s** | **9.4 tokens/s** |

> **Evaluation Notes:**
> - **Resource Footprint:** Peak process memory is approximately **1.2 GB RAM**, operating comfortably within standard CPU hardware limits.
> - **Metric Definition:** The *Evidence match* is an automated programmatic score verifying whether the extracted answer contains the expected gold Sanskrit phrase.
> - Full latency breakdowns, ablation comparisons, and failure analysis are documented in [`report/Technical_Report.pdf`](report/Technical_Report.pdf).

---

## Limitations

- **Small model capacity:** The 1.5B quantized model can produce flawed English translations of classical Sanskrit. The extracted Sanskrit evidence sentence displayed above the answer serves as the verified ground truth.
- **Corpus scale:** The current dataset consists of 5 classical stories (120 sentences).
- **Linguistic processing:** The pipeline relies on character n-grams rather than formal Sandhi splitting or deep Sanskrit morphological analysis.

---

## Related Material

- **Technical Report:** [`report/Technical_Report.pdf`](report/Technical_Report.pdf) (Detailed 9-page report covering system flow, preprocessing, retrieval/generation methodology, resource metrics, and failure analysis).
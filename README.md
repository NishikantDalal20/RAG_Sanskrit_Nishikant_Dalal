# Sanskrit Document Retrieval-Augmented Generation (RAG) System

## Project Overview

A Retrieval-Augmented Generation system that answers questions about five short Sanskrit stories.
Questions can be typed in Devanagari or in Roman letters (ITRANS or IAST). Everything runs locally on
the CPU. No GPU and no online API is used.

Author: Nishikant Dalal

## How It Works

```
Indexing (once):
  data/raw/*.txt -> clean -> split at danda marks -> overlapping 3-sentence chunks
                 -> E5 embeddings -> FAISS index (+ chunk list for BM25)

Query:
  question -> transliterate to Devanagari -> hybrid retrieval (dense + BM25 n-grams, RRF)
           -> top 3 chunks -> best-matching sentence -> local LLM (llama.cpp, 4-bit) -> answer
```

- **Retriever:** `intfloat/multilingual-e5-small` + FAISS, combined with BM25 on character 3-grams using Reciprocal Rank Fusion.
- **Generator:** Qwen2.5-1.5B-Instruct (Q4_K_M GGUF, about 1.1 GB) through `llama-cpp-python`, CPU only.
- The retriever and generator never call each other; only `code/rag_pipeline.py` connects them.

## Installation and Setup

Run commands from the project root. Python 3.12 is recommended. Allow about 3 GB of free disk space and an
internet connection for the first-time model downloads. The embedding model and LLM are downloaded separately:
the embedding model on the first index build, and the LLM with the download script. The application explicitly
uses CPU inference (`device="cpu"` for embeddings and `n_gpu_layers=0` for the LLM), so no GPU or CUDA setup is required.

1. **Create and activate a virtual environment.**

   **Windows PowerShell**
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **Linux / macOS (Terminal)**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies.** Install CPU-compatible PyTorch and `llama-cpp-python` first, then install the
   project dependencies from `requirements.txt`.

   **Windows PowerShell**
   ```powershell
   python -m pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   pip install -r requirements.txt
   ```

   **Linux (Terminal)**
   ```bash
   python -m pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install llama-cpp-python
   pip install -r requirements.txt
   ```

   **macOS (Terminal)**
   ```bash
   python -m pip install --upgrade pip
   pip install torch
   pip install llama-cpp-python
   pip install -r requirements.txt
   ```

   If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and
   activate the environment again.
   If Devanagari output is not displayed correctly in PowerShell, set UTF-8 mode before running commands:
   ```powershell
   $env:PYTHONUTF8 = "1"
   ```

3. **Build the document index.** This processes the `.txt` files in `data/raw/` and downloads the embedding
   model on first use.
   ```bash
   python code/main.py index
   ```

4. **Download the local language model.** The configured GGUF model is saved under `models/`.
   ```bash
   python code/download_model.py
   ```

## Running the Project

Run commands from the project root with the virtual environment activated. Ask a question in Devanagari or Roman
transliteration (ITRANS by default; IAST diacritics are detected).

```bash
python code/main.py ask "भोजराजः किं घोषितवान्?"
python code/main.py ask "bhojarājaḥ kim"
```

Start interactive chat or run without the LLM to see retrieval results only:

```bash
python code/main.py chat
python code/main.py ask "ghaNTAkarNaH kaH" --no-llm
```

To launch the optional Gradio interface:

```bash
python code/app_gradio.py
```

Open http://127.0.0.1:7860 in your browser.

If you start from the original Word file, run `python code/convert_docx.py` first. The converted story files are
already included in `data/raw/`.

Each answer prints the Devanagari query, retrieved passages and their story IDs, the best-matching sentence, the
model's answer, and timings.

### Input formats

| Style | Example |
| --- | --- |
| Devanagari | `घण्टाकर्णः कः?` |
| ITRANS (default) | `ghaNTAkarNaH kaH` |
| IAST (detected by diacritics) | `bhojarājaḥ kim` |

ITRANS is case-sensitive: `A`, `I`, `U` are long vowels.

## Configuration

All settings are in `config.yaml`: chunk size and overlap, retrieval mode (`dense`, `bm25`, `hybrid`),
dense/BM25 weights, `top_k`, model file, context size, `max_tokens` and CPU threads.
After changing chunking or the embedding model, delete the `index/` folder and run `python code/main.py index` again.

## Evaluation and Results

Run the retrieval evaluation to compare dense, BM25, and hybrid search:

```bash
python code/evaluate.py
```

To run answer evaluation with the local LLM and save results to `data/eval_answers.csv`:

```bash
python code/evaluate.py --with-llm
```

After running `python code/evaluate.py --with-llm` to create `data/eval_answers.csv`, calculate the automatic
evidence-match score and timings with:

```bash
python code/auto_score.py
```

Test questions (19, with gold phrases) are in `data/eval_questions.json`.

### Results (Intel Core i3-1125G4, 8 GB RAM, no GPU)

All measurements reported below were obtained on this hardware using the included evaluation questions.

| Retrieval mode | Hit@1 | Hit@3 | MRR |
| --- | --- | --- | --- |
| Dense (E5 + FAISS) | 0.53 | 0.84 | 0.66 |
| BM25 (character n-grams) | 0.74 | 0.89 | 0.79 |
| Hybrid (RRF, dense weight 0.3) | 0.74 | 1.00 | 0.85 |

| Generation design | Evidence match | Seconds per answer | Tokens/s |
| --- | --- | --- | --- |
| LLM reads 3 chunks | 6/19 (32%) | 16.0 | 4.8 |
| Best sentence, then LLM | 12/19 (63%) | 7.1 | 9.4 |

Peak memory is about 1.2 GB. The evidence match is an automatic score (the answer contains the expected
phrase), not human grading. Full discussion is in the report (`report/`).

## Project Structure

```
RAG_Sanskrit_Nishikant_Dalal/
|
|-- README.md
|-- requirements.txt
|-- config.yaml
|-- code/
|   |-- config.py
|   |-- convert_docx.py
|   |-- preprocess.py
|   |-- indexer.py
|   |-- retriever.py
|   |-- extractive.py
|   |-- download_model.py
|   |-- generator.py
|   |-- rag_pipeline.py
|   |-- main.py
|   |-- app_gradio.py
|   |-- evaluate.py
|   \-- auto_score.py
|-- data/
|   |-- raw/                 Sanskrit source documents
|   |-- processed/           cleaned text (generated)
|   \-- eval_questions.json
|-- index/                   generated FAISS index and chunks
|-- models/                  downloaded GGUF model
\-- report/
    \-- Technical_Report.pdf
```

The `index/` and `models/` directories are created locally and excluded from Git.

## Limitations

- The 1.5B model is weak at Sanskrit: its English line is often wrong even when the Sanskrit sentence shown is right.
  The Sanskrit sentence printed first is the part to trust.
- Small corpus (5 stories) and a small test set (19 questions) written from the same text.
- Single-sentence selection sometimes drops the neighbouring sentence that names the subject or the cause.
- No sandhi splitter or morphological analysis.

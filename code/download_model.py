"""Step 4a: download the quantised LLM (about 1 GB) into models/.

Run once:  python code/download_model.py
"""
from huggingface_hub import hf_hub_download

from config import load_config


def main():
    cfg = load_config()
    g = cfg["generation"]
    path = hf_hub_download(
        repo_id=g["repo_id"], filename=g["filename"], local_dir=cfg["paths"]["model_dir"]
    )
    print("Model saved to", path)


if __name__ == "__main__":
    main()
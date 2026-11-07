"""Load config.yaml and turn the relative paths into absolute ones."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # the project folder


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = ROOT / value
    return cfg
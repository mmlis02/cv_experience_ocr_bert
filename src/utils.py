"""Shared helpers: global seed, project paths and device selection."""

import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42

# Project paths (all relative to the repository root)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "images"
OCR_DIR = DATA_DIR / "ocr"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = ROOT / "models"
BASELINES_DIR = MODELS_DIR / "baselines"
BERT_DIR = MODELS_DIR / "bert"

OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
METRICS_DIR = OUTPUT_DIR / "metrics"


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU and MPS) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> str:
    """Return "mps" if Apple Silicon GPU is available, otherwise "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"

"""Shared helpers: global seed, project paths and device selection."""

import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42

# Target cap: exp_year and predictions are clipped to [0, EXP_CAP]
EXP_CAP = 20

# Experience bands as gap-free intervals on the continuous (clipped) target:
# [0, 3), [3, 6), [6, 11), [11, EXP_CAP] -- the assignment's 0-2, 3-5, 6-10, 11-20
BAND_EDGES = [0, 3, 6, 11]
BAND_LABELS = ["0-2", "3-5", "6-10", "11-20"]

OCCUPATIONS = [
    "registered_nurses",
    "elementary_middle_school_teachers",
    "software_developers",
    "accountants_auditors",
    "construction_laborers",
]

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


def to_band(years) -> np.ndarray:
    """Map years of experience (scalar or array) to band labels; values are clipped to [0, EXP_CAP] first."""
    clipped = np.clip(np.asarray(years, dtype=float), 0, EXP_CAP)
    idx = np.searchsorted(BAND_EDGES, clipped, side="right") - 1
    return np.asarray(BAND_LABELS)[idx]


def get_device() -> str:
    """Return "mps" if Apple Silicon GPU is available, otherwise "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"

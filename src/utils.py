"""Shared helpers: global seed, project paths and device selection."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
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


VARIANTS = ("raw", "clean", "clean_dropped_lowconf")
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


def load_split_texts(split: str, variant: str = "clean") -> pd.DataFrame:
    """CVs of one split (train / val / test) with the text of one variant: id, text, exp_year, occupation."""
    if variant not in VARIANTS + ("reference_clean",):
        raise ValueError(f"unknown variant {variant!r}")
    ids = pd.read_csv(SPLITS_DIR / f"{split}.csv", dtype={"id": str})[["id"]]
    texts = pd.read_csv(PROCESSED_DIR / f"{variant}.csv", dtype={"id": str})
    return ids.merge(texts, on="id", how="left", validate="one_to_one")


def clip_years(y) -> np.ndarray:
    return np.clip(np.asarray(y, dtype=float), 0, EXP_CAP)


def save_predictions(model: str, variant: str, split: str, df: pd.DataFrame, y_pred) -> Path:
    """output/predictions/<split>/<model>__<variant>.csv with id, occupation, y_true, y_pred."""
    out_dir = PREDICTIONS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}__{variant}.csv"
    pd.DataFrame({"id": df["id"], "occupation": df["occupation"], "y_true": df["exp_year"],
                  "y_pred": clip_years(y_pred)}).to_csv(path, index=False)
    return path


def save_metrics(name: str, payload: dict, subdir: str = "baselines") -> Path:
    out_dir = METRICS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def to_band(years) -> np.ndarray:
    """Map years of experience (scalar or array) to band labels; values are clipped to [0, EXP_CAP] first."""
    clipped = np.clip(np.asarray(years, dtype=float), 0, EXP_CAP)
    idx = np.searchsorted(BAND_EDGES, clipped, side="right") - 1
    return np.asarray(BAND_LABELS)[idx]


def get_device() -> str:
    """Return "mps" if Apple Silicon GPU is available, otherwise "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"

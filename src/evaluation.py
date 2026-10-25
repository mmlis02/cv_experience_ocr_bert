"""Metrics (MAE, RMSE, +/-k accuracy, bands, R2) and paired bootstrap CIs."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score

from src import utils


def collect_val_metrics(subdirs=("baselines", "bert")) -> pd.DataFrame:
    """One row per (model, variant) from the metrics JSONs written by the training scripts."""
    rows = []
    for subdir in subdirs:
        for path in sorted((utils.METRICS_DIR / subdir).glob("*.json")):
            m = json.loads(path.read_text())
            if "val" not in m:
                continue
            rows.append({"model": m["model"], "variant": m["variant"], **{f"val_{k}": v for k, v in m["val"].items()},
                         "train_seconds": m.get("train_seconds"),
                         "inference_seconds_per_cv": m.get("inference_seconds_per_cv")})
    return pd.DataFrame(rows)


def regression_metrics(y_true, y_pred) -> dict:
    """All assignment metrics; both arrays are clipped to [0, EXP_CAP] first."""
    y_true, y_pred = utils.clip_years(y_true), utils.clip_years(y_pred)
    err = np.abs(y_pred - y_true)
    band_true, band_pred = utils.to_band(y_true), utils.to_band(y_pred)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((y_pred - y_true) ** 2).mean())),
        "acc_1": float((err <= 1).mean()),
        "acc_2": float((err <= 2).mean()),
        "acc_3": float((err <= 3).mean()),
        "band_accuracy": float(accuracy_score(band_true, band_pred)),
        "band_macro_f1": float(f1_score(band_true, band_pred, labels=utils.BAND_LABELS, average="macro",
                                        zero_division=0)),
        "r2": float(r2_score(y_true, y_pred)),
    }

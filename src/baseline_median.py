"""Baseline 0: median predictor.

Always predicts the median exp_year of the training set. Every other model must beat it.

Usage:
    python -m src.baseline_median [--variant clean]
"""

import argparse
import time

import joblib
import numpy as np

from src import utils
from src.evaluation import regression_metrics

NAME = "median"


def model_path(variant: str):
    return utils.BASELINES_DIR / f"{NAME}__{variant}.joblib"


def load(variant: str = "clean") -> dict:
    return joblib.load(model_path(variant))


def predict(texts, variant: str = "clean") -> np.ndarray:
    """The training median for every text (the text is ignored)."""
    return np.full(len(texts), load(variant)["median"], dtype=float)


def main(variant: str) -> dict:
    train = utils.load_split_texts("train", variant)
    val = utils.load_split_texts("val", variant)

    start = time.perf_counter()
    model = {"median": float(np.median(train["exp_year"]))}
    train_seconds = time.perf_counter() - start
    utils.BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path(variant))

    start = time.perf_counter()
    y_val = predict(val["text"], variant)
    infer = (time.perf_counter() - start) / len(val)

    utils.save_predictions(NAME, variant, "val", val, y_val)
    metrics = {"model": NAME, "variant": variant, "median": model["median"],
               "train_seconds": train_seconds, "inference_seconds_per_cv": infer,
               "val": regression_metrics(val["exp_year"], y_val)}
    utils.save_metrics(f"{NAME}__{variant}", metrics)
    print(f"[{NAME}/{variant}] train median {model['median']:.2f} · val MAE {metrics['val']['mae']:.3f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=utils.VARIANTS, default="clean")
    main(parser.parse_args().variant)

"""Baseline 1: TF-IDF + Ridge regression.

Grid on the validation set over:
- features: word 1-2 grams, or word 1-2 grams + character 2-5 grams (char_wb)
- keep_numbers: keep digits as tokens ("2019", "11") or replace every digit run with "0"
- alpha of the Ridge regression

The best configuration by validation MAE is refit on train and saved.

Usage:
    python -m src.baseline_tfidf [--variant clean]
"""

import argparse
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src import utils
from src.evaluation import regression_metrics

NAME = "tfidf"
ALPHAS = np.logspace(-2, 2, 9)
FEATURES = ("word", "word+char")


def strip_numbers(texts):
    """Replace every digit run with "0": dates and years no longer carry their value."""
    return [re.sub(r"\d+", "0", t) for t in texts]


def build_pipeline(features: str, keep_numbers: bool, alpha: float) -> Pipeline:
    # Default sklearn tokenisation drops 1-char tokens; this pattern keeps "0" and "11"
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    vectorizer = word if features == "word" else FeatureUnion([
        ("word", word),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    steps = [] if keep_numbers else [("strip_numbers", FunctionTransformer(strip_numbers))]
    return Pipeline(steps + [("tfidf", vectorizer), ("ridge", Ridge(alpha=alpha, random_state=utils.SEED))])


def model_path(variant: str):
    return utils.BASELINES_DIR / f"{NAME}__{variant}.joblib"


def load(variant: str = "clean") -> Pipeline:
    return joblib.load(model_path(variant))


def predict(texts, variant: str = "clean", model: Pipeline | None = None) -> np.ndarray:
    model = model or load(variant)
    return utils.clip_years(model.predict(list(texts)))


def main(variant: str) -> dict:
    train = utils.load_split_texts("train", variant)
    val = utils.load_split_texts("val", variant)

    rows, start = [], time.perf_counter()
    for features in FEATURES:
        for keep_numbers in (True, False):
            for alpha in ALPHAS:
                pipe = build_pipeline(features, keep_numbers, alpha).fit(train["text"], train["exp_year"])
                mae = regression_metrics(val["exp_year"], predict(val["text"], model=pipe))["mae"]
                rows.append({"features": features, "keep_numbers": keep_numbers, "alpha": alpha, "val_mae": mae})
    tuning_seconds = time.perf_counter() - start
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["val_mae"].idxmin()]

    start = time.perf_counter()
    model = build_pipeline(best["features"], bool(best["keep_numbers"]), best["alpha"]).fit(train["text"], train["exp_year"])
    train_seconds = time.perf_counter() - start
    utils.BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path(variant))

    start = time.perf_counter()
    y_val = predict(val["text"], model=model)
    infer = (time.perf_counter() - start) / len(val)

    utils.save_predictions(NAME, variant, "val", val, y_val)
    grid.round(4).to_csv(utils.TABLES_DIR / f"tfidf_grid__{variant}.csv", index=False)
    best_by_numbers = grid.loc[grid.groupby(["features", "keep_numbers"])["val_mae"].idxmin()]
    metrics = {"model": NAME, "variant": variant,
               "best": {"features": best["features"], "keep_numbers": bool(best["keep_numbers"]),
                        "alpha": float(best["alpha"])},
               "best_val_mae_by_setting": best_by_numbers.round(4).to_dict("records"),
               "n_features": int(len(model.named_steps["ridge"].coef_)),
               "tuning_seconds": tuning_seconds, "train_seconds": train_seconds,
               "inference_seconds_per_cv": infer, "val": regression_metrics(val["exp_year"], y_val)}
    utils.save_metrics(f"{NAME}__{variant}", metrics)
    print(best_by_numbers.round(4).to_string(index=False))
    print(f"[{NAME}/{variant}] best {metrics['best']} · val MAE {metrics['val']['mae']:.3f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=utils.VARIANTS, default="clean")
    main(parser.parse_args().variant)

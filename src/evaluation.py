"""Metrics (MAE, RMSE, +/-k accuracy, bands, R2) and paired bootstrap CIs.

The final evaluation: every model is scored on the same test set, which is used here for
the first time. Also answers the analytical questions:

- Q1 BERT's gain over each model (paired bootstrap), errors by band and occupation,
     the 10 largest BERT errors, and a probe (delete a job / shift all dates +5 years)
- Q2 each clean-trained model on the reference text of the same test CVs
- Q3 raw vs clean (vs low-confidence lines removed) text, with paired bootstrap
- Q4 BERT's and the rules' error by sex and by race

Test predictions are cached in output/predictions/; delete them to recompute.

Usage:
    python -m src.evaluation
"""

import importlib
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score

from src import utils

# Display order of the models in every table and figure
MODELS = ["median", "tfidf", "embeddings", "bert", "rules"]
LABELS = {"median": "Median", "tfidf": "TF-IDF + Ridge", "embeddings": "Embeddings + Ridge",
          "bert": "Fine-tuned BERT", "rules": "Date rules (bonus)"}
MODULES = {"median": "src.baseline_median", "tfidf": "src.baseline_tfidf", "embeddings": "src.baseline_embeddings",
           "bert": "src.train_bert", "rules": "src.baseline_rules"}
# Variants each model was trained on (BERT: clean and raw only)
TRAINED_VARIANTS = {m: list(utils.VARIANTS) for m in MODELS} | {"bert": ["clean", "raw"]}
N_BOOT = 2000
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


# ------------------------------------------------------------------ metrics

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


def train_seconds(model: str, variant: str) -> float:
    if model == "rules":
        return 0.0
    subdir = "bert" if model == "bert" else "baselines"
    m = json.loads((utils.METRICS_DIR / subdir / f"{model}__{variant}.json").read_text())
    return float(m["train_seconds"])


# ------------------------------------------------------------------ bootstrap

def paired_bootstrap(err_a, err_b, n_boot: int = N_BOOT, seed: int = utils.SEED) -> dict:
    """MAE(a) - MAE(b) on the same CVs, with a 95% percentile CI from paired resampling.

    Positive = b has the lower error. Significant when the CI excludes 0.
    """
    err_a, err_b = np.abs(np.asarray(err_a)), np.abs(np.asarray(err_b))
    idx = np.random.default_rng(seed).integers(0, len(err_a), (n_boot, len(err_a)))
    diffs = err_a[idx].mean(axis=1) - err_b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mae_a": err_a.mean(), "mae_b": err_b.mean(), "diff": err_a.mean() - err_b.mean(),
            "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)}


def bootstrap_mean(values, n_boot: int = N_BOOT, seed: int = utils.SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))
    return tuple(np.percentile(values[idx].mean(axis=1), [2.5, 97.5]))


# ------------------------------------------------------------------ predictions

def model_predict(model: str, texts, variant: str) -> np.ndarray:
    """predict(texts) of the model trained on `variant`."""
    module = importlib.import_module(MODULES[model])
    if model == "rules":
        return module.predict(texts)
    return module.predict(texts, variant=variant)


def test_predictions(model: str, trained_on: str, text_variant: str | None = None) -> pd.DataFrame:
    """Cached test predictions of a model trained on `trained_on`, applied to `text_variant` texts."""
    text_variant = text_variant or trained_on
    split = "test" if text_variant == trained_on else f"test_on_{text_variant}"
    path = utils.PREDICTIONS_DIR / split / f"{model}__{trained_on}.csv"
    timing_path = utils.PREDICTIONS_DIR / split / f"{model}__{trained_on}.timing.json"
    if path.exists():
        return pd.read_csv(path, dtype={"id": str})
    df = utils.load_split_texts("test", text_variant)
    start = time.perf_counter()
    y_pred = model_predict(model, df["text"].tolist(), trained_on)
    seconds = time.perf_counter() - start
    utils.save_predictions(model, trained_on, split, df, y_pred)
    timing_path.write_text(json.dumps({"inference_seconds_per_cv": seconds / len(df), "n": len(df)}))
    print(f"  predicted {model} ({trained_on}) on test/{text_variant}: {seconds:.1f}s", flush=True)
    return pd.read_csv(path, dtype={"id": str})


def inference_seconds(model: str, variant: str) -> float:
    path = utils.PREDICTIONS_DIR / "test" / f"{model}__{variant}.timing.json"
    return json.loads(path.read_text())["inference_seconds_per_cv"]


# ------------------------------------------------------------------ main comparison

def model_comparison(preds: dict) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        p = preds[model]
        m = regression_metrics(p["y_true"], p["y_pred"])
        rows.append({"model": LABELS[model], "mae_years": m["mae"], "rmse_years": m["rmse"], "acc_1": m["acc_1"],
                     "acc_2": m["acc_2"], "acc_3": m["acc_3"], "band_accuracy": m["band_accuracy"],
                     "band_macro_f1": m["band_macro_f1"], "r2": m["r2"],
                     "training_time_seconds": train_seconds(model, "clean"),
                     "inference_time_per_cv_seconds": inference_seconds(model, "clean")})
    table = pd.DataFrame(rows)
    times = ["training_time_seconds", "inference_time_per_cv_seconds"]
    return table.round(4).assign(**{c: table[c].round(6) for c in times})


def bert_gain(preds: dict) -> pd.DataFrame:
    """MAE reduction of BERT versus every other model (positive = BERT better)."""
    bert_err = preds["bert"]["y_pred"] - preds["bert"]["y_true"]
    rows = []
    for model in MODELS:
        if model == "bert":
            continue
        other_err = preds[model]["y_pred"] - preds[model]["y_true"]
        r = paired_bootstrap(other_err, bert_err)
        rows.append({"baseline": LABELS[model], "baseline_mae": r["mae_a"], "bert_mae": r["mae_b"],
                     "mae_reduction": r["diff"], "relative_reduction": r["diff"] / r["mae_a"],
                     "ci95_low": r["ci_low"], "ci95_high": r["ci_high"], "significant": r["significant"],
                     "conclusion": ("BERT significantly better" if r["ci_low"] > 0 else
                                    "BERT significantly worse" if r["ci_high"] < 0 else "no significant difference")})
    return pd.DataFrame(rows).round(4)


def errors_by(preds: dict, group: str) -> pd.DataFrame:
    """MAE and mean signed error of every model by true band or by occupation."""
    rows = []
    for model in MODELS:
        p = preds[model].assign(band=lambda d: utils.to_band(d["y_true"]), err=lambda d: d["y_pred"] - d["y_true"])
        for key, g in p.groupby(group):
            rows.append({"model": LABELS[model], group: key, "n": len(g), "mae": g["err"].abs().mean(),
                         "mean_error": g["err"].mean()})
    out = pd.DataFrame(rows)
    order = utils.BAND_LABELS if group == "band" else utils.OCCUPATIONS
    out[group] = pd.Categorical(out[group], order, ordered=True)
    out["model"] = pd.Categorical(out["model"], [LABELS[m] for m in MODELS], ordered=True)
    return out.sort_values([group, "model"]).round(4)


def compression(preds: dict) -> pd.DataFrame:
    """Regression to the middle: slope of predicted on true (1 = none) and bias at the extremes."""
    rows = []
    for model in MODELS:
        p = preds[model]
        slope, intercept = np.polyfit(p["y_true"], p["y_pred"], 1)
        band = utils.to_band(p["y_true"])
        err = p["y_pred"] - p["y_true"]
        rows.append({"model": LABELS[model], "slope_pred_on_true": slope, "intercept": intercept,
                     "mean_error_juniors_0_2": err[band == "0-2"].mean(),
                     "mean_error_seniors_11_20": err[band == "11-20"].mean(),
                     "compresses_to_middle": bool(slope < 0.9)})
    return pd.DataFrame(rows).round(4)


# ------------------------------------------------------------------ largest BERT errors

def _gaps_and_overlaps(text: str) -> tuple[float, float]:
    from src.baseline_rules import extract_ranges, merge_periods

    ranges = extract_ranges(text)
    raw_months = sum(e - s for s, e in ranges)
    merged = merge_periods(ranges)
    overlap = (raw_months - sum(e - s for s, e in merged)) / 12
    gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
    return overlap, max(gaps, default=0) / 12


def _years_after_last_job(text: str) -> float:
    """Years between the end of the last job and the data snapshot (the rules' reference date)."""
    from src.baseline_rules import REFERENCE_DATE, extract_ranges

    ranges = extract_ranges(text)
    snapshot = REFERENCE_DATE[0] * 12 + REFERENCE_DATE[1] - 1
    return (snapshot - max(e for _, e in ranges)) / 12 if ranges else 0.0


def error_by_career_pattern(preds: dict) -> pd.DataFrame:
    """Mean and absolute error of every model for CVs with / without gaps (between jobs, after the last job)."""
    texts = utils.load_split_texts("test", "clean").set_index("id")["text"]
    pattern = pd.DataFrame({"id": texts.index,
                            "gap_between_jobs": [_gaps_and_overlaps(t)[1] > 0.25 for t in texts],
                            "last_job_ended_before_snapshot": [_years_after_last_job(t) > 0 for t in texts]})
    rows = []
    for model in MODELS:
        p = preds[model].merge(pattern, on="id").assign(err=lambda d: d["y_pred"] - d["y_true"])
        for col in ("gap_between_jobs", "last_job_ended_before_snapshot"):
            for value, g in p.groupby(col):
                rows.append({"model": LABELS[model], "pattern": col, "present": bool(value), "n": len(g),
                             "mae": g["err"].abs().mean(), "mean_error": g["err"].mean()})
    return pd.DataFrame(rows).round(4)


def largest_bert_errors(preds: dict, k: int = 10) -> pd.DataFrame:
    """The k largest absolute BERT errors with the likely cause of each."""
    from src.baseline_rules import extract_ranges
    from transformers import AutoTokenizer

    clean = utils.load_split_texts("test", "clean").set_index("id")["text"]
    ref = utils.load_split_texts("test", "reference_clean").set_index("id")["text"]
    tokenizer = AutoTokenizer.from_pretrained(utils.BERT_DIR / "bert__clean")
    max_len = json.loads((utils.BERT_DIR / "bert__clean" / "train_config.json").read_text())["max_len"]
    p = preds["bert"].merge(preds["rules"][["id", "y_pred"]].rename(columns={"y_pred": "rules_pred"}), on="id")
    p["error"] = p["y_pred"] - p["y_true"]
    rows = []
    for _, r in p.reindex(p["error"].abs().sort_values(ascending=False).index).head(k).iterrows():
        text = clean[r["id"]]
        n_jobs = len(extract_ranges(text))
        overlap, gap = _gaps_and_overlaps(text)
        ocr_dates = sorted(extract_ranges(text)) != sorted(extract_ranges(ref[r["id"]]))
        # End date without a full month ("2023." or "2023.0"): cut off in the CV image
        truncated_date = bool(re.search(r"\d{4}\.\d{2}-\d{4}(?:\.\d?)?(?![\d.])", text))
        n_tokens = len(tokenizer(text)["input_ids"])
        direction = "over" if r["error"] > 0 else "under"
        if n_tokens > max_len:
            cause, why = "truncation", f"{n_tokens} tokens > max_len {max_len}: part of the CV never reached BERT."
        elif truncated_date:
            cause, why = "missing dates", "A job's end month is cut off in the CV image (e.g. '2023.'), so one date is incomplete."
        elif ocr_dates:
            cause, why = "OCR misread a date", "The date ranges in the OCR text differ from the reference text."
        elif overlap > 0.1:
            cause, why = "overlapping jobs", f"Jobs overlap by {overlap:.1f} years; adding the ranges would double-count them."
        elif gap > 0.25:
            cause, why = "career gap", (f"A {gap:.1f}-year gap between jobs; BERT {direction}estimates, as if it "
                                        "counted part of the span from first start to last end.")
        elif (after := _years_after_last_job(text)) > 1:
            ranges = sorted(extract_ranges(text))
            grad = min((int(y) for y in YEAR_RE.findall(text.split("EDUCATION")[-1])), default=None)
            cause = "career gap"
            why = (f"The last job ended {after:.1f} years before the data snapshot (2023.01)"
                   + (f" and graduation was in {grad}" if grad else "")
                   + f"; BERT {direction}estimates, as if part of the time since then counted as experience.")
        else:
            cause, why = "other", (f"Dates are read correctly ({n_jobs} job(s), rules get {r['rules_pred']:.2f}); "
                                   f"BERT {direction}estimates the arithmetic itself.")
        rows.append({"id": r["id"], "occupation": r["occupation"], "y_true": r["y_true"], "bert_pred": r["y_pred"],
                     "rules_pred": r["rules_pred"], "error": r["error"], "n_jobs": n_jobs, "cause": cause,
                     "explanation": why, "text": text})
    return pd.DataFrame(rows).round(3)


# ------------------------------------------------------------------ Q1 probe

def _job_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """(start, end) line spans of each job: its title + dates line and the bullet lines below it."""
    from src.baseline_rules import RANGE_RE

    starts = [i for i, l in enumerate(lines) if RANGE_RE.search(l)]
    blocks = []
    for s in starts:
        e = s + 1
        while e < len(lines) and lines[e].startswith("- ") and e not in starts:
            e += 1
        blocks.append((s, e))
    return blocks


def delete_one_job(text: str, rng) -> str:
    lines = text.split("\n")
    blocks = _job_blocks(lines)
    s, e = blocks[rng.integers(len(blocks))]
    return "\n".join(lines[:s] + lines[e:])


def shift_years(text: str, years: int = 5) -> str:
    """Add `years` to every year in the text (job dates and graduation)."""
    return YEAR_RE.sub(lambda m: str(int(m.group(1)) + years), text)


def probe(n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does each model move like a recruiter when a job is deleted or every date shifts by +5 years?

    The expected change is given by the date rules: minus the deleted job's duration, and 0
    for the shift (durations do not change; there are no "Present" end dates in this data).
    """
    from src.baseline_rules import extract_ranges

    test = utils.load_split_texts("test", "clean")
    test = test[test["text"].apply(lambda t: len(extract_ranges(t)) >= 2)].sample(n, random_state=utils.SEED)
    rng = np.random.default_rng(utils.SEED)
    versions = {"original": test["text"].tolist(),
                "delete_one_job": [delete_one_job(t, rng) for t in test["text"]],
                "shift_plus_5_years": [shift_years(t) for t in test["text"]]}
    out = test[["id", "exp_year"]].copy()
    for model in MODELS:
        for name, texts in versions.items():
            out[f"{model}__{name}"] = model_predict(model, texts, "clean")
    rows = []
    for exp in ("delete_one_job", "shift_plus_5_years"):
        expected = out[f"rules__{exp}"] - out["rules__original"]
        for model in MODELS:
            change = out[f"{model}__{exp}"] - out[f"{model}__original"]
            rows.append({"experiment": exp, "model": LABELS[model], "mean_expected_change": expected.mean(),
                         "mean_model_change": change.mean(),
                         "mean_abs_deviation_from_expected": (change - expected).abs().mean(),
                         "corr_with_expected": change.corr(expected) if expected.std() > 0 else np.nan,
                         "share_moving_right_direction": float(np.mean(np.sign(change) == np.sign(expected)))
                         if exp == "delete_one_job" else np.nan})
    return pd.DataFrame(rows).round(4), out.round(3)


# ------------------------------------------------------------------ Q2 / Q3 / Q4

def ocr_vs_reference(preds: dict) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        on_ref = test_predictions(model, "clean", "reference_clean")
        ocr = preds[model]
        r = paired_bootstrap(ocr["y_pred"] - ocr["y_true"], on_ref["y_pred"] - on_ref["y_true"])
        rows.append({"model": LABELS[model], "mae_ocr_text": r["mae_a"], "mae_reference_text": r["mae_b"],
                     "gap_ocr_minus_reference": r["diff"], "ci95_low": r["ci_low"], "ci95_high": r["ci_high"],
                     "significant": r["significant"]})
    return pd.DataFrame(rows).round(4)


def variant_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test MAE of every model trained and tested on each text variant; paired bootstrap vs clean."""
    rows, boots = [], []
    for model in MODELS:
        clean = test_predictions(model, "clean")
        for variant in TRAINED_VARIANTS[model]:
            p = test_predictions(model, variant)
            rows.append({"model": LABELS[model], "variant": variant, "test_mae": np.abs(p["y_pred"] - p["y_true"]).mean()})
            if variant != "clean":
                r = paired_bootstrap(p["y_pred"] - p["y_true"], clean["y_pred"] - clean["y_true"])
                boots.append({"model": LABELS[model], "comparison": f"{variant} - clean", "mae_variant": r["mae_a"],
                              "mae_clean": r["mae_b"], "diff": r["diff"], "ci95_low": r["ci_low"],
                              "ci95_high": r["ci_high"], "significant": r["significant"]})
    table = pd.DataFrame(rows).pivot(index="model", columns="variant", values="test_mae")
    table = table.reindex([LABELS[m] for m in MODELS])[list(utils.VARIANTS)]
    return table.round(4).reset_index(), pd.DataFrame(boots).round(4)


def threshold_errors(preds: dict, thresholds=(3, 5, 10)) -> pd.DataFrame:
    """What errors cost in practice: large misses, over- vs underestimates, and wrong decisions at
    "N+ years required" thresholds (wrongly rejected = true >= N but predicted < N)."""
    rows = []
    for model in MODELS:
        p = preds[model]
        err = p["y_pred"] - p["y_true"]
        row = {"model": LABELS[model], "n": len(p), "errors_over_2y": int((err.abs() > 2).sum()),
               "errors_over_5y": int((err.abs() > 5).sum()), "max_abs_error": err.abs().max(),
               "share_overestimated": (err > 0).mean(), "share_underestimated": (err < 0).mean()}
        for t in thresholds:
            row[f"wrongly_rejected_{t}y"] = int(((p["y_true"] >= t) & (p["y_pred"] < t)).sum())
            row[f"wrongly_passed_{t}y"] = int(((p["y_true"] < t) & (p["y_pred"] >= t)).sum())
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def fairness(preds: dict) -> pd.DataFrame:
    meta = pd.read_csv(utils.RAW_DIR / "metadata.csv", dtype={"id": str})[["id", "sex", "race"]]
    rows = []
    for model in ("bert", "rules"):
        p = preds[model].merge(meta, on="id")
        p["err"] = p["y_pred"] - p["y_true"]
        for attr in ("sex", "race"):
            for group, g in p.groupby(attr):
                mae_lo, mae_hi = bootstrap_mean(g["err"].abs())
                me_lo, me_hi = bootstrap_mean(g["err"])
                rows.append({"model": LABELS[model], "attribute": attr, "group": group, "n": len(g),
                             "mae": g["err"].abs().mean(), "mae_ci95_low": mae_lo, "mae_ci95_high": mae_hi,
                             "mean_error": g["err"].mean(), "mean_error_ci95_low": me_lo,
                             "mean_error_ci95_high": me_hi, "mean_true_years": g["y_true"].mean()})
    return pd.DataFrame(rows).round(4)


# ------------------------------------------------------------------ main

def main() -> None:
    from src import visualization as viz

    utils.set_seed()
    print("Test predictions (cached after the first run):", flush=True)
    preds = {m: test_predictions(m, "clean") for m in MODELS}
    assert all(p["id"].tolist() == preds["bert"]["id"].tolist() for p in preds.values()), "test ids differ"

    T, F = utils.TABLES_DIR, utils.FIGURES_DIR
    comparison = model_comparison(preds)
    comparison.to_csv(T / "model_comparison.csv", index=False)
    gain = bert_gain(preds)
    gain.to_csv(T / "bert_gain_bootstrap.csv", index=False)
    by_band, by_occ = errors_by(preds, "band"), errors_by(preds, "occupation")
    by_band.to_csv(T / "mae_by_band.csv", index=False)
    by_occ.to_csv(T / "mae_by_occupation.csv", index=False)
    comp = compression(preds)
    comp.to_csv(T / "compression_to_middle.csv", index=False)
    top = largest_bert_errors(preds)
    top.to_csv(T / "bert_top10_errors.csv", index=False)
    error_by_career_pattern(preds).to_csv(T / "error_by_career_pattern.csv", index=False)

    print("Q1 probe...", flush=True)
    probe_summary, probe_detail = probe()
    probe_summary.to_csv(T / "q1_probe_summary.csv", index=False)
    probe_detail.to_csv(T / "q1_probe_predictions.csv", index=False)
    print("Q2 OCR vs reference...", flush=True)
    q2 = ocr_vs_reference(preds)
    q2.to_csv(T / "q2_ocr_vs_reference.csv", index=False)
    print("Q3 text variants...", flush=True)
    q3, q3_boot = variant_sensitivity()
    q3.to_csv(T / "q3_variant_mae.csv", index=False)
    q3_boot.to_csv(T / "q3_variant_bootstrap.csv", index=False)
    q4 = fairness(preds)
    q4.to_csv(T / "q4_fairness.csv", index=False)
    threshold_errors(preds).to_csv(T / "q4_error_costs.csv", index=False)

    labelled = {LABELS[m]: preds[m] for m in MODELS}
    viz.plot_pred_vs_true(labelled, F / "pred_vs_true.png")
    viz.plot_mae_by_group(by_band, "band", utils.BAND_LABELS, F / "mae_by_band.png")
    viz.plot_mae_by_group(by_occ, "occupation", utils.OCCUPATIONS, F / "mae_by_occupation.png")
    viz.plot_band_confusion(labelled, F / "band_confusion.png")
    viz.plot_bert_gain(gain, F / "bert_gain.png")
    viz.plot_fairness(q4, F / "q4_fairness.png")

    summary = {"n_test": len(preds["bert"]), "n_bootstrap": N_BOOT, "seed": utils.SEED,
               "test_mae": {LABELS[m]: float(np.abs(preds[m]["y_pred"] - preds[m]["y_true"]).mean()) for m in MODELS},
               "bert_gain": gain.to_dict("records")}
    (utils.METRICS_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for name, df in [("model comparison", comparison), ("BERT gain", gain), ("compression", comp),
                         ("top-10 BERT errors", top.drop(columns="text")), ("Q1 probe", probe_summary),
                         ("Q2", q2), ("Q3", q3), ("Q3 bootstrap", q3_boot), ("Q4", q4)]:
            print(f"\n== {name}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()

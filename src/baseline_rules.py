"""Bonus baseline: date arithmetic with regular expressions (no training).

Extracts every job date range ("2016.08-2019.11"), including open ranges ending in
"Present"/"Current" and ranges truncated by the CV rendering ("2020.02-2023." or
"2020.02-2023.0"), merges overlapping periods and adds up the months.

Education years are ignored: only YYYY.MM-YYYY.MM ranges count as jobs.

Usage:
    python -m src.baseline_rules [--variant clean]
"""

import argparse
import re
import time

import numpy as np

from src import utils
from src.evaluation import regression_metrics

NAME = "rules"

# "Present"/"Current" means the dataset snapshot: the latest job end date in PopResume is 2023.01
REFERENCE_DATE = (2023, 1)
_DATE = r"((?:19|20)\d{2})\.(\d{1,2})?"
# Separator optional so the glued ranges of the raw OCR ("2020.112023.01") are also read
RANGE_RE = re.compile(
    rf"(?<!\d)((?:19|20)\d{{2}})\.(\d{{2}})\s*[-‐-―]?\s*(?:((?:19|20)\d{{2}})\.?(\d{{0,2}})|(present|current|now))",
    re.I,
)


def extract_ranges(text: str) -> list[tuple[int, int]]:
    """Job periods as (start, end) in months since year 0; truncated end months count as January."""
    periods = []
    for y1, m1, y2, m2, open_end in RANGE_RE.findall(text or ""):
        start = int(y1) * 12 + int(m1) - 1
        if open_end:
            end = REFERENCE_DATE[0] * 12 + REFERENCE_DATE[1] - 1
        else:
            month = int(m2) if m2 and 1 <= int(m2) <= 12 else 1
            end = int(y2) * 12 + month - 1
        if end >= start:
            periods.append((start, end))
    return periods


def merge_periods(periods: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(periods):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(p) for p in merged]


def years_of_experience(text: str) -> float:
    return sum(end - start for start, end in merge_periods(extract_ranges(text))) / 12


def predict(texts, variant: str = "clean") -> np.ndarray:
    """Sum of merged job periods, in years, clipped to [0, EXP_CAP]. `variant` is unused (no training)."""
    return utils.clip_years([years_of_experience(t) for t in texts])


def main(variant: str) -> dict:
    val = utils.load_split_texts("val", variant)
    start = time.perf_counter()
    y_val = predict(val["text"])
    infer = (time.perf_counter() - start) / len(val)
    n_ranges = [len(extract_ranges(t)) for t in val["text"]]

    utils.save_predictions(NAME, variant, "val", val, y_val)
    metrics = {"model": NAME, "variant": variant, "train_seconds": 0.0, "inference_seconds_per_cv": infer,
               "reference_date": f"{REFERENCE_DATE[0]}.{REFERENCE_DATE[1]:02d}",
               "val_cvs_without_ranges": int(sum(n == 0 for n in n_ranges)),
               "val_mean_ranges_per_cv": float(np.mean(n_ranges)),
               "val": regression_metrics(val["exp_year"], y_val)}
    utils.save_metrics(f"{NAME}__{variant}", metrics)
    print(f"[{NAME}/{variant}] val MAE {metrics['val']['mae']:.3f} · CVs without ranges "
          f"{metrics['val_cvs_without_ranges']}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=utils.VARIANTS, default="clean")
    main(parser.parse_args().variant)

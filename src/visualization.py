"""Figures for the README, notebooks and Streamlit app."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src import utils

# Shared style
SERIES_1 = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "font.size": 10,
})


def _hist(ax, values, bins, cap):
    ax.hist(values, bins=bins, color=SERIES_1, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(cap, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)


def plot_exp_year_distribution(meta: pd.DataFrame, path=None, cap: int = utils.EXP_CAP):
    """Histogram of raw exp_year, overall and per occupation, with the cap marked."""
    top = max(int(np.ceil(meta["exp_year"].max())), cap + 1)
    bins = np.arange(0, top + 1, 1)
    occupations = [o for o in utils.OCCUPATIONS if o in set(meta["occupation"])]

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    grid = fig.add_gridspec(2, len(occupations))
    ax_all = fig.add_subplot(grid[0, :])
    _hist(ax_all, meta["exp_year"], bins, cap)
    n_cap = int((meta["exp_year"] >= cap).sum())
    ax_all.set_title(f"All occupations (n={len(meta)}) · {n_cap} CVs ({100 * n_cap / len(meta):.1f}%) at ≥ {cap} years",
                     loc="left")
    ax_all.set_xlabel("Years of experience (exp_year, before clipping)")
    ax_all.set_ylabel("CVs")
    ax_all.annotate(f"cap = {cap}", xy=(cap, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", va="top", color=TEXT_SECONDARY)

    for i, occ in enumerate(occupations):
        ax = fig.add_subplot(grid[1, i], sharex=ax_all)
        values = meta.loc[meta["occupation"] == occ, "exp_year"]
        _hist(ax, values, bins, cap)
        ax.set_title(f"{occ.replace('_', ' ')}\nmedian {values.median():.1f} · n={len(values)}",
                     loc="left", fontsize=9)
        ax.set_xlabel("Years")
        if i == 0:
            ax.set_ylabel("CVs")

    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


SERIES_2 = "#eb6834"


def _short(occupation: str) -> str:
    return {"elementary_middle_school_teachers": "teachers", "accountants_auditors": "accountants",
            "construction_laborers": "construction", "registered_nurses": "nurses",
            "software_developers": "software dev."}.get(occupation, occupation)


def plot_ocr_quality(per_cv: pd.DataFrame, summary: pd.DataFrame, path=None):
    """CER and WER per CV by occupation (box plots) and share of years read correctly."""
    occupations = [o for o in utils.OCCUPATIONS if o in set(per_cv["occupation"])]
    labels = [_short(o) for o in occupations]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, col, name in zip(axes[:2], ["cer", "wer"], ["CER", "WER"]):
        data = [100 * per_cv.loc[per_cv["occupation"] == o, col] for o in occupations]
        ax.boxplot(data, tick_labels=labels, widths=0.5, patch_artist=True, showfliers=True,
                   boxprops={"facecolor": SERIES_1, "edgecolor": SERIES_1, "alpha": 0.35},
                   medianprops={"color": SERIES_1, "linewidth": 2},
                   whiskerprops={"color": TEXT_SECONDARY}, capprops={"color": TEXT_SECONDARY},
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": TEXT_SECONDARY,
                               "markeredgecolor": "none", "alpha": 0.5})
        mean = 100 * per_cv[col].mean()
        ax.set_title(f"{name} per CV (%) · mean {mean:.2f}%", loc="left")
        ax.set_ylabel(f"{name} (%)")
        ax.tick_params(axis="x", labelsize=8)

    ax = axes[2]
    s = summary.set_index("occupation").loc[occupations]
    x = np.arange(len(occupations))
    for k, (col, color, name) in enumerate([("pct_years_correct", SERIES_1, "OCR as is"),
                                           ("pct_years_correct_after_date_fix", SERIES_2, "after date-range repair")]):
        bars = ax.bar(x + (k - 0.5) * 0.38, s[col], width=0.36, color=color, label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, color=TEXT_SECONDARY, padding=2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% of reference years found in OCR")
    ax.set_title("Years read correctly", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_confidence(lines: pd.DataFrame, threshold: float, path=None):
    """Distribution of OCR line confidence and how often lines are read exactly, by confidence bin."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    ax = axes[0]
    ax.hist(lines["score"], bins=np.arange(0.925, 1.0005, 0.0025), color=SERIES_1, edgecolor=SURFACE, linewidth=1)
    ax.axvline(threshold, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    share = (lines["score"] < threshold).mean()
    ax.annotate(f"threshold {threshold}\n{100 * share:.1f}% of lines below", xy=(threshold, 1),
                xycoords=("data", "axes fraction"), xytext=(-6, -4), textcoords="offset points",
                ha="right", va="top", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(f"OCR line confidence · {len(lines):,} lines · min {lines['score'].min():.3f}", loc="left")
    ax.set_xlabel("PaddleOCR recognition score")
    ax.set_ylabel("Lines")

    ax = axes[1]
    edges = np.array([0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0])
    binned = lines.assign(bin=pd.cut(lines["score"], edges))
    rate = binned.groupby("bin", observed=True)["exact_match"].agg(["mean", "size"])
    labels = [f"{iv.left:.3f}–{iv.right:.3f}" for iv in rate.index]
    bars = ax.bar(range(len(rate)), 100 * rate["mean"], color=SERIES_1, width=0.6)
    ax.bar_label(bars, labels=[f"{100 * m:.0f}%\nn={n:,}" for m, n in zip(rate["mean"], rate["size"])],
                 fontsize=7, color=TEXT_SECONDARY, padding=2)
    ax.set_xticks(range(len(rate)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% of lines found verbatim in the reference")
    ax.set_title("Exact-read rate by confidence bin", loc="left")
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


def plot_training_curve(history: pd.DataFrame, best_epoch: int, baselines: dict | None = None, title="", path=None):
    """Training loss and validation MAE per epoch (two panels: different units, no dual axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], color=SERIES_1, linewidth=2, marker="o", markersize=6)
    ax.set_title("Training loss (scaled target)", loc="left")
    ax.set_xlabel("Epoch")
    ax = axes[1]
    ax.plot(history["epoch"], history["val_mae"], color=SERIES_1, linewidth=2, marker="o", markersize=6, label="BERT")
    best = history.loc[history["epoch"] == best_epoch, "val_mae"].iloc[0]
    ax.annotate(f"best: epoch {best_epoch}\n{best:.3f} years", xy=(best_epoch, best), xytext=(8, 12),
                textcoords="offset points", fontsize=9, color=TEXT_PRIMARY)
    for name, value in (baselines or {}).items():
        ax.axhline(value, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate(f"{name} {value:.2f}", xy=(1, value), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title("Validation MAE (years)", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylim(bottom=0)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


# Box colour per reading-order column: categorical slots 1-3 of the reference palette
COLUMN_COLORS = {"full": SERIES_1, "left": "#eb6834", "right": "#1baf7a"}


def plot_reading_order(record: dict, order: dict, image, ax=None, title=None, crop=True):
    """CV image with its OCR boxes, numbered in reading order and coloured by column.

    Every box of the same visual line shares one number, so a job title and its
    dates on the same line read as a single step.
    """
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 11))
    ax.imshow(image)
    ax.grid(False)
    ax.axis("off")
    boxes = [line["box"] for line in record["lines"]]
    for n, line in enumerate(order["lines"], 1):
        color = COLUMN_COLORS[line["column"]]
        for i in line["box_ids"]:
            x1, y1, x2, y2 = boxes[i]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.2))
        x1, y1, _, y2 = boxes[line["box_ids"][0]]
        ax.text(x1 - 6, (y1 + y2) / 2, str(n), ha="right", va="center", fontsize=7, color="white",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": color, "edgecolor": "none"})
    if order["split_x"] is not None:
        ax.axvline(order["split_x"], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    if crop and boxes:
        ax.set_ylim(min(max(b[3] for b in boxes) + 40, record["height"]), 0)
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    return ax

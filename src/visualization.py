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


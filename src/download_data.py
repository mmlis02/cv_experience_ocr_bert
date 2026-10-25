"""Stratified sampling and download of no-photo CV images from PopResume.

For each occupation the image shards are streamed, filtered to has_photo == False,
shuffled with seed 42 (buffer 5000) and the first N CVs are kept. Images are saved
as data/raw/images/<id>.png. The run is resumable: finished occupations are skipped
and images already on disk are not rewritten.

Usage:
    python -m src.download_data            # 800 CVs per occupation
    python -m src.download_data --n 5      # quick test
"""

import argparse
import io
import json
import time

import pandas as pd
from datasets import Image, load_dataset
from PIL import Image as PILImage

from src import utils
from src.visualization import plot_exp_year_distribution

HF_REPO = "sumin-yu/PopResume"
IMAGE_SHARDS = "hf://datasets/" + HF_REPO + "/data/images/{occupation}-*.parquet"
BUFFER_SIZE = 5000

# Only the target and the attributes needed for the fairness check (Question 4).
# age, names, state and region are deliberately not stored.
METADATA_COLUMNS = ["id", "occupation", "exp_year", "sex", "race"]

PARTS_DIR = utils.RAW_DIR / "parts"
METADATA_PATH = utils.RAW_DIR / "metadata.csv"
SAMPLED_IDS_PATH = utils.RAW_DIR / "sampled_ids.csv"
REFERENCE_TEXT_PATH = utils.RAW_DIR / "reference_text.csv"
STATS_PATH = utils.METRICS_DIR / "download_stats.json"


def stream_occupation(occupation: str, buffer_size: int = BUFFER_SIZE):
    """Stream the no-photo CVs of one occupation in a seeded shuffled order."""
    ds = load_dataset(
        "parquet",
        data_files=IMAGE_SHARDS.format(occupation=occupation),
        split="train",
        streaming=True,
        # Pushed down to parquet row-group statistics: photo row groups are skipped
        # without being downloaded. The filter below keeps the assignment semantics.
        filters=[("has_photo", "==", False)],
    )
    ds = ds.cast_column("image", Image(decode=False))  # raw bytes, decoded only when saved
    return ds.filter(lambda r: not r["has_photo"]).shuffle(seed=utils.SEED, buffer_size=buffer_size)


def save_png(image_bytes: bytes, path) -> None:
    """Write the image as PNG via a temporary file so interrupted writes never look complete."""
    tmp = path.with_suffix(".png.tmp")
    PILImage.open(io.BytesIO(image_bytes)).convert("RGB").save(tmp, format="PNG")
    tmp.replace(path)


def sample_occupation(occupation: str, n: int, buffer_size: int = BUFFER_SIZE) -> dict:
    """Sample n CVs of one occupation; returns timing stats. Skips work already done."""
    part_path = PARTS_DIR / f"{occupation}.csv"
    if part_path.exists():
        part = pd.read_csv(part_path, dtype={"id": str})
        if len(part) >= n and all((utils.RAW_IMAGES_DIR / f"{i}.png").exists() for i in part["id"]):
            print(f"[{occupation}] already sampled ({len(part)} CVs), skipping")
            return {"occupation": occupation, "n": len(part), "skipped": True}

    start = time.perf_counter()
    rows, n_written = [], 0
    for r in stream_occupation(occupation, buffer_size).take(n):
        img_path = utils.RAW_IMAGES_DIR / f"{r['id']}.png"
        if not img_path.exists():
            save_png(r["image"]["bytes"], img_path)
            n_written += 1
        rows.append({"id": r["id"], "occupation": occupation, "exp_year": r["exp_year"],
                     "sex": r["sex"], "race": r["race"]})
        if len(rows) == 1:
            print(f"[{occupation}] first CV after {time.perf_counter() - start:.0f}s (shuffle buffer filled)")
    elapsed = time.perf_counter() - start

    pd.DataFrame(rows, columns=METADATA_COLUMNS).to_csv(part_path, index=False)
    print(f"[{occupation}] {len(rows)} CVs ({n_written} new images) in {elapsed:.0f}s")
    return {"occupation": occupation, "n": len(rows), "new_images": n_written,
            "seconds": round(elapsed, 1), "skipped": False}


def save_reference_text(ids: set) -> pd.DataFrame:
    """Keep only id + text of the sampled CVs (used only to measure OCR quality)."""
    text = load_dataset(HF_REPO, "text", split="train").select_columns(["id", "text"]).to_pandas()
    ref = text[text["id"].isin(ids)].sort_values("id").reset_index(drop=True)
    ref.to_csv(REFERENCE_TEXT_PATH, index=False)
    return ref


def exp_year_summary(meta: pd.DataFrame, cap: int = utils.EXP_CAP) -> pd.DataFrame:
    """Distribution of raw exp_year, overall and per occupation, with the share at the cap."""
    groups = [("all", meta)] + list(meta.groupby("occupation"))
    summary = pd.DataFrame([{
        "occupation": name,
        "n": len(g),
        "mean": g["exp_year"].mean(),
        "std": g["exp_year"].std(),
        "min": g["exp_year"].min(),
        "q25": g["exp_year"].quantile(0.25),
        "median": g["exp_year"].median(),
        "q75": g["exp_year"].quantile(0.75),
        "max": g["exp_year"].max(),
        f"n_ge_{cap}": int((g["exp_year"] >= cap).sum()),
        f"pct_ge_{cap}": 100 * (g["exp_year"] >= cap).mean(),
    } for name, g in groups])
    return summary.round(2)


def main(n: int, buffer_size: int = BUFFER_SIZE) -> None:
    for d in (utils.RAW_IMAGES_DIR, PARTS_DIR, utils.TABLES_DIR, utils.FIGURES_DIR, utils.METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    stats = [sample_occupation(occ, n, buffer_size) for occ in utils.OCCUPATIONS]

    meta = pd.concat(
        [pd.read_csv(PARTS_DIR / f"{occ}.csv", dtype={"id": str}).head(n) for occ in utils.OCCUPATIONS],
        ignore_index=True,
    )
    assert meta["id"].is_unique, "CV ids must be unique across occupations"
    meta.to_csv(METADATA_PATH, index=False)
    meta[["id", "occupation"]].to_csv(SAMPLED_IDS_PATH, index=False)

    ref = save_reference_text(set(meta["id"]))
    missing = len(meta) - len(ref)
    print(f"Reference text saved for {len(ref)} CVs" + (f" ({missing} missing!)" if missing else ""))

    summary = exp_year_summary(meta)
    summary.to_csv(utils.TABLES_DIR / "exp_year_distribution.csv", index=False)
    plot_exp_year_distribution(meta, utils.FIGURES_DIR / "exp_year_distribution.png")

    total = time.perf_counter() - start
    image_sizes = [(utils.RAW_IMAGES_DIR / f"{i}.png").stat().st_size for i in meta["id"]]
    run = {
        "n_per_occupation": n,
        "buffer_size": buffer_size,
        "seed": utils.SEED,
        "total_seconds": round(total, 1),
        "mean_image_kb": round(sum(image_sizes) / len(image_sizes) / 1024, 1),
        "total_images_mb": round(sum(image_sizes) / 1024**2, 1),
        "occupations": stats,
    }
    STATS_PATH.write_text(json.dumps(run, indent=2))

    print(summary.to_string(index=False))
    pct_cap = summary.loc[summary["occupation"] == "all", f"pct_ge_{utils.EXP_CAP}"].item()
    if pct_cap > 15:
        print(f"WARNING: {pct_cap:.1f}% of sampled CVs have exp_year >= {utils.EXP_CAP} (> 15%): "
              "consider raising the cap (see docs/assignment.md).")
    print(f"Done in {total:.0f}s; mean image {run['mean_image_kb']} KB, total {run['total_images_mb']} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=800, help="CVs per occupation (default 800)")
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE, help="shuffle buffer size")
    args = parser.parse_args()
    main(args.n, args.buffer_size)

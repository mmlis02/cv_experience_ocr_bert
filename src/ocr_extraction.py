"""PaddleOCR on each CV image: cached per-CV JSON with text, confidence and boxes.

Every image in data/raw/metadata.csv is read once with PaddleOCR (CPU) and saved as
data/ocr/<id>.json. Existing JSONs are skipped, so the run is resumable. Failures are
appended to data/ocr/failed.log and reported in output/metrics/ocr_summary.json.

The reading order is rebuilt from the boxes at load time (reading_order / ocr_text),
so it can be changed without re-running the OCR.

Usage:
    python -m src.ocr_extraction                         # all sampled CVs, recommended model
    python -m src.ocr_extraction --model server --limit 50
    python -m src.ocr_extraction --benchmark             # mobile vs server on 20 CVs + parallel test
    python -m src.ocr_extraction --figures               # reading-order figures
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src import utils

# Detection + recognition model per configuration. Model names are explicit because
# lang="en" alone resolves to PP-OCRv6_medium in PaddleOCR 3.7, and lang="en" with
# ocr_version="PP-OCRv5" mixes the server detector with the mobile recogniser.
MODELS = {
    "mobile": ("PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"),
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
}
DEFAULT_MODEL = "mobile"
DEFAULT_THREADS = 8  # one process: all 8 cores of the M2 (PaddleOCR's default of 10 oversubscribes them)
# Full runs: 4 processes x 2 threads were 2.15x faster than 1 x 8 (output/tables/ocr_parallel_test.csv)
DEFAULT_WORKERS, DEFAULT_WORKER_THREADS = 4, 2

FAILED_LOG = utils.OCR_DIR / "failed.log"
SUMMARY_PATH = utils.METRICS_DIR / "ocr_summary.json"
BENCHMARK_DIR = utils.DATA_DIR / "ocr_benchmark"


# --------------------------------------------------------------------------- OCR

def build_ocr(model: str = DEFAULT_MODEL, cpu_threads: int = DEFAULT_THREADS):
    """PaddleOCR 3.x pipeline with the assignment settings (no orientation, no unwarping), on CPU."""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR

    det, rec = MODELS[model]
    return PaddleOCR(
        text_detection_model_name=det,
        text_recognition_model_name=rec,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        cpu_threads=cpu_threads,
    )


def run_ocr(ocr, image_path: Path, model: str) -> dict:
    """OCR one image; returns the JSON-ready record (one entry per detected text line)."""
    from PIL import Image

    with Image.open(image_path) as im:
        width, height = im.size
    start = time.perf_counter()
    res = ocr.predict(str(image_path))[0]
    seconds = time.perf_counter() - start
    lines = [
        {"text": text, "score": round(float(score), 4), "box": [int(v) for v in box]}
        for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"])
    ]
    return {
        "id": image_path.stem,
        "model": model,
        "det_model": MODELS[model][0],
        "rec_model": MODELS[model][1],
        "width": width,
        "height": height,
        "seconds": round(seconds, 3),
        "n_lines": len(lines),
        "mean_score": round(float(np.mean([l["score"] for l in lines])), 4) if lines else None,
        "lines": lines,
    }


def write_json(record: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    tmp.replace(path)


def load_ocr(cv_id: str, ocr_dir: Path = utils.OCR_DIR) -> dict:
    return json.loads((ocr_dir / f"{cv_id}.json").read_text())


# ----------------------------------------------------------------- reading order

def _find_column_split(boxes: np.ndarray, width: int) -> float | None:
    """x position of a vertical gap separating two columns, or None for a single column.

    Candidate splits are scanned in the middle 20-80% of the page. A split is accepted
    when few boxes cross it (full-width headers may), both sides hold at least 4 boxes,
    and the two sides overlap vertically, i.e. they really sit side by side.
    """
    if len(boxes) < 8:
        return None
    best_x, best_cross = None, None
    for x in np.arange(0.2 * width, 0.8 * width, 2):
        cross = int(((boxes[:, 0] < x) & (boxes[:, 2] > x)).sum())
        if best_cross is None or cross < best_cross:
            best_x, best_cross = x, cross
    if best_cross > max(2, 0.1 * len(boxes)):
        return None
    left = boxes[boxes[:, 2] <= best_x]
    right = boxes[boxes[:, 0] >= best_x]
    if len(left) < 4 or len(right) < 4:
        return None
    overlap = min(left[:, 3].max(), right[:, 3].max()) - max(left[:, 1].min(), right[:, 1].min())
    shorter = min(left[:, 3].max() - left[:, 1].min(), right[:, 3].max() - right[:, 1].min())
    if overlap < 0.5 * shorter:
        return None
    # Split at the centre of the empty gap around best_x
    gap_lo = max([b[2] for b in boxes if b[2] <= best_x], default=best_x)
    gap_hi = min([b[0] for b in boxes if b[0] >= best_x], default=best_x)
    return float((gap_lo + gap_hi) / 2)


def _group_lines(idx: list[int], boxes: np.ndarray) -> list[list[int]]:
    """Group boxes into visual lines (top to bottom), each line sorted left to right.

    A box joins the current line when it overlaps it vertically by at least half
    of the smaller height.
    """
    order = sorted(idx, key=lambda i: (boxes[i, 1] + boxes[i, 3]) / 2)
    lines, top, bottom = [], None, None
    for i in order:
        y1, y2 = boxes[i, 1], boxes[i, 3]
        if lines:
            overlap = min(bottom, y2) - max(top, y1)
            if overlap >= 0.5 * min(bottom - top, y2 - y1):
                lines[-1].append(i)
                top, bottom = min(top, y1), max(bottom, y2)
                continue
        lines.append([i])
        top, bottom = y1, y2
    return [sorted(line, key=lambda i: boxes[i, 0]) for line in lines]


def reading_order(record: dict) -> dict:
    """Rebuild the reading order of one OCR record.

    Boxes that cross the column split (full-width headers or sections) cut the page
    into horizontal bands. Inside each band the left column is read completely, then
    the right column; inside a column, boxes are grouped into lines top to bottom.

    Returns {"layout", "split_x", "lines": [{"text", "box_ids", "column"}]}.
    """
    if not record["lines"]:
        return {"layout": "empty", "split_x": None, "lines": []}
    boxes = np.array([l["box"] for l in record["lines"]], dtype=float)
    split = _find_column_split(boxes, record["width"])
    if split is None:
        groups = [(line, "full") for line in _group_lines(list(range(len(boxes))), boxes)]
    else:
        spanning = [i for i in range(len(boxes)) if boxes[i, 0] < split < boxes[i, 2]]
        rest = [i for i in range(len(boxes)) if i not in set(spanning)]
        y_mid = (boxes[:, 1] + boxes[:, 3]) / 2
        # Each spanning box closes a horizontal band at its vertical centre
        cuts = [(y_mid[i], i) for i in sorted(spanning, key=lambda i: y_mid[i])] + [(np.inf, None)]
        groups, start = [], -np.inf
        for cut, span_id in cuts:
            band = [i for i in rest if start <= y_mid[i] < cut]
            left = [i for i in band if (boxes[i, 0] + boxes[i, 2]) / 2 < split]
            right = [i for i in band if (boxes[i, 0] + boxes[i, 2]) / 2 >= split]
            groups += [(line, "left") for line in _group_lines(left, boxes)]
            groups += [(line, "right") for line in _group_lines(right, boxes)]
            if span_id is not None:
                groups.append(([span_id], "full"))
            start = cut
    lines = [
        {"text": " ".join(record["lines"][i]["text"] for i in ids), "box_ids": ids, "column": col}
        for ids, col in groups
    ]
    return {"layout": "two_columns" if split is not None else "one_column", "split_x": split, "lines": lines}


def ocr_text(record: dict) -> str:
    """OCR text of one CV in reading order, one visual line per text line."""
    return "\n".join(line["text"] for line in reading_order(record)["lines"])


# ------------------------------------------------------------------- batch runs

_WORKER_OCR = None


def _init_worker(model: str, cpu_threads: int) -> None:
    global _WORKER_OCR
    _WORKER_OCR = build_ocr(model, cpu_threads)


def _process(args) -> tuple[str, dict | None, str | None]:
    cv_id, image_path, out_path, model = args
    try:
        record = run_ocr(_WORKER_OCR, Path(image_path), model)
        write_json(record, Path(out_path))
        return cv_id, {"seconds": record["seconds"], "mean_score": record["mean_score"],
                       "n_lines": record["n_lines"]}, None
    except Exception:  # keep going; the failure is logged and reported
        return cv_id, None, traceback.format_exc(limit=3)


def process_ids(ids, model=DEFAULT_MODEL, out_dir=utils.OCR_DIR, workers=1,
                cpu_threads=DEFAULT_THREADS, log_failures=True, verbose=True) -> dict:
    """OCR the given CV ids into out_dir, skipping existing JSONs. Returns run stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, str(utils.RAW_IMAGES_DIR / f"{i}.png"), str(out_dir / f"{i}.json"), model)
            for i in ids if not (out_dir / f"{i}.json").exists()]
    if verbose:
        print(f"[{model}] {len(ids) - len(todo)} cached, {len(todo)} to process "
              f"({workers} worker(s) x {cpu_threads} threads)")
    failed, start = {}, time.perf_counter()
    if todo:
        if workers == 1:
            _init_worker(model, cpu_threads)
            results = map(_process, todo)
        else:
            pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model, cpu_threads))
            results = pool.imap_unordered(_process, todo)
        for k, (cv_id, stats, err) in enumerate(results, 1):
            if err:
                failed[cv_id] = err
                if log_failures:
                    with open(out_dir / "failed.log", "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {cv_id}\n{err}\n")
            if verbose and (k % 25 == 0 or k == len(todo)):
                elapsed = time.perf_counter() - start
                eta = elapsed / k * (len(todo) - k)
                print(f"  {k}/{len(todo)} done · {elapsed / k:.1f}s/CV wall · ETA {eta / 60:.0f} min", flush=True)
        if workers > 1:
            pool.close()
            pool.join()
    return {"processed": len(todo) - len(failed), "failed": failed,
            "wall_seconds": time.perf_counter() - start}


def hardware_info() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    import paddle
    import paddleocr
    return {
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(int(sysctl("hw.memsize") or 0) / 1024**3, 1),
        "device": "cpu",
        "python": platform.python_version(),
        "paddlepaddle": paddle.__version__,
        "paddleocr": paddleocr.__version__,
    }


def _file_time_span(ids) -> float:
    times = sorted((utils.OCR_DIR / f"{i}.json").stat().st_mtime for i in ids if (utils.OCR_DIR / f"{i}.json").exists())
    return times[-1] - times[0] if len(times) > 1 else 0.0


def summarize(ids, model: str, workers: int, cpu_threads: int, wall_seconds=None, n_processed=0) -> dict:
    """ocr_summary.json: time per CV, hardware, confidence and failed images."""
    records = [load_ocr(i) for i in ids if (utils.OCR_DIR / f"{i}.json").exists()]
    failed = sorted(set(ids) - {r["id"] for r in records})
    seconds = np.array([r["seconds"] for r in records])
    scores = np.concatenate([[l["score"] for l in r["lines"]] for r in records]) if records else np.array([])
    layouts = pd.Series([reading_order(r)["layout"] for r in records]).value_counts().to_dict()
    summary = {
        "n_images": len(ids),
        "n_ok": len(records),
        "n_failed": len(failed),
        "failed_ids": failed,
        "det_model": MODELS[model][0],
        "rec_model": MODELS[model][1],
        "workers": workers,
        "cpu_threads": cpu_threads,
        "seconds_per_cv_mean": round(float(seconds.mean()), 2) if len(seconds) else None,
        "seconds_per_cv_median": round(float(np.median(seconds)), 2) if len(seconds) else None,
        # seconds_per_cv_* is the latency of one process; the wall time per CV is the throughput
        "wall_seconds_last_run": round(wall_seconds, 1) if n_processed else None,
        "wall_seconds_per_cv_last_run": round(wall_seconds / n_processed, 2) if n_processed else None,
        # Span between the first and last JSON written: wall time of the run(s) that produced them
        "wall_hours_from_file_times": round(_file_time_span(ids) / 3600, 2) if records else None,
        "mean_line_confidence": round(float(scores.mean()), 4) if len(scores) else None,
        "mean_cv_confidence": round(float(np.mean([r["mean_score"] for r in records if r["mean_score"]])), 4)
        if records else None,
        "share_lines_below_0.9": round(float((scores < 0.9).mean()), 4) if len(scores) else None,
        "mean_lines_per_cv": round(float(np.mean([r["n_lines"] for r in records])), 1) if records else None,
        "layouts": layouts,
        "hardware": hardware_info(),
    }
    utils.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------------------- benchmark

def benchmark_ids(per_occupation: int = 4) -> list[str]:
    meta = pd.read_csv(utils.RAW_DIR / "metadata.csv", dtype={"id": str})
    return (meta.groupby("occupation", group_keys=False)
                .apply(lambda g: g.sample(per_occupation, random_state=utils.SEED))["id"].tolist())


def compare_models(ids, models=("mobile", "server"), cpu_threads=DEFAULT_THREADS) -> pd.DataFrame:
    """Time per CV, CER, WER and year recall of each model on the same CVs."""
    from src.ocr_quality import cer, wer, year_recall
    from src.preprocessing import repair_date_ranges

    ref = pd.read_csv(utils.RAW_DIR / "reference_text.csv", dtype={"id": str}).set_index("id")["text"]
    rows = []
    for model in models:
        out_dir = BENCHMARK_DIR / model
        process_ids(ids, model, out_dir, workers=1, cpu_threads=cpu_threads, log_failures=False)
        for cv_id in ids:
            record = load_ocr(cv_id, out_dir)
            text = ocr_text(record)
            ok, total = year_recall(text, ref[cv_id])
            ok_fixed, _ = year_recall(repair_date_ranges(text), ref[cv_id])
            rows.append({"model": model, "id": cv_id, "seconds": record["seconds"],
                         "mean_score": record["mean_score"], "cer": cer(text, ref[cv_id]),
                         "wer": wer(text, ref[cv_id]), "cer_after_date_fix": cer(repair_date_ranges(text), ref[cv_id]),
                         "wer_after_date_fix": wer(repair_date_ranges(text), ref[cv_id]), "years_ok": ok, "years_ok_after_date_fix": ok_fixed,
                         "years_total": total})
    per_cv = pd.DataFrame(rows)
    per_cv.to_csv(utils.TABLES_DIR / "ocr_model_comparison_per_cv.csv", index=False)
    table = per_cv.groupby("model").agg(
        n=("id", "size"), seconds_per_cv=("seconds", "mean"), mean_confidence=("mean_score", "mean"),
        cer=("cer", "mean"), wer=("wer", "mean"), cer_after_date_fix=("cer_after_date_fix", "mean"),
        wer_after_date_fix=("wer_after_date_fix", "mean"), years_ok=("years_ok", "sum"),
        years_ok_after_date_fix=("years_ok_after_date_fix", "sum"), years_total=("years_total", "sum"))
    table["pct_years_correct"] = 100 * table["years_ok"] / table["years_total"]
    table["pct_years_correct_after_date_fix"] = 100 * table["years_ok_after_date_fix"] / table["years_total"]
    table = table.reset_index().round(4)
    table.to_csv(utils.TABLES_DIR / "ocr_model_comparison.csv", index=False)
    return table


def parallel_test(ids, model=DEFAULT_MODEL, configs=((1, 8), (1, 4), (2, 4), (4, 2))) -> pd.DataFrame:
    """Wall-clock throughput of (workers, threads per worker) configurations on the same CVs."""
    import shutil

    rows = []
    for workers, threads in configs:
        out_dir = BENCHMARK_DIR / f"parallel_{workers}x{threads}"
        shutil.rmtree(out_dir, ignore_errors=True)
        # Wall time includes loading the model once per worker, as in a real run
        stats = process_ids(ids, model, out_dir, workers=workers, cpu_threads=threads,
                            log_failures=False, verbose=False)
        rows.append({"model": model, "workers": workers, "threads_per_worker": threads, "n": len(ids),
                     "wall_seconds": round(stats["wall_seconds"], 1),
                     "wall_seconds_per_cv": round(stats["wall_seconds"] / len(ids), 2),
                     "failed": len(stats["failed"])})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table["speedup_vs_1x8"] = (table["wall_seconds"].iloc[0] / table["wall_seconds"]).round(2)
    table.to_csv(utils.TABLES_DIR / "ocr_parallel_test.csv", index=False)
    return table


# ------------------------------------------------------------------- figures

def synthetic_two_column(id_left: str, id_right: str, path: Path) -> Path:
    """Two-column test page: a full-width banner over the bodies of two real CVs side by side.

    PopResume CVs use a single-column template, so this page is the check that the
    column detection and the left-then-right reading order work when a layout needs it.
    """
    from PIL import Image, ImageDraw, ImageFont

    left = Image.open(utils.RAW_IMAGES_DIR / f"{id_left}.png").convert("RGB")
    right = Image.open(utils.RAW_IMAGES_DIR / f"{id_right}.png").convert("RGB")
    body_top, body_bottom, body_width = 230, 1000, 760
    page = Image.new("RGB", (1680, 1300), "white")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(page).text((100, 60), "SYNTHETIC TWO-COLUMN PAGE: LEFT CV BODY | RIGHT CV BODY, READ LEFT THEN RIGHT",
                              fill="black", font=font)
    page.paste(left.crop((60, body_top, 60 + body_width, body_bottom)), (40, 160))
    page.paste(right.crop((60, body_top, 60 + body_width, body_bottom)), (880, 160))
    page.save(path)
    return path


def save_reading_order_figures(model: str = DEFAULT_MODEL) -> list[Path]:
    """Numbered reading-order boxes for 3 CVs of different occupations + the two-column check."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from src.visualization import plot_reading_order

    meta = pd.read_csv(utils.RAW_DIR / "metadata.csv", dtype={"id": str})
    picks = [meta[meta["occupation"] == occ].sample(1, random_state=utils.SEED).iloc[0]
             for occ in ["registered_nurses", "software_developers", "construction_laborers"]]
    out_dir = BENCHMARK_DIR / model
    ocr = None
    paths = []
    for row in picks:
        if not (out_dir / f"{row['id']}.json").exists():
            ocr = ocr or build_ocr(model)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_ocr(ocr, utils.RAW_IMAGES_DIR / f"{row['id']}.png", model), out_dir / f"{row['id']}.json")
        record = load_ocr(row["id"], out_dir)
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
        plot_reading_order(record, reading_order(record), Image.open(utils.RAW_IMAGES_DIR / f"{row['id']}.png"), ax,
                           title=f"{row['occupation'].replace('_', ' ')} · id {row['id']} · {model} model")
        path = utils.FIGURES_DIR / f"ocr_reading_order_{row['occupation']}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # Two-column check on a synthetic page built from two of the CVs above
    synth = synthetic_two_column(picks[0]["id"], picks[1]["id"], utils.RAW_DIR / "synthetic_two_column.png")
    ocr = ocr or build_ocr(model)
    record = run_ocr(ocr, synth, model)
    fig, ax = plt.subplots(figsize=(11, 8.5), layout="constrained")
    order = reading_order(record)
    plot_reading_order(record, order, Image.open(synth), ax,
                       title=f"Two-column check (synthetic page) · detected layout: {order['layout']}")
    path = utils.FIGURES_DIR / "ocr_reading_order_two_column_check.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths + [path]


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel OCR processes")
    parser.add_argument("--threads", type=int, default=DEFAULT_WORKER_THREADS, help="CPU threads per process")
    parser.add_argument("--limit", type=int, default=None, help="only the first N sampled CVs")
    parser.add_argument("--benchmark", nargs="?", const="all", choices=["all", "compare", "parallel"],
                        help="mobile vs server on 20 CVs (compare), parallel test (parallel) or both (all)")
    parser.add_argument("--figures", action="store_true", help="reading-order figures for 3 CVs + two-column check")
    args = parser.parse_args()

    for d in (utils.OCR_DIR, utils.TABLES_DIR, utils.METRICS_DIR, utils.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        ids = benchmark_ids()
        if args.benchmark in ("all", "parallel"):
            print(parallel_test(ids).to_string(index=False))
        if args.benchmark in ("all", "compare"):
            print(compare_models(ids).to_string(index=False))
        return
    if args.figures:
        for path in save_reading_order_figures(args.model):
            print("saved", path)
        return

    ids = pd.read_csv(utils.RAW_DIR / "sampled_ids.csv", dtype={"id": str})["id"].tolist()
    if args.limit:
        ids = ids[: args.limit]
    stats = process_ids(ids, args.model, utils.OCR_DIR, args.workers, args.threads)
    summary = summarize(ids, args.model, args.workers, args.threads, stats["wall_seconds"], stats["processed"])
    print(json.dumps({k: v for k, v in summary.items() if k != "hardware"}, indent=2))


if __name__ == "__main__":
    main()

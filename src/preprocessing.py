"""Cleaning, reading order, PII removal and masking of direct answers.

Builds three text variants from the cached OCR (same columns: id, text, exp_year, occupation):

- raw.csv                    reading-order OCR text, no repair
- clean.csv                  + date-range repair, bullet/dash normalisation, broken-line and
                               hyphenation joining
- clean_dropped_lowconf.csv  clean, after dropping OCR lines below LOW_CONF_THRESHOLD

PII removal (name, email, phone, address) and masking of direct answers (total experience,
age, date of birth) are rules, not cleaning choices, so they are applied to every variant.
Job dates and graduation years are kept: they are the evidence the models must reason from.
The reference text gets the same clean pipeline (reference_clean.csv); it is only used to
evaluate Question 2 and is never a model input.

exp_year is clipped to [0, EXP_CAP].

Usage:
    python -m src.preprocessing
    python -m src.preprocessing --token-lengths     # DistilBERT tokens per CV
"""

import json
import re

import numpy as np
import pandas as pd

from src import utils

# Two "YYYY.MM" dates glued together: the OCR often drops the en dash of a date range
# ("2020.11–2023.01" is read as "2020.112023.01"), which hides the second year.
_MONTH_DATE = r"(?:19|20)\d{2}\.(?:0[1-9]|1[0-2])"
_GLUED_RANGE_RE = re.compile(rf"(?<!\d)({_MONTH_DATE})({_MONTH_DATE})(?!\d)")

# Lines below this OCR confidence are dropped in the clean_dropped_lowconf variant.
# Line scores are very high (min 0.929, mean 0.991, sd 0.0085); 0.97 ~ mean - 2.5 sd
# flags the low tail (2.1% of lines) as outliers. See notebooks/03 for the analysis.
LOW_CONF_THRESHOLD = 0.97

MASK = "[MASKED]"

US_STATES = (
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM "
    "NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR"
).split()

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}")
ADDRESS_RE = re.compile(rf"\b(?:{'|'.join(US_STATES)})\s*,\s*USA\b")

# Phrases that state the answer directly. Job date ranges and graduation years never match:
# every pattern needs words such as "years ... experience", "age", "old" or "born".
DIRECT_ANSWER_PATTERNS = {
    "total_experience": re.compile(
        r"(?:\b(?:over|more than|nearly|almost|about|around)\s+)?(?:\b\d{1,2}|\b(?:one|two|three|four|five|six|seven|eight|"
        r"nine|ten|eleven|twelve|fifteen|twenty))\+?\s*(?:years?|yrs?)'?\s*(?:of\s+)?(?:\w+\s+){0,3}experience\b"
        r"|\bexperience\s*[:\-]\s*\d{1,2}\+?\s*(?:years?|yrs?)\b"
        r"|\b(?:a|over a|more than a|two)\s+decades?\s+of\s+(?:\w+\s+)?experience\b",
        re.I),
    "age": re.compile(
        r"\bage\s*[:\-]?\s*\d{2}\b|\baged\s+\d{2}\b|\b\d{2}\s*(?:years?|yrs?)[\s-]*old\b", re.I),
    "date_of_birth": re.compile(
        r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s*date)\s*[:\-]?\s*[\w./, -]{0,20}?\d{4}\b"
        r"|\bborn\s+(?:on|in)?\s*[\w./, -]{0,20}?\d{4}\b", re.I),
}


# ------------------------------------------------------------------ text repair

def repair_date_ranges(text: str) -> str:
    """Re-insert the dash between two glued YYYY.MM dates."""
    return _GLUED_RANGE_RE.sub(r"\1-\2", text)


def normalize_symbols(text: str) -> str:
    """Unify dashes to "-" and bullet markers at line starts to "- "."""
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"(?m)^\s*[-•·*]\s*", "- ", text)


def join_broken_lines(lines: list[str]) -> list[str]:
    """Join hyphenated line breaks ("manage-" + "ment") and continuation lines.

    A line continues the previous one when it starts with a lowercase letter and the
    previous line does not end a sentence or a field.
    """
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if out and re.search(r"[A-Za-z]-$", out[-1]) and re.match(r"[a-z]", s):
            out[-1] = out[-1][:-1] + s
        elif out and re.match(r"[a-z]", s) and not re.search(r"[.:;|,]$", out[-1]):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


# ------------------------------------------------------------------ PII and masking

def detect_name(lines: list[str]) -> str | None:
    """The candidate's name: the first line (letters only, at most 5 words) when the contact line follows it."""
    lines = [l.strip() for l in lines if l.strip()]
    is_contact = len(lines) > 1 and bool(EMAIL_RE.search(lines[1]) or PHONE_RE.search(lines[1]))
    if is_contact and not re.search(r"[\d@|]", lines[0]) and len(lines[0].split()) <= 5:
        return lines[0]
    return None


def remove_pii(lines: list[str], name: str | None = None) -> tuple[list[str], dict]:
    """Remove the name, email, phone and address. Returns (lines, what was found).

    `name` should be detected on the full CV (detect_name) so it is still removed when
    other lines, such as the contact line, have been dropped.
    """
    found = {"name": False, "email": False, "phone": False, "address": False}
    name = name or detect_name(lines)
    # Only the full name is removed: single name tokens ("Day", "Hall") are often ordinary words
    name_re = re.compile(r"\b" + r"\s+".join(map(re.escape, name.split())) + r"\b", re.I) if name else None
    out = []
    for line in lines:
        if name_re and name_re.search(line):
            found["name"] = True
            line = name_re.sub(" ", line)
        for key, rx in (("email", EMAIL_RE), ("phone", PHONE_RE), ("address", ADDRESS_RE)):
            if rx.search(line):
                found[key] = True
                line = rx.sub(" ", line)
        line = re.sub(r"\s*\|\s*(?:\|\s*)*", " | ", line)
        line = re.sub(r"\s+", " ", line).strip(" |")
        if line:
            out.append(line)
    return out, found


def mask_direct_answers(text: str) -> tuple[str, dict]:
    """Replace phrases that state total experience, age or date of birth with [MASKED]."""
    hits = {}
    for key, rx in DIRECT_ANSWER_PATTERNS.items():
        text, n = rx.subn(MASK, text)
        hits[key] = n
    return text, hits


# ------------------------------------------------------------------ pipelines

def ocr_lines(record: dict, min_score: float | None = None) -> list[str]:
    """Reading-order lines of an OCR record, optionally without low-confidence boxes."""
    from src.ocr_extraction import reading_order

    keep = None
    if min_score is not None:
        keep = {i for i, l in enumerate(record["lines"]) if l["score"] >= min_score}
    lines = []
    for line in reading_order(record)["lines"]:
        ids = [i for i in line["box_ids"] if keep is None or i in keep]
        if ids:
            lines.append(" ".join(record["lines"][i]["text"] for i in ids))
    return lines


def process_lines(lines: list[str], clean: bool, name: str | None = None) -> tuple[str, dict]:
    """PII removal + masking (always) and, if clean, repair and normalisation."""
    name = name or detect_name(lines)
    if clean:
        lines = [repair_date_ranges(normalize_symbols(l)) for l in lines]
        lines = join_broken_lines(lines)
    lines, pii = remove_pii(lines, name)
    text, masked = mask_direct_answers("\n".join(lines))
    return text, {**{f"pii_{k}": v for k, v in pii.items()}, **{f"masked_{k}": v for k, v in masked.items()}}


def build_variants(meta: pd.DataFrame, ref: pd.Series) -> tuple[dict, pd.DataFrame]:
    from src.ocr_extraction import load_ocr

    base = meta[["id", "occupation"]].assign(exp_year=np.clip(meta["exp_year"], 0, utils.EXP_CAP))
    texts = {"raw": [], "clean": [], "clean_dropped_lowconf": [], "reference_clean": []}
    log = []
    for cv_id in base["id"]:
        record = load_ocr(cv_id)
        name = detect_name(ocr_lines(record))  # on the full CV, before any line is dropped
        raw, _ = process_lines(ocr_lines(record), clean=False, name=name)
        clean, info_clean = process_lines(ocr_lines(record), clean=True, name=name)
        lowconf, _ = process_lines(ocr_lines(record, LOW_CONF_THRESHOLD), clean=True, name=name)
        reference, info_ref = process_lines(ref[cv_id].split("\n"), clean=True)
        texts["raw"].append(raw)
        texts["clean"].append(clean)
        texts["clean_dropped_lowconf"].append(lowconf)
        texts["reference_clean"].append(reference)
        n_lines = len(record["lines"])
        n_low = sum(l["score"] < LOW_CONF_THRESHOLD for l in record["lines"])
        log.append({"id": cv_id, **info_clean,
                    **{f"ref_{k}": v for k, v in info_ref.items() if k.startswith("masked")},
                    "glued_date_ranges": len(_GLUED_RANGE_RE.findall("\n".join(ocr_lines(record)))),
                    "n_ocr_lines": n_lines, "n_lowconf_lines": n_low})
    frames = {name: base.assign(text=t)[["id", "text", "exp_year", "occupation"]] for name, t in texts.items()}
    return frames, pd.DataFrame(log)


CONF_BINS = [0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0]


def confidence_lines(meta: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    """One row per OCR line: confidence, line type and whether its text appears exactly in the reference."""
    from src.ocr_extraction import load_ocr
    from src.ocr_quality import line_type, normalize_text

    rows = []
    for cv_id in meta["id"]:
        ref_norm = normalize_text(ref[cv_id])
        for k, line in enumerate(load_ocr(cv_id)["lines"]):
            text = normalize_text(line["text"])
            rows.append({"id": cv_id, "score": line["score"], "line_type": line_type(line["text"], k),
                         "exact_match": bool(text) and text in ref_norm})
    return pd.DataFrame(rows)


def confidence_table(lines: pd.DataFrame) -> pd.DataFrame:
    binned = lines.assign(bin=pd.cut(lines["score"], CONF_BINS))
    table = binned.groupby("bin", observed=True).agg(lines=("score", "size"), exact_match_rate=("exact_match", "mean"))
    table["share_of_lines"] = table["lines"] / len(lines)
    table["cumulative_share"] = table["share_of_lines"].cumsum()
    return table.reset_index().assign(bin=lambda d: d["bin"].astype(str)).round(4)


def token_lengths(tokenizer_name: str = "distilbert-base-uncased") -> pd.DataFrame:
    """Tokens per CV (with [CLS]/[SEP]) for every text variant, and how many exceed 512."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    rows = []
    for variant in utils.VARIANTS:
        df = pd.read_csv(utils.PROCESSED_DIR / f"{variant}.csv", dtype={"id": str})
        n = np.array([len(ids) for ids in tokenizer(df["text"].tolist())["input_ids"]])
        rows.append({"variant": variant, "tokenizer": tokenizer_name, "mean": n.mean(), "median": np.median(n),
                     "p95": np.percentile(n, 95), "p99": np.percentile(n, 99), "max": int(n.max()),
                     "n_over_128": int((n > 128).sum()), "n_over_256": int((n > 256).sum()),
                     "n_over_512": int((n > 512).sum())})
    table = pd.DataFrame(rows).round(1)
    table.to_csv(utils.TABLES_DIR / "token_lengths.csv", index=False)
    return table


def main() -> None:
    from src.visualization import plot_confidence

    meta = pd.read_csv(utils.RAW_DIR / "metadata.csv", dtype={"id": str})
    ref = pd.read_csv(utils.RAW_DIR / "reference_text.csv", dtype={"id": str}).set_index("id")["text"]

    lines = confidence_lines(meta, ref)
    conf = confidence_table(lines)
    conf.to_csv(utils.TABLES_DIR / "ocr_confidence_by_bin.csv", index=False)
    (lines[lines["score"] < LOW_CONF_THRESHOLD]["line_type"].value_counts()
     .rename_axis("line_type").reset_index(name="lines_below_threshold")
     .to_csv(utils.TABLES_DIR / "ocr_lowconf_lines_by_type.csv", index=False))
    plot_confidence(lines, LOW_CONF_THRESHOLD, utils.FIGURES_DIR / "ocr_confidence.png")
    print(conf.to_string(index=False))

    frames, log = build_variants(meta, ref)

    utils.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(utils.PROCESSED_DIR / f"{name}.csv", index=False)
    log.to_csv(utils.TABLES_DIR / "preprocessing_log.csv", index=False)

    masked_cols = [c for c in log if c.startswith("masked_")]
    report = {
        "n_cvs": len(log),
        "cvs_with_direct_answer_masked": int((log[masked_cols].sum(axis=1) > 0).sum()),
        "masked_phrases_by_type": {c.removeprefix("masked_"): int(log[c].sum()) for c in masked_cols},
        "reference_cvs_with_direct_answer_masked":
            int((log[[f"ref_{c}" for c in masked_cols]].sum(axis=1) > 0).sum()),
        "cvs_with_pii_removed": {k: int(log[f"pii_{k}"].sum()) for k in ("name", "email", "phone", "address")},
        "cvs_with_glued_date_ranges_repaired": int((log["glued_date_ranges"] > 0).sum()),
        "glued_date_ranges_repaired": int(log["glued_date_ranges"].sum()),
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "lowconf_lines_dropped": int(log["n_lowconf_lines"].sum()),
        "lowconf_share_of_lines": round(float(log["n_lowconf_lines"].sum() / log["n_ocr_lines"].sum()), 4),
        "cvs_with_lowconf_lines": int((log["n_lowconf_lines"] > 0).sum()),
        "mean_chars": {name: round(float(df["text"].str.len().mean()), 1) for name, df in frames.items()},
    }
    (utils.METRICS_DIR / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-lengths", action="store_true", help="only measure DistilBERT tokens per CV")
    if parser.parse_args().token_lengths:
        print(token_lengths().to_string(index=False))
    else:
        main()

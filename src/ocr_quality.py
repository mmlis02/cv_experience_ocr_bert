"""CER / WER and year-recall of OCR text against the reference text.

Both texts are normalised the same way before scoring, so the metrics measure
reading errors rather than differences in case, bullet markers, separators,
dashes or line breaks.

Usage:
    python -m src.ocr_quality      # per-CV metrics, tables and figures in output/
"""

import difflib
import re
import unicodedata
from collections import Counter

import jiwer
import pandas as pd

from src import utils

YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
DATE_RANGE_RE = re.compile(r"(?:19|20)\d{2}\.\d{2}\s*-\s*(?:19|20)\d{2}\.\d{2}")
BULLET_RE = re.compile(r"(?m)^\s*[-•·*]\s*")


def normalize_text(text: str) -> str:
    """Normalise case, punctuation and whitespace the same way for OCR and reference text.

    Lowercase; unify unicode dashes and quotes; drop bullet markers at line starts
    (the OCR reads them inconsistently and they carry no content); put single spaces
    around "|" separators; collapse all whitespace. Other punctuation is kept because
    it is part of dates ("2019.11-2023.01") and emails.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = BULLET_RE.sub("", text)
    text = re.sub(r"\s*\|\s*", " | ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def cer(ocr_text: str, ref_text: str) -> float:
    """Character error rate of the OCR text against the reference (after normalisation)."""
    return jiwer.cer(normalize_text(ref_text), normalize_text(ocr_text))


def wer(ocr_text: str, ref_text: str) -> float:
    """Word error rate of the OCR text against the reference (after normalisation)."""
    return jiwer.wer(normalize_text(ref_text), normalize_text(ocr_text))


def years(text: str) -> list[str]:
    """All four-digit years (1900-2099) in the text."""
    return YEAR_RE.findall(text or "")


def year_recall(ocr_text: str, ref_text: str) -> tuple[int, int]:
    """(years read correctly, years in the reference), counting repeated years separately."""
    ref, ocr = Counter(years(ref_text)), Counter(years(ocr_text))
    return sum(min(n, ocr[y]) for y, n in ref.items()), sum(ref.values())


# ------------------------------------------------------------------ error analysis

def line_type(line: str, position: int) -> str:
    """Coarse type of a reference CV line, used to see where the OCR fails."""
    s = line.strip()
    if position == 0:
        return "name"
    if "@" in s or re.search(r"\+1-\d", s):
        return "contact"
    if s.isupper():
        return "section header"
    if DATE_RANGE_RE.search(normalize_text(s)):
        return "job title + dates"
    if BULLET_RE.match(line):
        return "bullet"
    if YEAR_RE.search(s):
        return "education + year"
    return "other"


def line_errors(ocr_lines: list[str], ref_text: str) -> list[dict]:
    """CER of every reference line against its best-matching OCR line."""
    ocr_norm = [normalize_text(l) for l in ocr_lines]
    rows = []
    ref_lines = [l for l in ref_text.split("\n") if l.strip()]
    for pos, line in enumerate(ref_lines):
        target = normalize_text(line)
        if not target:
            continue
        best = max(ocr_norm, key=lambda o: difflib.SequenceMatcher(None, target, o).ratio(), default="")
        out = jiwer.process_characters(target, best or " ")
        rows.append({"line_type": line_type(line, pos), "ref_chars": len(target),
                     "char_errors": out.substitutions + out.deletions + out.insertions})
    return rows


def word_error_pairs(ocr_text: str, ref_text: str) -> list[tuple[str, str]]:
    """(reference words, OCR words) for every non-matching chunk of the word alignment."""
    out = jiwer.process_words(normalize_text(ref_text), normalize_text(ocr_text))
    ref_w, hyp_w = out.references[0], out.hypotheses[0]
    return [(" ".join(ref_w[c.ref_start_idx:c.ref_end_idx]), " ".join(hyp_w[c.hyp_start_idx:c.hyp_end_idx]))
            for c in out.alignments[0] if c.type != "equal"]


def generalize(s: str) -> str:
    """Replace digits with # so recurring error patterns group together."""
    return re.sub(r"\d", "#", s)


# ------------------------------------------------------------------ main

def evaluate(texts: pd.Series, ocr_lines: dict, ref: pd.Series, meta: pd.DataFrame):
    """Per-CV metrics, per-line-type errors and the most frequent word errors."""
    from src.preprocessing import repair_date_ranges

    rows, line_rows, pairs = [], [], Counter()
    for cv_id, text in texts.items():
        ok, total = year_recall(text, ref[cv_id])
        ok_fix, _ = year_recall(repair_date_ranges(text), ref[cv_id])
        rows.append({"id": cv_id, "cer": cer(text, ref[cv_id]), "wer": wer(text, ref[cv_id]),
                     "cer_after_date_fix": cer(repair_date_ranges(text), ref[cv_id]),
                     "years_ok": ok, "years_ok_after_date_fix": ok_fix, "years_total": total})
        line_rows += [{"id": cv_id, **r} for r in line_errors(ocr_lines[cv_id], ref[cv_id])]
        pairs.update((generalize(r), generalize(h)) for r, h in word_error_pairs(text, ref[cv_id]))
    per_cv = pd.DataFrame(rows).merge(meta[["id", "occupation"]], on="id")
    by_line = pd.DataFrame(line_rows)
    top = pd.DataFrame([(r, h, n) for (r, h), n in pairs.most_common(25)],
                       columns=["reference (digits as #)", "ocr (digits as #)", "count"])
    return per_cv, by_line, top


def summarize(per_cv: pd.DataFrame) -> pd.DataFrame:
    groups = [("all", per_cv)] + list(per_cv.groupby("occupation"))
    table = pd.DataFrame([{
        "occupation": name, "n": len(g),
        "cer_mean": g["cer"].mean(), "cer_median": g["cer"].median(), "cer_p90": g["cer"].quantile(0.9),
        "wer_mean": g["wer"].mean(), "wer_median": g["wer"].median(), "wer_p90": g["wer"].quantile(0.9),
        "cer_after_date_fix_mean": g["cer_after_date_fix"].mean(),
        "pct_years_correct": 100 * g["years_ok"].sum() / g["years_total"].sum(),
        "pct_years_correct_after_date_fix": 100 * g["years_ok_after_date_fix"].sum() / g["years_total"].sum(),
        "pct_cvs_all_years_correct_after_date_fix":
            100 * (g["years_ok_after_date_fix"] == g["years_total"]).mean(),
    } for name, g in groups])
    return table.round(4)


def main() -> None:
    from src.ocr_extraction import load_ocr, ocr_text, reading_order
    from src.visualization import plot_ocr_quality

    meta = pd.read_csv(utils.RAW_DIR / "metadata.csv", dtype={"id": str})
    ref = pd.read_csv(utils.RAW_DIR / "reference_text.csv", dtype={"id": str}).set_index("id")["text"]
    records = {i: load_ocr(i) for i in meta["id"]}
    texts = pd.Series({i: ocr_text(r) for i, r in records.items()})
    ocr_lines = {i: [l["text"] for l in reading_order(r)["lines"]] for i, r in records.items()}

    per_cv, by_line, top = evaluate(texts, ocr_lines, ref, meta)
    summary = summarize(per_cv)
    line_table = (by_line.groupby("line_type")[["ref_chars", "char_errors"]].sum()
                  .assign(cer=lambda d: d["char_errors"] / d["ref_chars"],
                          share_of_all_errors=lambda d: d["char_errors"] / d["char_errors"].sum())
                  .sort_values("share_of_all_errors", ascending=False).round(4).reset_index())

    utils.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_cv.to_csv(utils.TABLES_DIR / "ocr_quality_per_cv.csv", index=False)
    summary.to_csv(utils.TABLES_DIR / "ocr_quality_by_occupation.csv", index=False)
    line_table.to_csv(utils.TABLES_DIR / "ocr_errors_by_line_type.csv", index=False)
    top.to_csv(utils.TABLES_DIR / "ocr_top_word_errors.csv", index=False)
    plot_ocr_quality(per_cv, summary, utils.FIGURES_DIR / "ocr_quality.png")

    print(summary.to_string(index=False))
    print(line_table.to_string(index=False))
    print(top.head(12).to_string(index=False))


if __name__ == "__main__":
    main()

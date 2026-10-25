"""CER / WER and year-recall of OCR text against the reference text.

Both texts are normalised the same way before scoring, so the metrics measure
reading errors rather than differences in case, punctuation spacing or line breaks.
"""

import re
import unicodedata
from collections import Counter

import jiwer

YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def normalize_text(text: str) -> str:
    """Lowercase, unify unicode dashes/quotes and collapse all whitespace to single spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
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

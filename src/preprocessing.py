"""Cleaning, reading order, PII removal and masking of direct answers."""

import re

# Two "YYYY.MM" dates glued together: the OCR often drops the en dash of a date range
# ("2020.11–2023.01" is read as "2020.112023.01"), which hides the second year.
_MONTH_DATE = r"(?:19|20)\d{2}\.(?:0[1-9]|1[0-2])"
_GLUED_RANGE_RE = re.compile(rf"(?<!\d)({_MONTH_DATE})({_MONTH_DATE})(?!\d)")


def repair_date_ranges(text: str) -> str:
    """Re-insert the dash between two glued YYYY.MM dates."""
    return _GLUED_RANGE_RE.sub(r"\1-\2", text)

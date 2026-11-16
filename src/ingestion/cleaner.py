"""
cleaner.py
----------
Text-cleaning utilities applied to raw text extracted from PDFs before
chunking. Handles the noise typical of scanned/printed tourism brochures:
running headers/footers, page numbers, duplicated whitespace, hyphenated
line-wraps, and stray OCR artifacts.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Patterns that typically indicate a running header/footer or bare page number
# on its own line, e.g. "Page 12", "12", "| Page 4 |".
_PAGE_NUMBER_LINE = re.compile(r"^\s*(\|?\s*)?(page\s*)?\d{1,4}(\s*\|)?\s*$", re.IGNORECASE)

# Collapse 3+ blank lines into a maximum of 2.
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")

# Collapse runs of spaces/tabs.
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# De-hyphenate words broken across a line wrap, e.g. "moun-\ntain" -> "mountain"
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")

# Common repeated boilerplate lines seen across the source brochures.
_BOILERPLATE_LINES = {
    "cover",
    "cover inside",
    "back inside",
    "back page",
}


def _is_boilerplate(line: str) -> bool:
    stripped = line.strip().lower()
    if not stripped:
        return True
    if stripped in _BOILERPLATE_LINES:
        return True
    if _PAGE_NUMBER_LINE.match(stripped):
        return True
    return False


def remove_page_artifacts(text: str) -> str:
    """Strip bare page-number lines and known boilerplate lines."""
    lines = text.split("\n")
    kept: List[str] = [line for line in lines if not _is_boilerplate(line)]
    return "\n".join(kept)


def dehyphenate(text: str) -> str:
    """Rejoin words that were split across a line break with a hyphen."""
    return _HYPHEN_LINEBREAK.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Collapse excess whitespace while preserving paragraph breaks."""
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    # Trim trailing whitespace on each line.
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def remove_repeated_running_headers(pages: Iterable[str], min_repeats: int = 3) -> List[str]:
    """
    Detect lines that repeat verbatim across many pages (typical running
    headers/footers such as a book title on every page) and strip them.

    Args:
        pages: list of raw page texts (one string per page).
        min_repeats: minimum number of pages a line must appear on, verbatim,
            to be considered a running header/footer.

    Returns:
        List of page texts with the detected running lines removed.
    """
    pages = list(pages)
    line_counts: dict[str, int] = {}
    for page in pages:
        seen_this_page = set()
        for raw_line in page.split("\n"):
            line = raw_line.strip()
            if not line or len(line) > 80:
                continue
            if line in seen_this_page:
                continue
            seen_this_page.add(line)
            line_counts[line] = line_counts.get(line, 0) + 1

    running_lines = {
        line for line, count in line_counts.items() if count >= max(min_repeats, len(pages) // 3)
    }

    cleaned_pages = []
    for page in pages:
        kept_lines = [
            line for line in page.split("\n") if line.strip() not in running_lines
        ]
        cleaned_pages.append("\n".join(kept_lines))
    return cleaned_pages


def clean_text(raw_text: str) -> str:
    """
    Full single-page cleaning pipeline: dehyphenate -> remove boilerplate ->
    normalize whitespace. Use `remove_repeated_running_headers` separately at
    the document level before calling this per-page.
    """
    text = dehyphenate(raw_text)
    text = remove_page_artifacts(text)
    text = normalize_whitespace(text)
    return text

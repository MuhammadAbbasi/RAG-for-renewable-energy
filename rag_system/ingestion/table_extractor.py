"""
table_extractor.py — Structured table extraction from PDFs using pdfplumber.

Tables are converted to Markdown format so they can be stored as regular
text chunks in the vector database and queried naturally.

Example output for a 3-column table:
    | Parametro | Valore | Unità |
    |-----------|--------|-------|
    | Potenza   | 3.5    | MW    |
    | Altezza   | 150    | m     |
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _rows_to_markdown(rows: list[list]) -> str:
    """Convert a list of rows (each row = list of cell values) to Markdown table."""
    if not rows:
        return ""

    # Sanitize cells: replace None with empty string, flatten newlines
    clean_rows = []
    for row in rows:
        clean_row = []
        for cell in row:
            if cell is None:
                cell = ""
            cell = str(cell).replace("\n", " ").replace("|", "\\|").strip()
            clean_row.append(cell)
        clean_rows.append(clean_row)

    # Determine column count from the widest row
    col_count = max(len(r) for r in clean_rows)

    # Pad all rows to the same width
    padded = [r + [""] * (col_count - len(r)) for r in clean_rows]

    # Build Markdown
    header    = "| " + " | ".join(padded[0]) + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in padded[1:]]

    return "\n".join([header, separator] + data_rows)


def extract_tables_from_pdf(pdf_path: Path) -> dict[int, list[str]]:
    """
    Extract all tables from a PDF, organized by page number.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping 1-based page_number → list of Markdown table strings.
        Pages with no tables are not included.
    """
    result: dict[int, list[str]] = {}

    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber is not installed. Run: pip install pdfplumber")
        return result

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_number = page_idx + 1
                try:
                    tables = page.extract_tables()
                    if not tables:
                        continue

                    md_tables = []
                    for tbl in tables:
                        if not tbl or len(tbl) < 2:
                            # Skip single-row or empty tables (likely noise)
                            continue
                        md = _rows_to_markdown(tbl)
                        if md:
                            md_tables.append(md)

                    if md_tables:
                        result[page_number] = md_tables

                except Exception as exc:
                    logger.warning(
                        "Table extraction failed on page %d of %s: %s",
                        page_number, pdf_path.name, exc
                    )

    except Exception as exc:
        logger.error("Cannot open %s with pdfplumber: %s", pdf_path.name, exc)

    return result


def tables_to_text(tables: list[str], page_number: int, filename: str) -> str:
    """
    Wrap extracted tables with context text for chunking.
    Each table gets a header line so the chunker treats it as a self-contained unit.
    """
    parts = []
    for i, tbl in enumerate(tables, start=1):
        header = f"[Tabella {i} — Pagina {page_number} — {filename}]"
        parts.append(f"{header}\n{tbl}")
    return "\n\n".join(parts)

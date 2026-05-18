"""
pdf_parser.py — Per-page PDF parsing using PyMuPDF.

For each page this module determines:
  - Whether the page has a native text layer or is scanned (image-only).
  - Extracts raw text (if text layer exists).
  - Extracts embedded images as PIL Image objects for further processing.
  - Returns a list of PageContent dataclass instances.

Downstream consumers:
  - ocr_engine.py  → receives pages where is_scanned=True
  - image_captioner.py → receives images list from any page
  - table_extractor.py → works independently via pdfplumber on the same file
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

# Silence MuPDF's C-level warnings ("format error: non-page object in page tree"
# etc.) which are printed directly to stderr and cannot be suppressed via Python
# logging. We redirect stderr around the noisy calls.
fitz.TOOLS.mupdf_display_errors(False)   # PyMuPDF ≥ 1.18 — suppress MuPDF errors


@contextmanager
def _quiet_mupdf():
    """Context manager that swallows any remaining MuPDF stderr output."""
    # mupdf_display_errors(False) handles most cases; this is a belt-and-suspenders
    # guard for any residual output on very malformed PDFs.
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = old_stderr

from rag_system import config

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Holds all extracted content for a single PDF page."""
    pdf_path: str
    project_id: str
    page_number: int          # 1-based
    total_pages: int
    text: str                 # Native text layer (empty if scanned)
    is_scanned: bool          # True → needs OCR
    images: list[dict]        # List of {"bytes": bytes, "width": int, "height": int, "xref": int}
    ocr_text: Optional[str] = None            # Filled later by ocr_engine
    image_captions: list[str] = field(default_factory=list)  # Filled later by image_captioner
    tables: list[str] = field(default_factory=list)          # Filled later by table_extractor


def is_page_scanned(page: fitz.Page) -> bool:
    """
    A page is considered scanned if its text layer has fewer characters
    than SCANNED_TEXT_THRESHOLD (default 50).  This catches pages that are
    just images with maybe a header/footer.
    """
    text = page.get_text("text").strip()
    return len(text) < config.SCANNED_TEXT_THRESHOLD


def extract_images_from_page(page: fitz.Page, doc: fitz.Document) -> list[dict]:
    """
    Extract embedded images from a page, skipping tiny icons.
    Returns a list of dicts with raw bytes and dimensions.
    """
    images = []
    image_list = page.get_images(full=True)

    for img_info in image_list:
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            width  = base_image["width"]
            height = base_image["height"]

            # Skip images that are too small (icons, logos, decorative elements)
            if width < config.IMAGE_MIN_WIDTH or height < config.IMAGE_MIN_HEIGHT:
                continue

            images.append({
                "bytes":  base_image["image"],
                "ext":    base_image["ext"],
                "width":  width,
                "height": height,
                "xref":   xref,
            })
        except Exception as exc:
            logger.warning("Could not extract image xref=%s on page: %s", xref, exc)

    return images


def parse_pdf(pdf_path: Path, project_id: str) -> list[PageContent]:
    """
    Open a PDF and return a list of PageContent objects, one per page.

    Args:
        pdf_path:   Absolute path to the PDF file.
        project_id: The project folder name this PDF belongs to.

    Returns:
        List of PageContent. Empty list if the PDF cannot be opened.
    """
    pages: list[PageContent] = []

    try:
        with _quiet_mupdf():
            doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Cannot open PDF %s: %s", pdf_path, exc)
        return []

    total_pages = len(doc)
    logger.info("Parsing %s (%d pages)", pdf_path.name, total_pages)

    for page_idx in range(total_pages):
        page_number = page_idx + 1
        try:
            with _quiet_mupdf():
                page       = doc[page_idx]          # may raise "cannot find page N in tree"
                scanned    = is_page_scanned(page)
                native_txt = "" if scanned else page.get_text("text").strip()
                images     = extract_images_from_page(page, doc)

            pages.append(PageContent(
                pdf_path    = str(pdf_path),
                project_id  = project_id,
                page_number = page_number,
                total_pages = total_pages,
                text        = native_txt,
                is_scanned  = scanned,
                images      = images,
            ))
        except Exception as exc:
            logger.warning("Error on page %d of %s: %s", page_number, pdf_path.name, exc)
            # Still add a blank entry so page count stays consistent
            pages.append(PageContent(
                pdf_path    = str(pdf_path),
                project_id  = project_id,
                page_number = page_number,
                total_pages = total_pages,
                text        = "",
                is_scanned  = True,
                images      = [],
            ))

    doc.close()
    return pages


def render_page_as_image(pdf_path: Path, page_number: int) -> Optional[bytes]:
    """
    Render a single PDF page as a PNG image (for OCR via qwen2.5vl).

    Args:
        pdf_path:    Path to the PDF.
        page_number: 1-based page number.

    Returns:
        PNG bytes or None on failure.
    """
    try:
        with _quiet_mupdf():
            doc  = fitz.open(str(pdf_path))
            page = doc[page_number - 1]
            mat  = fitz.Matrix(config.OCR_DPI / 72, config.OCR_DPI / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            doc.close()
        return png_bytes
    except Exception as exc:
        logger.error("Cannot render page %d of %s: %s", page_number, pdf_path, exc)
        return None

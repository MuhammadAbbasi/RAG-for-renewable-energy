"""
chunker.py — Semantic, sentence-aware text chunking (Italian-optimised).

Converts processed page content into LangChain Document objects ready for
embedding. Each document carries rich metadata so Qdrant can filter results
by project, file, page, and content type.

Content types stored:
  - "text"          → native text layer from PDF
  - "ocr"           → text produced by Surya OCR (scanned page)
  - "table"         → Markdown table extracted by pdfplumber
  - "image_caption" → rich description produced by qwen2.5vl
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag_system import config
from rag_system.ingestion.pdf_parser import PageContent

logger = logging.getLogger(__name__)

# Italian-aware sentence separators (ordered from largest to smallest unit)
_ITALIAN_SEPARATORS = [
    "\n\n",        # Paragraph break
    "\n",          # Line break
    ". ",          # Sentence end (Italian)
    "! ",
    "? ",
    "; ",
    ": ",
    ", ",
    " ",
    "",
]

# One shared splitter instance (thread-safe, stateless)
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=_ITALIAN_SEPARATORS,
    length_function=len,   # character count (close enough to token count for Italian)
    is_separator_regex=False,
)


def _make_metadata(
    page: PageContent,
    content_type: str,
    chunk_index: int = 0,
) -> dict:
    """Build the metadata dict stored alongside each chunk in Qdrant."""
    filename = Path(page.pdf_path).name
    return {
        "project_id":    page.project_id,
        "filename":      filename,
        "pdf_path":      page.pdf_path,
        "page_number":   page.page_number,
        "total_pages":   page.total_pages,
        "content_type":  content_type,
        "chunk_index":   chunk_index,
        # Human-readable source string shown in Open-WebUI citations
        "source": f"{page.project_id} › {filename} › p.{page.page_number}",
    }


def _split_text(text: str, base_metadata: dict) -> list[Document]:
    """Split a text block into chunks, assigning sequential chunk indices."""
    if not text or len(text.strip()) < config.CHUNK_MIN_CHARS:
        return []

    raw_chunks = _splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(raw_chunks):
        if len(chunk.strip()) < config.CHUNK_MIN_CHARS:
            continue
        meta = {**base_metadata, "chunk_index": i}
        docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def chunk_page(page: PageContent) -> list[Document]:
    """
    Convert a fully-processed PageContent into a list of Documents.

    Processing order:
      1. Native text  (content_type="text")
      2. OCR text     (content_type="ocr")      — if page was scanned
      3. Tables       (content_type="table")    — one Document per table
      4. Img captions (content_type="image_caption") — one Document per image
    """
    docs: list[Document] = []

    # ── 1. Native text ────────────────────────────────────────────────────────
    if page.text and not page.is_scanned:
        meta = _make_metadata(page, content_type="text")
        docs.extend(_split_text(page.text, meta))

    # ── 2. OCR text ───────────────────────────────────────────────────────────
    if page.ocr_text:
        meta = _make_metadata(page, content_type="ocr")
        docs.extend(_split_text(page.ocr_text, meta))

    # ── 3. Tables (each table is a separate Document — not further split) ─────
    for tbl_idx, table_md in enumerate(page.tables):
        if len(table_md.strip()) < config.CHUNK_MIN_CHARS:
            continue
        meta = _make_metadata(page, content_type="table", chunk_index=tbl_idx)
        docs.append(Document(page_content=table_md, metadata=meta))

    # ── 4. Image captions ─────────────────────────────────────────────────────
    for cap_idx, caption in enumerate(page.image_captions):
        if not caption or len(caption.strip()) < config.CHUNK_MIN_CHARS:
            continue
        meta = _make_metadata(page, content_type="image_caption", chunk_index=cap_idx)
        # Captions are already concise — split only if very long
        docs.extend(_split_text(caption, meta))

    return docs


def chunk_pages(pages: list[PageContent]) -> list[Document]:
    """
    Chunk all pages from a PDF into Documents.

    Args:
        pages: List of PageContent objects (fully processed by pipeline).

    Returns:
        Flat list of Documents ready for embedding and storage.
    """
    all_docs: list[Document] = []
    for page in pages:
        page_docs = chunk_page(page)
        all_docs.extend(page_docs)

    logger.info(
        "Chunked %d pages → %d documents (file: %s)",
        len(pages),
        len(all_docs),
        Path(pages[0].pdf_path).name if pages else "?",
    )
    return all_docs

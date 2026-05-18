"""
pipeline.py — Main indexing pipeline.

This is the orchestrator that ties together every processing step:
  1. Scan DATA_DIR for project folders (auto-detects new ones).
  2. For each project folder:
       a. Check if the folder contains any PDF files.
       b. If EMPTY or UNREADABLE → log to skipped_projects.csv and continue.
       c. If OK → process each new/changed PDF through the full pipeline.
  3. For each PDF:
       a. pdf_parser       → extract text, detect scanned pages, extract images
       b. ocr_engine       → OCR scanned pages (EasyOCR GPU, parallel pre-render)
       c. table_extractor  → extract tables (pdfplumber)
       d. image_captioner  → caption embedded images (qwen2.5vl, skipped on scanned pages)
       e. chunker          → semantic chunking
       f. embedder         → bge-m3 vectors (batch API, ~10× faster)
       g. vector_store     → upsert into Qdrant
       h. tracker          → mark as indexed

Speed improvements vs. v1:
  • Parallel PNG rendering: all scanned pages are rendered to PNG concurrently
    (ThreadPoolExecutor, pure CPU) before GPU OCR starts — overlaps CPU+GPU work.
  • Batch embeddings: all chunks embedded in one Ollama /api/embed call instead
    of N sequential calls.  500 chunks: ~3s vs ~30s.
  • Fixed sleep: the 20s VRAM-swap cooldown only triggers when image captions
    were actually generated (not when images exist but all were skipped).
  • Multi-project workers: INDEXING_WORKERS > 1 processes multiple projects in
    parallel.  Text-only PDFs (no GPU OCR/captioning) benefit most; a GPU
    semaphore serialises GPU-heavy phases automatically.
  • Skip captioning on scanned pages: images on scanned pages ARE the page
    scan — captioning them with qwen2.5vl causes 8-min timeouts for zero gain.

Empty / skipped project folders are written to:
  logs/skipped_projects.csv  (appended every run, no duplicates)
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from rag_system import config
from rag_system.indexing import tracker, embedder, vector_store
from rag_system.ingestion import (
    pdf_parser,
    ocr_engine,
    image_captioner,
    table_extractor,
    chunker,
)

# Wiki extraction — optional, import gracefully so RAG still works if wiki module fails
try:
    from rag_system.wiki import extractor as wiki_extractor
    from rag_system.wiki import store as wiki_store
    _WIKI_AVAILABLE = True
except Exception as _wiki_import_err:
    _WIKI_AVAILABLE = False
    import warnings
    warnings.warn(f"Wiki module not available: {_wiki_import_err}")

logger = logging.getLogger(__name__)

# ── GPU serialisation ────────────────────────────────────────────────────────
# All GPU-heavy work (OCR, captioning, model unload) goes through this lock.
# This prevents two workers from fighting over VRAM when INDEXING_WORKERS > 1.
# CPU-only work (parsing, table extraction, chunking, embedding*) runs freely.
# * bge-m3 embedding is GPU-light and Ollama queues concurrent requests.
_GPU_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Ollama model management helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unload_model(model_name: str, timeout: float = 15.0):
    """
    Ask Ollama to evict a model from VRAM immediately (keep_alive=0).
    Critical after qwen2.5vl captioning so bge-m3 can load without VRAM OOM.
    """
    url     = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = {"model": model_name, "keep_alive": 0, "prompt": ""}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
        if resp.status_code in (200, 404):
            logger.info("  Model unloaded from VRAM: %s", model_name)
        else:
            logger.warning(
                "  Unload request for %s returned %d", model_name, resp.status_code
            )
    except Exception as exc:
        logger.warning("  Could not unload model %s: %s", model_name, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Empty-project CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDNAMES = [
    "project_id",
    "folder_path",
    "reason",
    "pdf_count",
    "empty_pdf_names",
    "detected_at",
]


def _ensure_csv_header():
    csv_path = config.SKIPPED_PROJECTS_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()


def _update_skipped_csv(
    project_id: str, reason: str, pdf_count: int, empty_pdf_names: list[str]
):
    csv_path = config.SKIPPED_PROJECTS_CSV
    rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    updated = False
    for row in rows:
        if row["project_id"] == project_id:
            row.update({
                "reason": reason,
                "pdf_count": str(pdf_count),
                "empty_pdf_names": "; ".join(empty_pdf_names),
                "detected_at": datetime.utcnow().isoformat(timespec="seconds"),
            })
            updated = True
    if updated:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)


def _log_skipped_project(
    project_id: str,
    folder_path: Path,
    reason: str,
    pdf_count: int = 0,
    empty_pdf_names: list[str] = None,
):
    _ensure_csv_header()
    csv_path = config.SKIPPED_PROJECTS_CSV
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            existing_ids = {row["project_id"] for row in csv.DictReader(f)}
        if project_id in existing_ids:
            logger.debug("Project '%s' already in skipped CSV — updating.", project_id)
            _update_skipped_csv(project_id, reason, pdf_count, empty_pdf_names or [])
            return

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writerow({
            "project_id":      project_id,
            "folder_path":     str(folder_path),
            "reason":          reason,
            "pdf_count":       str(pdf_count),
            "empty_pdf_names": "; ".join(empty_pdf_names or []),
            "detected_at":     datetime.utcnow().isoformat(timespec="seconds"),
        })

    # Also write to SQLite tracker
    try:
        tracker.record_skipped_project(project_id, folder_path, reason)
    except Exception:
        pass

    logger.warning("Skipped project: '%s' — %s", project_id, reason)


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_project_folder(
    folder: Path,
) -> tuple[bool, str, list[Path], list[str]]:
    """
    Returns (is_valid, reason, valid_pdfs, empty_pdf_names).
    A folder is valid if it contains at least one non-empty, readable PDF.
    """
    pdf_files  = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
    pdf_files  = list({p.resolve(): p for p in pdf_files}.values())

    if not pdf_files:
        return False, "no PDF files found", [], []

    valid_pdfs: list[Path] = []
    empty_names: list[str] = []

    for pdf in pdf_files:
        try:
            if pdf.stat().st_size < 100:
                empty_names.append(pdf.name)
                continue
            # Quick open check
            import fitz
            doc = fitz.open(str(pdf))
            n   = len(doc)
            doc.close()
            if n == 0:
                empty_names.append(pdf.name)
            else:
                valid_pdfs.append(pdf)
        except Exception:
            empty_names.append(pdf.name)

    if not valid_pdfs:
        reason = (
            f"all {len(pdf_files)} PDF(s) are empty or unreadable"
            if empty_names else "no readable PDFs"
        )
        return False, reason, [], empty_names

    return True, "", valid_pdfs, empty_names


# ─────────────────────────────────────────────────────────────────────────────
# Core PDF processor
# ─────────────────────────────────────────────────────────────────────────────

def _render_page(args: tuple) -> tuple[int, bytes]:
    """Worker: render one page to PNG bytes. Returns (page_number, png_bytes)."""
    pdf_path, page_number = args
    png = pdf_parser.render_page_as_image(pdf_path, page_number)
    return page_number, (png or b"")


def _process_pdf(pdf_path: Path, project_id: str) -> int:
    """
    Run the full ingestion pipeline on a single PDF file.

    Speed optimisations applied here:
      1. Parallel PNG rendering (ThreadPoolExecutor, CPU-bound, no GIL issue).
      2. Batch embeddings via Ollama /api/embed (one HTTP call for all chunks).
      3. GPU lock: OCR and captioning acquire _GPU_LOCK to prevent multi-worker
         VRAM fights; CPU work (parsing, chunking) runs freely in parallel.
      4. Conditional VRAM swap cooldown: only sleep when captions were actually
         generated (not just when images exist).

    Returns:
        Number of chunks successfully indexed.
    """
    logger.info("  → Processing: %s", pdf_path.name)

    # ── Pre-validation: detect corrupted PDFs before native libraries crash ──
    # pdfplumber uses pdfminer (pure Python) — if it can't open the file,
    # the PDF is malformed. Letting PyMuPDF/EasyOCR touch it risks SIGABRT.
    try:
        import pdfplumber as _pdfplumber
        with _pdfplumber.open(str(pdf_path)) as _probe:
            _ = len(_probe.pages)  # force full open
    except Exception as _probe_exc:
        logger.error(
            "  Pre-validation failed for %s — PDF is corrupted/unreadable (%s). Skipping.",
            pdf_path.name, _probe_exc,
        )
        return 0

    # ── Step 1: Parse pages ──────────────────────────────────────────────────
    pages = pdf_parser.parse_pdf(pdf_path, project_id)
    if not pages:
        logger.warning("  No pages extracted from %s", pdf_path.name)
        return 0

    # ── Step 2: Extract tables (pdfplumber, CPU) ─────────────────────────────
    logger.info("  [2/6] Extracting tables from %s…", pdf_path.name)
    tables_by_page = table_extractor.extract_tables_from_pdf(pdf_path)

    # ── Step 3: Per-page enrichment ──────────────────────────────────────────
    scanned_pages  = [p for p in pages if p.is_scanned]
    image_pages    = [p for p in pages if p.images and not p.is_scanned]
    n_scanned      = len(scanned_pages)
    n_images       = sum(len(p.images) for p in image_pages)
    logger.info(
        "  [3/6] Enriching %d pages  (%d scanned → OCR, %d embedded images → caption)…",
        len(pages), n_scanned, n_images,
    )

    # ── 3a: Pre-render all scanned pages to PNG in parallel (CPU threads) ────
    #  While the previous step's results are still warm in cache, we kick off
    #  all page renders simultaneously.  Each render is a PyMuPDF call that is
    #  purely CPU-bound and releases the GIL, so multiple threads help.
    png_cache: dict[int, bytes] = {}
    if scanned_pages:
        render_args = [(pdf_path, p.page_number) for p in scanned_pages]
        _render_workers = min(len(render_args), 4)   # cap at 4 threads
        logger.info(
            "    Pre-rendering %d scanned page(s) in parallel (%d threads)…",
            len(render_args), _render_workers,
        )
        with ThreadPoolExecutor(max_workers=_render_workers) as ex:
            for page_num, png in ex.map(_render_page, render_args):
                png_cache[page_num] = png
        logger.info("    All pages pre-rendered ✓")

    # ── 3b: OCR scanned pages (GPU, serialised) ──────────────────────────────
    captions_generated = False

    if scanned_pages:
        with _GPU_LOCK:
            for page in scanned_pages:
                pg       = f"p.{page.page_number}/{page.total_pages}"
                png_bytes = png_cache.get(page.page_number, b"")
                if png_bytes:
                    logger.info("    %s — OCR (scanned page)…", pg)
                    page.ocr_text = ocr_engine.ocr_page_bytes(png_bytes)
                    logger.info(
                        "    %s — OCR done (%d chars)", pg, len(page.ocr_text or "")
                    )

    # ── 3c: Attach tables to pages ───────────────────────────────────────────
    for page in pages:
        if page.page_number in tables_by_page:
            page.tables = [
                table_extractor.tables_to_text(
                    [tbl], page.page_number, pdf_path.name
                )
                for tbl in tables_by_page[page.page_number]
            ]

    # ── 3d: Caption embedded images (GPU, serialised, skip scanned pages) ────
    # IMPORTANT: Scanned pages have images that ARE the page scan itself.
    # Captioning them with qwen2.5vl always times out (~8 min each) and the
    # content is already covered by OCR.  Only caption embedded graphics on
    # text-based pages.
    if image_pages:
        with _GPU_LOCK:
            for page in image_pages:
                pg = f"p.{page.page_number}/{page.total_pages}"
                logger.info("    %s — captioning %d image(s)…", pg, len(page.images))
                context_text = page.text or page.ocr_text or ""
                page.image_captions = [
                    cap for cap in image_captioner.caption_images_batch(
                        page.images, context=context_text
                    )
                    if cap
                ]
                if page.image_captions:
                    captions_generated = True
                logger.info("    %s — captioning done", pg)

    # ── Step 4: Semantic chunking ────────────────────────────────────────────
    logger.info("  [4/6] Chunking %s…", pdf_path.name)
    docs = chunker.chunk_pages(pages)
    if not docs:
        logger.warning("  No chunks produced for %s", pdf_path.name)
        return 0
    logger.info("  [4/6] %d chunks produced", len(docs))

    # ── GPU model swap cooldown ──────────────────────────────────────────────
    # Only needed if captions were actually generated (qwen2.5vl was used).
    # Skips the 20s wait for PDFs that had images but all were on scanned pages
    # (i.e., they were skipped by the is_scanned guard above).
    if captions_generated:
        with _GPU_LOCK:
            logger.info("  Unloading vision model from VRAM (qwen2.5vl → bge-m3 swap)…")
            _unload_model(config.VISION_MODEL)
            logger.info("  Waiting 20s for GPU memory to clear…")
            time.sleep(20)

    # ── Step 5: Embed (batch API — one HTTP call for all chunks) ─────────────
    logger.info("  [5/6] Embedding %d chunks…", len(docs))
    docs_with_vectors = embedder.embed_documents(docs)
    if not docs_with_vectors:
        logger.warning("  Embedding failed for all chunks in %s", pdf_path.name)
        return 0
    logger.info(
        "  [5/6] Embedding done (%d/%d succeeded)", len(docs_with_vectors), len(docs)
    )

    # ── Step 6: Upsert into Qdrant ───────────────────────────────────────────
    steps_total = "7" if (_WIKI_AVAILABLE and config.WIKI_ENABLED) else "6"
    logger.info("  [6/%s] Upserting into Qdrant…", steps_total)
    vector_store.delete_file_points(str(pdf_path), project_id)
    vector_store.upsert_documents(docs_with_vectors, project_id)

    chunk_count = len(docs_with_vectors)
    logger.info(
        "  ✓ %s: %d pages → %d chunks indexed",
        pdf_path.name, len(pages), chunk_count,
    )

    # ── Step 7: Wiki structured extraction ──────────────────────────────────
    if _WIKI_AVAILABLE and config.WIKI_ENABLED:
        logger.info("  [7/7] Wiki extraction: %s…", pdf_path.name)
        try:
            wiki_store.init_db()
            wiki_extractor.extract_and_store(pdf_path, project_id)
        except Exception as wiki_exc:
            # Wiki extraction failure must NEVER break the main indexing pipeline
            logger.warning("  Wiki extraction failed for %s: %s", pdf_path.name, wiki_exc)

    return chunk_count


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_indexing(
    data_dir: Optional[Path] = None,
    project_filter: Optional[str] = None,
) -> dict:
    """
    Scan the data directory and index all new/changed PDF files.

    Args:
        data_dir:       Root directory containing project subfolders.
                        Defaults to config.DATA_DIR.
        project_filter: If set, only process this specific project folder name.

    Parallelism:
        config.INDEXING_WORKERS controls how many projects are processed
        concurrently.  Default 2.  Increase for text-heavy (no OCR) datasets;
        keep at 1-2 when most PDFs are scanned (GPU bottleneck).
        All GPU operations use _GPU_LOCK so VRAM is never double-booked.
    """
    config.ensure_dirs()
    tracker.init_db()

    data_dir = data_dir or config.DATA_DIR
    if not data_dir.is_dir():
        logger.error("DATA_DIR does not exist: %s", data_dir)
        return {}

    # Discover project folders
    project_folders = sorted([
        f for f in data_dir.iterdir()
        if f.is_dir() and not f.name.startswith(".")
    ])

    if project_filter:
        project_folders = [f for f in project_folders if f.name == project_filter]

    logger.info("Found %d project folder(s) in %s", len(project_folders), data_dir)

    # ── Pre-scan: collect valid PDFs across all projects ─────────────────────
    logger.info("\nPre-scanning all project folders…")
    _prescan: list[tuple[str, Path, list[Path]]] = []
    _prescan_skipped: list[str] = []

    for folder in project_folders:
        project_id = folder.name
        is_valid, reason, valid_pdfs, empty_names = _validate_project_folder(folder)
        if is_valid:
            _prescan.append((project_id, folder, valid_pdfs))
        else:
            _prescan_skipped.append(project_id)

    _all_pdfs_flat   = [(proj, pdf) for proj, _, pdfs in _prescan for pdf in pdfs]
    _total_pdfs      = len(_all_pdfs_flat)
    _already_indexed = sum(1 for _, pdf in _all_pdfs_flat if tracker.is_already_indexed(pdf))
    _to_process      = _total_pdfs - _already_indexed

    workers = max(1, config.INDEXING_WORKERS)
    logger.info("═" * 60)
    logger.info("  INDEXING RUN SUMMARY")
    logger.info("  Parallel workers:  %d", workers)
    logger.info("  Projects found:    %d  (%d skipped, %d valid)",
                len(project_folders), len(_prescan_skipped), len(_prescan))
    logger.info("  Total PDFs:        %d", _total_pdfs)
    logger.info("  Already indexed:   %d  (will skip)", _already_indexed)
    logger.info("  To process:        %d", _to_process)
    logger.info("═" * 60 + "\n")

    # Shared counters (protected by a simple lock for multi-worker safety)
    _stats_lock = threading.Lock()
    stats = {
        "total_projects": len(project_folders),
        "skipped_projects": 0,
        "processed_pdfs": 0,
        "skipped_pdfs": 0,
        "failed_pdfs": 0,
        "total_chunks": 0,
    }
    _pdf_global_idx = [0]   # mutable so inner closure can update it

    def _process_project(project_id: str, folder: Path, valid_pdfs: list[Path]):
        """Process all PDFs in one project folder. Runs in a worker thread."""
        logger.info("\n══ Project: %s ══", project_id)

        _, _, _, empty_names = _validate_project_folder(folder)
        if empty_names:
            logger.warning(
                "  Project '%s' has %d empty/unreadable PDF(s): %s",
                project_id, len(empty_names), ", ".join(empty_names),
            )

        for pdf_path in valid_pdfs:
            with _stats_lock:
                _pdf_global_idx[0] += 1
                global_idx = _pdf_global_idx[0]

            if tracker.is_already_indexed(pdf_path):
                logger.debug(
                    "  [PDF %d/%d] Skipping (unchanged): %s",
                    global_idx, _total_pdfs, pdf_path.name,
                )
                with _stats_lock:
                    stats["skipped_pdfs"] += 1
                continue

            with _stats_lock:
                remaining = _to_process - stats["processed_pdfs"] - stats["failed_pdfs"]

            logger.info(
                "\n  \u250c\u2500 [PDF %d/%d \u2014 %d remaining] %s",
                global_idx, _total_pdfs, remaining, pdf_path.name,
            )

            try:
                chunk_count = _process_pdf(pdf_path, project_id)
                if chunk_count > 0:
                    tracker.mark_indexed(pdf_path, project_id, chunk_count)
                    with _stats_lock:
                        stats["processed_pdfs"] += 1
                        stats["total_chunks"]   += chunk_count
                    logger.info(
                        "  \u2514\u2500 Done [PDF %d/%d] -- %d chunks  (processed %d/%d)",
                        global_idx, _total_pdfs, chunk_count,
                        stats["processed_pdfs"], _to_process,
                    )
                else:
                    tracker.mark_failed(pdf_path, project_id, "No chunks produced")
                    with _stats_lock:
                        stats["failed_pdfs"] += 1
                    logger.warning(
                        "  No chunks [PDF %d/%d]", global_idx, _total_pdfs
                    )
            except Exception as exc:
                logger.exception("  FAILED to process %s: %s", pdf_path.name, exc)
                tracker.mark_failed(pdf_path, project_id, str(exc))
                with _stats_lock:
                    stats["failed_pdfs"] += 1

    # ── tqdm progress bar — wraps the PDF-level counter ──────────────────────
    show_bar = config.SHOW_PROGRESS and _TQDM_AVAILABLE and _to_process > 0
    _pbar = (
        _tqdm(
            total=_to_process,
            unit="pdf",
            desc="Indexing",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} PDFs [{elapsed}<{remaining}, {rate_fmt}]",
        )
        if show_bar else None
    )

    # Patch _process_project to tick the bar after each completed PDF
    _orig_process_project = _process_project

    def _process_project_tracked(project_id, folder, valid_pdfs):
        _orig_process_project(project_id, folder, valid_pdfs)
        # Bar ticks happen per-PDF inside the closure; close here on finish
        pass

    if _pbar is not None:
        # Monkey-patch stats update to also advance the bar
        _orig_mark_done = tracker.mark_indexed

        def _mark_and_tick(path, proj, chunks):
            _orig_mark_done(path, proj, chunks)
            _pbar.update(1)
            _pbar.set_postfix({"last": Path(path).name[:30]}, refresh=False)

        tracker.mark_indexed = _mark_and_tick

        _orig_mark_failed = tracker.mark_failed

        def _fail_and_tick(path, proj, reason):
            _orig_mark_failed(path, proj, reason)
            _pbar.update(1)

        tracker.mark_failed = _fail_and_tick

    # Run project workers
    try:
        if workers == 1 or len(_prescan) == 1:
            for project_id, folder, valid_pdfs in _prescan:
                _process_project(project_id, folder, valid_pdfs)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_project, pid, folder, pdfs): pid
                    for pid, folder, pdfs in _prescan
                }
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error("Worker for project '%s' crashed: %s", pid, exc)
    finally:
        if _pbar is not None:
            _pbar.close()
            # Restore originals
            tracker.mark_indexed = _orig_mark_done
            tracker.mark_failed  = _orig_mark_failed

    # Log skipped projects
    for project_id in _prescan_skipped:
        folder = data_dir / project_id
        is_valid, reason, valid_pdfs, empty_names = _validate_project_folder(folder)
        _log_skipped_project(
            project_id=project_id,
            folder_path=folder,
            reason=reason,
            pdf_count=len(empty_names),
            empty_pdf_names=empty_names,
        )
        stats["skipped_projects"] += 1

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("INDEXING COMPLETE")
    logger.info("  Projects scanned:  %d", stats["total_projects"])
    logger.info("  Projects skipped:  %d  (see %s)",
                stats["skipped_projects"], config.SKIPPED_PROJECTS_CSV)
    logger.info("  PDFs indexed:      %d", stats["processed_pdfs"])
    logger.info("  PDFs unchanged:    %d", stats["skipped_pdfs"])
    logger.info("  PDFs failed:       %d", stats["failed_pdfs"])
    logger.info("  Total chunks:      %d", stats["total_chunks"])
    logger.info("=" * 60)

    return stats

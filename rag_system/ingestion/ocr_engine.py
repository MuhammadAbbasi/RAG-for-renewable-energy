"""
ocr_engine.py - GPU-accelerated OCR for scanned PDF pages.

Engine priority (fastest → slowest):
  1. EasyOCR  - CRAFT + CRNN models, CUDA-accelerated, Italian+English
                ~1-3s per page on GPU vs 30-120s with a VLM
  2. qwen2.5vl - Ollama vision model (fallback when EasyOCR unavailable)
  3. Tesseract - CPU-only last resort

EasyOCR loads its models once at first call and keeps them in VRAM.
Subsequent pages are processed in ~1s each.

Install (run once):
  pip install easyocr numpy      # pulls torch automatically on Python ≤ 3.12
  # For Python 3.13+, install PyTorch manually first, then easyocr:
  #   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  #   pip install easyocr --no-deps
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import httpx
from PIL import Image

from rag_system import config

logger = logging.getLogger(__name__)


# ─── EasyOCR (primary - GPU) ──────────────────────────────────────────────────

_easyocr_reader = None          # Loaded lazily; kept alive for the whole process
_easyocr_attempted = False      # Avoid retrying a failed init on every page


def _get_easyocr_reader():
    """
    Return a cached EasyOCR Reader instance.
    Models are downloaded once to ~/.EasyOCR/ and loaded into GPU VRAM.
    Subsequent calls return the cached reader instantly.
    """
    global _easyocr_reader, _easyocr_attempted

    if _easyocr_attempted:
        return _easyocr_reader   # None if init previously failed

    _easyocr_attempted = True

    try:
        import easyocr
        logger.info("Loading EasyOCR models (Italian + English) on GPU…")
        _easyocr_reader = easyocr.Reader(
            ["it", "en"],
            gpu=True,
            verbose=False,
        )
        logger.info("EasyOCR ready ✓  (GPU-accelerated OCR active)")
    except ImportError:
        logger.warning(
            "EasyOCR not installed - falling back to qwen2.5vl for OCR.\n"
            "  Install: pip install easyocr numpy"
        )
        _easyocr_reader = None
    except Exception as exc:
        logger.warning("EasyOCR init failed (%s) - falling back to qwen2.5vl", exc)
        _easyocr_reader = None

    return _easyocr_reader


def _ocr_with_easyocr(png_bytes: bytes) -> str:
    """
    Run GPU-accelerated OCR using EasyOCR (CRAFT text detection + CRNN recognition).

    Returns extracted text as newline-separated paragraphs, or "" on failure.
    """
    reader = _get_easyocr_reader()
    if reader is None:
        return ""

    try:
        import numpy as np
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        img_array = np.array(img)

        # paragraph=True merges nearby text detections into coherent blocks
        results = reader.readtext(
            img_array,
            detail=0,           # return text strings only, not bounding boxes
            paragraph=True,
        )
        text = "\n".join(r for r in results if r.strip())
        if text:
            logger.debug("EasyOCR: extracted %d chars", len(text))
        return text

    except Exception as exc:
        logger.warning("EasyOCR inference error: %s", exc)
        return ""


# ─── qwen2.5vl via Ollama (secondary fallback) ────────────────────────────────

_OCR_SYSTEM_PROMPT = (
    "Sei un sistema OCR esperto per documenti tecnici italiani. "
    "Estrai TUTTO il testo visibile nell'immagine, mantenendo la struttura originale. "
    "Per le tabelle preserva righe e colonne separando le celle con ' | '. "
    "Non aggiungere commenti - solo il testo estratto."
)

_OCR_USER_PROMPT = (
    "Estrai tutto il testo da questa pagina del documento. "
    "Trascrivi fedelmente ogni parola, numero e simbolo visibile, "
    "mantenendo l'ordine di lettura naturale."
)


def _resize_for_ocr(png_bytes: bytes, max_dim: int = 1600) -> bytes:
    """Resize image so neither side exceeds max_dim. Returns PNG bytes."""
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        if w > max_dim or h > max_dim:
            ratio = min(max_dim / w, max_dim / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        logger.warning("OCR image resize failed: %s", exc)
        return png_bytes


def _ocr_with_vision_model(png_bytes: bytes) -> str:
    """
    Fallback: call qwen2.5vl via Ollama native API to extract text from a page image.
    Slower (~30-120s/page) but handles complex layouts and handwriting better.
    """
    resized   = _resize_for_ocr(png_bytes)
    b64_image = base64.b64encode(resized).decode("utf-8")

    payload = {
        "model": config.VISION_MODEL,
        "messages": [
            {"role": "system", "content": _OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _OCR_USER_PROMPT,
                "images": [b64_image],
            },
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 2048},
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "").strip()
            if text:
                logger.debug("qwen2.5vl OCR: extracted %d chars", len(text))
            return text
    except httpx.TimeoutException:
        logger.warning("qwen2.5vl OCR timed out (image size: %d bytes)", len(png_bytes))
        return ""
    except Exception as exc:
        logger.error("qwen2.5vl OCR failed: %s", exc)
        return ""


# ─── Tesseract (tertiary fallback - CPU) ──────────────────────────────────────

def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_with_tesseract(png_bytes: bytes) -> str:
    """CPU fallback: pytesseract with Italian language pack."""
    try:
        import pytesseract
        img  = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang="ita", config="--psm 6")
        return text.strip()
    except Exception as exc:
        logger.error("Tesseract OCR failed: %s", exc)
        return ""


# ─── Public API ───────────────────────────────────────────────────────────────

def ocr_engine_available() -> bool:
    return True


def ocr_page_bytes(png_bytes: bytes) -> str:
    """
    Run OCR on a rendered PDF page image.

    Engine priority:
      1. EasyOCR  (GPU CUDA - CRAFT+CRNN, ~1-3s/page)  ← primary
      2. qwen2.5vl via Ollama  (GPU VLM, ~30-120s/page)  ← if EasyOCR not installed
      3. Tesseract (CPU)                                  ← last resort

    Args:
        png_bytes: Raw PNG bytes from pdf_parser.render_page_as_image()

    Returns:
        Extracted text. Empty string if page is blank or all OCR fails.
    """
    if not png_bytes:
        return ""

    # ── 1. EasyOCR (primary - GPU, fast) ─────────────────────────────────────
    text = _ocr_with_easyocr(png_bytes)
    if text:
        return text

    # ── 2. qwen2.5vl fallback ────────────────────────────────────────────────
    # Skip qwen2.5vl for very small images (< 50 KB).
    # In practice these are blank, encrypted, or corrupted pages - qwen2.5vl
    # always times out on them (seen: 32028-byte pages in ANAS_PA_AUTOST_n_027770_21).
    _MIN_VLM_SIZE = 50 * 1024   # 50 KB
    if len(png_bytes) < _MIN_VLM_SIZE:
        logger.info(
            "Skipping qwen2.5vl OCR - image too small (%d bytes < %d KB threshold), "
            "likely blank or encrypted page",
            len(png_bytes), _MIN_VLM_SIZE // 1024,
        )
    else:
        logger.info("EasyOCR returned empty - trying qwen2.5vl…")
        text = _ocr_with_vision_model(png_bytes)
        if text:
            return text

    # ── 3. Tesseract last resort ─────────────────────────────────────────────
    if _tesseract_available():
        logger.info("qwen2.5vl returned empty - falling back to Tesseract")
        return _ocr_with_tesseract(png_bytes)

    return ""


def ocr_image_bytes(img_bytes: bytes) -> str:
    """OCR on an arbitrary embedded image (e.g. a figure containing text labels)."""
    return ocr_page_bytes(img_bytes)

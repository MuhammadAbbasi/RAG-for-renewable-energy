"""
image_captioner.py - Deep image understanding via qwen2.5vl (Ollama).

This module sends each extracted image to the locally running qwen2.5vl
vision-language model and receives a detailed Italian-language caption.

The caption is designed to be highly descriptive - not just "a map" but
a full contextual description of what the map shows, including:
  - Geographic area, roads, boundaries
  - Legend items and their meaning
  - Any text visible in the image
  - Charts: axes, values, trends
  - Photographs: subjects, context, technical details

The captions are stored as regular text chunks in Qdrant, making images
fully searchable through the same embedding pipeline.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

import httpx
from PIL import Image

from rag_system import config

logger = logging.getLogger(__name__)

# ── Circuit breaker ──────────────────────────────────────────────────────────
# If qwen2.5vl returns 500 on 3 consecutive full-retry cycles, we assume the
# model is not loaded and skip all further captioning for this process run.
# This prevents burning 75 s × N images when the vision model is unavailable.
_consecutive_failures = 0
_CIRCUIT_BREAKER_THRESHOLD = 1
_circuit_open = False   # True = captioning disabled for this run


def _reset_circuit():
    global _consecutive_failures, _circuit_open
    _consecutive_failures = 0
    _circuit_open = False


# Prompt instructing the model to produce rich, structured descriptions in Italian
_CAPTION_SYSTEM_PROMPT = """Sei un esperto analista di documenti tecnici italiani.
Analizza questa immagine estratta da un documento PDF tecnico/ingegneristico/ambientale.

Fornisci una descrizione DETTAGLIATA e COMPLETA in italiano che include:
1. Tipo di immagine (mappa, grafico, fotografia, schema, planimetria, tavola tecnica, etc.)
2. Contenuto principale: cosa rappresenta esattamente
3. Per MAPPE: area geografica, comuni, strade, confini, elementi territoriali, legenda completa
4. Per GRAFICI: titolo, assi, valori, trend, unità di misura
5. Per TABELLE visive: intestazioni, dati principali, totali
6. Per FOTOGRAFIE: soggetto, contesto, elementi tecnici visibili
7. Qualsiasi testo visibile nell'immagine (titoli, etichette, note, numeri)
8. Colori significativi e il loro significato (es. legenda colori)

Sii preciso e tecnico. Questa descrizione sarà usata per recuperare informazioni dal documento.
Non omettere dettagli importanti."""

_CAPTION_USER_PROMPT = """Descrivi questa immagine in modo completo e dettagliato in italiano.
Includi tutto il testo visibile, i dettagli della legenda e il contesto tecnico."""


def _resize_image(img_bytes: bytes, max_dim: int = None) -> bytes:
    """
    Resize image so neither dimension exceeds max_dim (preserving aspect ratio).
    Returns PNG bytes.
    """
    max_dim = max_dim or config.IMAGE_MAX_DIM
    try:
        import PIL
        PIL.Image.MAX_IMAGE_PIXELS = 200_000_000  # allow up to 200 MP before skipping
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        if w * h > 200_000_000:
            logger.warning("Image too large (%dx%d = %dMP) - skipping caption", w, h, w*h//1_000_000)
            return img_bytes  # return original unchanged
        img = img.convert("RGB")
        w, h = img.size
        if w > max_dim or h > max_dim:
            ratio = min(max_dim / w, max_dim / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Image resize failed: %s", exc)
        return img_bytes


def caption_image(img_bytes: bytes, context: str = "") -> Optional[str]:
    """
    Send an image to qwen2.5vl via Ollama and return a detailed Italian caption.

    Args:
        img_bytes: Raw image bytes (any PIL-supported format).
        context:   Optional surrounding text context (e.g., nearby paragraph)
                   to help the model understand the image better.

    Returns:
        Detailed Italian description string, or None on failure.
    """
    global _consecutive_failures, _circuit_open
    if not img_bytes:
        return None

    # Resize to avoid sending huge payloads to Ollama
    resized = _resize_image(img_bytes)
    b64_img = base64.b64encode(resized).decode("utf-8")

    # Build user prompt - include context if provided
    user_prompt = _CAPTION_USER_PROMPT
    if context:
        user_prompt = (
            f"Contesto dal documento: {context[:500]}\n\n"
            + user_prompt
        )

    # Ollama native API: images go in the 'images' list as raw base64 (no data URI prefix)
    payload = {
        "model": config.VISION_MODEL,
        "messages": [
            {"role": "system", "content": _CAPTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
                "images": [b64_img],        # ← Ollama native format (not OpenAI image_url)
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }

    # Circuit breaker - bail immediately if vision model is known-unavailable
    if _circuit_open:
        return None

    # Retry with backoff - Ollama may return 500 under VRAM pressure
    max_retries = 4
    wait        = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload)

            if resp.status_code == 500:
                logger.warning(
                    "Ollama 500 on caption (attempt %d/%d) - waiting %.0fs…",
                    attempt, max_retries, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 60.0)
                continue

            resp.raise_for_status()
            caption = resp.json().get("message", {}).get("content", "").strip()
            if caption:
                logger.debug("Caption generated (%d chars)", len(caption))
            # Success - reset circuit breaker counter
            _consecutive_failures = 0
            return caption if caption else None

        except httpx.TimeoutException:
            logger.warning(
                "qwen2.5vl caption timeout (attempt %d/%d) - waiting %.0fs…",
                attempt, max_retries, wait,
            )
            time.sleep(wait)
            wait = min(wait * 2, 60.0)

        except Exception as exc:
            logger.error("Image captioning failed (attempt %d/%d): %s", attempt, max_retries, exc)
            time.sleep(wait)
            wait = min(wait * 2, 60.0)

    # All retries exhausted - update circuit breaker
    _consecutive_failures += 1
    if _consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
        _circuit_open = True
        logger.error(
            "Vision model returned 500 on %d consecutive attempts - "
            "disabling image captioning for this run. "
            "Pull the model with: ollama pull %s",
            _consecutive_failures, config.VISION_MODEL,
        )
    return None


def caption_images_batch(
    images: list[dict],
    context: str = "",
) -> list[Optional[str]]:
    """
    Caption a list of image dicts (as returned by pdf_parser).

    Args:
        images:  List of dicts with "bytes" key.
        context: Surrounding text context (same for all images on this page).

    Returns:
        List of caption strings (parallel to input list). None where captioning failed.
    """
    captions = []
    for i, img in enumerate(images):
        logger.info("  Captioning image %d/%d…", i + 1, len(images))
        cap = caption_image(img.get("bytes", b""), context=context)
        captions.append(cap)
    return captions

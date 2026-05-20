"""
embedder.py - Text embedding via bge-m3 (Ollama).

bge-m3 is a state-of-the-art multilingual embedding model that produces
1024-dimensional dense vectors. It handles Italian text excellently and
is already running in your local Ollama instance.

Speed strategy:
  1. Batch API  - POST /api/embed with {"input": [list_of_texts]}
                  Sends all chunks in ONE HTTP round-trip (~10× faster than
                  per-chunk calls for a typical 200-chunk PDF).
                  Requires Ollama ≥ 0.1.30. Detected automatically at runtime.

  2. Per-chunk  - Legacy POST /api/embeddings, used as fallback if the batch
                  endpoint returns 404.

Retry-with-backoff handles transient Ollama 500 errors (GPU model swaps).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from rag_system import config
from rag_system.indexing.priority import QUERY_GATE

logger = logging.getLogger(__name__)

_EMBED_URL_BATCH  = f"{config.OLLAMA_BASE_URL}/api/embed"      # Ollama ≥ 0.1.30, batch
_EMBED_URL_SINGLE = f"{config.OLLAMA_BASE_URL}/api/embeddings"  # legacy, per-text

# How many texts to include in one batch request (keep reasonable for RAM)
_BATCH_SIZE = 12  # small batches so QUERY_GATE can pause indexing quickly when a chat query arrives

# Retry settings for transient Ollama errors (500 / model-swap VRAM pressure)
_MAX_RETRIES    = 8
_RETRY_BASE_SEC = 8.0   # 8 → 16 → 32 → 64 → 120s (capped)

# Runtime feature flag - set to False after first 404 on /api/embed
_batch_api_supported: Optional[bool] = None


# ─── Batch embedding (Ollama ≥ 0.1.30) ───────────────────────────────────────

def _embed_batch(texts: list[str]) -> Optional[list[list[float]]]:
    """
    Embed multiple texts in a single Ollama API call via /api/embed.
    Returns a list of vectors (same order as input), or None on failure.
    Requires Ollama ≥ 0.1.30.
    """
    global _batch_api_supported
    if _batch_api_supported is False:
        return None

    payload = {"model": config.EMBED_MODEL, "input": texts}
    wait    = _RETRY_BASE_SEC

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(_EMBED_URL_BATCH, json=payload)

            if resp.status_code == 404:
                logger.info(
                    "Batch embed endpoint not available (Ollama < 0.1.30) - "
                    "falling back to per-chunk mode"
                )
                _batch_api_supported = False
                return None

            if resp.status_code == 500:
                logger.warning(
                    "Ollama 500 on batch embed (attempt %d/%d) - waiting %.0fs…",
                    attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 120.0)
                continue

            resp.raise_for_status()
            data = resp.json()
            # Response shape: {"embeddings": [[...], [...]]}
            vectors = data.get("embeddings") or data.get("embedding")
            if vectors and isinstance(vectors[0], list):
                _batch_api_supported = True
                logger.debug(
                    "Batch embed: %d texts → %d vectors (dim=%d)",
                    len(texts), len(vectors), len(vectors[0]),
                )
                return vectors
            # Unexpected shape - fall back
            logger.warning("Unexpected batch embed response shape: %s", list(data.keys()))
            _batch_api_supported = False
            return None

        except httpx.TimeoutException:
            logger.warning(
                "Batch embed timeout (attempt %d/%d, %d texts) - waiting %.0fs…",
                attempt, _MAX_RETRIES, len(texts), wait,
            )
            time.sleep(wait)
            wait = min(wait * 2, 120.0)

        except httpx.HTTPStatusError as exc:
            # 4xx/5xx from Ollama - log and retry
            logger.warning(
                "Ollama HTTP %d on batch embed (attempt %d/%d) - waiting %.0fs…",
                exc.response.status_code, attempt, _MAX_RETRIES, wait,
            )
            time.sleep(wait)
            wait = min(wait * 2, 120.0)

        except Exception as exc:
            logger.error("Batch embed error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            time.sleep(wait)
            wait = min(wait * 2, 120.0)

    logger.error("Batch embedding failed after %d retries", _MAX_RETRIES)
    return None


# ─── Per-chunk fallback (legacy /api/embeddings) ─────────────────────────────

def _embed_single_with_retry(text: str) -> Optional[list[float]]:
    """Single embed call with exponential backoff on 500 errors."""
    payload = {"model": config.EMBED_MODEL, "prompt": text.strip()}
    wait    = _RETRY_BASE_SEC

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(_EMBED_URL_SINGLE, json=payload)

            if resp.status_code == 500:
                logger.warning(
                    "Ollama 500 on embed (attempt %d/%d) -- waiting %.0fs for model swap...",
                    attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 120.0)
                continue

            resp.raise_for_status()
            return resp.json().get("embedding")

        except httpx.TimeoutException:
            logger.warning(
                "Embed timeout (attempt %d/%d) -- waiting %.0fs...",
                attempt, _MAX_RETRIES, wait,
            )
            time.sleep(wait)
            wait = min(wait * 2, 120.0)

        except httpx.HTTPStatusError as exc:
            # 4xx/5xx from Ollama - log and retry (model may be reloading)
            logger.warning(
                "Ollama HTTP %d on embed (attempt %d/%d) - waiting %.0fs…",
                exc.response.status_code, attempt, _MAX_RETRIES, wait,
            )
            time.sleep(wait)
            wait = min(wait * 2, 120.0)

        except Exception as exc:
            logger.error("Embed error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
            time.sleep(wait)
            wait = min(wait * 2, 120.0)

    logger.error(
        "Embedding failed after %d retries for text: '%s...'", _MAX_RETRIES, text[:60]
    )
    return None


# ─── Fallback model helpers ───────────────────────────────────────────────────

def _embed_single_fallback(text: str) -> Optional[list[float]]:
    """
    Single embed using the fallback model (nomic-embed-text, 768-dim).
    Used when the primary bge-m3 model fails after all retries.
    """
    if not config.EMBED_MODEL_FALLBACK:
        return None
    payload = {"model": config.EMBED_MODEL_FALLBACK, "prompt": text.strip()}
    wait = _RETRY_BASE_SEC
    for attempt in range(1, 4):          # fewer retries for fallback
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(_EMBED_URL_SINGLE, json=payload)
            if resp.status_code == 500:
                logger.warning(
                    "Ollama 500 on fallback embed (attempt %d/3) - waiting %.0fs…",
                    attempt, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 60.0)
                continue
            resp.raise_for_status()
            vec = resp.json().get("embedding")
            if vec:
                logger.debug(
                    "Fallback embed OK: %s → %d-dim (primary bge-m3 had failed)",
                    config.EMBED_MODEL_FALLBACK, len(vec),
                )
            return vec
        except Exception as exc:
            logger.warning("Fallback embed error (attempt %d/3): %s", attempt, exc)
            time.sleep(wait)
            wait = min(wait * 2, 60.0)
    logger.error("Fallback embedding also failed for: '%s...'", text[:60])
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def embed_text(text: str):
    """
    Embed a single text string.
    Tries bge-m3 first (1024-dim); falls back to nomic-embed-text (768-dim).
    Returns a vector or None.
    """
    if not text or not text.strip():
        return None
    vec = _embed_single_with_retry(text)
    if vec is None and config.EMBED_MODEL_FALLBACK:
        logger.warning("Primary embed failed - trying fallback model")
        vec = _embed_single_fallback(text)
    return vec



def embed_texts(texts):
    """
    Embed a list of texts as efficiently as possible.

    Strategy:
      1. Batch API (/api/embed, Ollama >= 0.1.30) -- all chunks in one round-trip
      2. Per-chunk (/api/embeddings) -- if batch not supported
      3. Fallback model (nomic-embed-text) -- for any text that still returns None
         NOTE: fallback vectors are 768-dim; if mixed with 1024-dim primary vectors
         in the same Qdrant collection this will cause dimension mismatches.
         The fallback is therefore only used when the PRIMARY model fails entirely
         (e.g. GPU OOM), not selectively per-chunk.
    """
    if not texts:
        return []

    results = [None] * len(texts)

    # Batch path
    batch_success = True
    batch_results = []

    for i in range(0, len(texts), _BATCH_SIZE):
        # ── Query-priority gate ───────────────────────────────────────────────
        # If a user query is in progress, pause here until it finishes.
        # This lets Ollama serve the LLM without fighting the embedding model.
        if not QUERY_GATE.is_set():
            logger.info(
                "Embedding batch %d paused - waiting for active query to finish…",
                i // _BATCH_SIZE,
            )
            QUERY_GATE.wait(timeout=300)   # wait up to 5 min, then proceed anyway
        # ─────────────────────────────────────────────────────────────────────

        batch = [t.strip() for t in texts[i: i + _BATCH_SIZE] if t and t.strip()]
        if not batch:
            batch_results.extend([None] * (min(i + _BATCH_SIZE, len(texts)) - i))
            continue
        vecs = _embed_batch(batch)
        if vecs is None:
            batch_success = False
            break
        batch_results.extend(vecs)

    if batch_success and len(batch_results) == len(texts):
        logger.info(
            "Batch embed complete: %d texts -> %d vectors",
            len(texts), sum(1 for v in batch_results if v),
        )
        return batch_results

    # Per-chunk path
    logger.info("Using per-chunk embedding for %d texts...", len(texts))
    primary_failed = 0
    for i, text in enumerate(texts):
        # Query-priority gate (per-chunk fallback path)
        if i % 8 == 0 and not QUERY_GATE.is_set():
            QUERY_GATE.wait(timeout=300)

        if not text or not text.strip():
            results[i] = None
            continue
        results[i] = _embed_single_with_retry(text)
        if results[i] is None:
            primary_failed += 1
        if i > 0 and i % 32 == 0:
            time.sleep(0.1)

    # Fallback: if ALL primary calls failed, retry with nomic-embed-text
    if primary_failed > 0 and primary_failed == len([t for t in texts if t and t.strip()]):
        logger.warning(
            "All %d primary embeds failed -- switching to fallback model %s",
            primary_failed, config.EMBED_MODEL_FALLBACK,
        )
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            results[i] = _embed_single_fallback(text)

    return results


def embed_documents(docs):
    """
    Embed a list of LangChain Documents.
    Returns list of (document, vector) tuples where embedding succeeded.
    """
    texts   = [doc.page_content for doc in docs]
    vectors = embed_texts(texts)
    paired  = []
    skipped = 0

    for doc, vec in zip(docs, vectors):
        if vec is not None:
            paired.append((doc, vec))
        else:
            skipped += 1
            logger.warning(
                "Skipping chunk (embed failed): %s p.%s",
                doc.metadata.get("filename", "?"),
                doc.metadata.get("page_number", "?"),
            )

    if skipped:
        logger.warning(
            "Skipped %d/%d chunks due to embedding failures", skipped, len(docs)
        )

    return paired


def test_connection():
    """Check that the embedding model is reachable and returns a valid vector."""
    try:
        vec = embed_text("test connessione modello embedding")
        if vec and len(vec) > 0:
            logger.info(
                "Embedding model OK: %s -> %d-dim vector",
                config.EMBED_MODEL, len(vec),
            )
            return True
        logger.error("Embedding returned empty vector.")
        return False
    except Exception as exc:
        logger.error("Cannot reach embedding model: %s", exc)
        return False

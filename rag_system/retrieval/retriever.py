"""
retriever.py - Hybrid retrieval: dense (Qdrant cosine) + BM25 keyword re-rank.

Strategy
--------
1. Dense pass  - embed the query with bge-m3; retrieve top_k * BM25_OVERRETRIEVE
   candidates from Qdrant (wider net so BM25 has enough material to re-rank).
2. BM25 pass   - build a BM25Okapi index over the candidate texts and score each
   against the raw query tokens; normalize to [0,1].
3. Merge        - final_score = (1-BM25_WEIGHT)*dense_score + BM25_WEIGHT*bm25_score
   Sort descending, return top_k.

BM25 disables automatically when rank_bm25 is not installed (pure dense fallback).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from rag_system import config
from rag_system.indexing import embedder, vector_store

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi as _BM25
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.info("rank_bm25 not installed - BM25 hybrid disabled (pure dense retrieval)")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _bm25_rerank(query: str, candidates: list[dict], weight: float) -> list[dict]:
    if not candidates or not _BM25_AVAILABLE or weight == 0:
        return candidates
    corpus = [_tokenize(c.get("text", "")) for c in candidates]
    try:
        bm25       = _BM25(corpus)
        raw_scores = list(bm25.get_scores(_tokenize(query)))  # convert numpy → plain list
    except Exception as exc:
        logger.warning("BM25 scoring failed (%s) - using dense scores only", exc)
        return candidates

    # If BM25 found zero keyword overlap, skip reranking entirely - use pure dense scores.
    # Without this guard, combined = 0.7*dense + 0.3*0.0 = 0.7*dense, effectively
    # raising the score_threshold by ~43% and silently dropping valid chunks.
    if not raw_scores or max(raw_scores) == 0:
        return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

    max_s = max(raw_scores)
    norm  = [s / max_s for s in raw_scores]

    for i, cand in enumerate(candidates):
        dense        = cand.get("score", 0.0)
        cand["score"]       = round((1 - weight) * dense + weight * norm[i], 4)
        cand["dense_score"] = round(dense, 4)
        cand["bm25_score"]  = round(norm[i], 4)

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def retrieve(
    query: str,
    project_id: Optional[str] = None,
    top_k: int = None,
    score_threshold: float = None,
    content_types: Optional[list[str]] = None,
    use_bm25: Optional[bool] = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query using hybrid search.

    Args:
        query:           Natural-language question (Italian or other).
        project_id:      Restrict to one project; None = search all.
        top_k:           Results to return (default config.RETRIEVAL_TOP_K).
        score_threshold: Minimum combined score (default config.RETRIEVAL_SCORE_THRESHOLD).
        content_types:   Filter e.g. ["text", "table"].
        use_bm25:        Override config.BM25_ENABLED for this call.

    Returns:
        List of {text, score, dense_score?, bm25_score?, metadata} dicts.
    """
    if not query or not query.strip():
        return []

    top_k           = top_k or config.RETRIEVAL_TOP_K
    score_threshold = score_threshold if score_threshold is not None \
                      else config.RETRIEVAL_SCORE_THRESHOLD
    bm25_on = (use_bm25 if use_bm25 is not None else config.BM25_ENABLED) and _BM25_AVAILABLE

    dense_k = top_k * config.BM25_OVERRETRIEVE if bm25_on else top_k

    try:
        query_vector = embedder.embed_text(query.strip())
    except Exception as exc:
        logger.error("Unexpected error embedding query '%s': %s", query[:80], exc)
        return []

    if query_vector is None:
        logger.error("Failed to embed query: '%s'", query[:80])
        return []

    try:
        candidates = vector_store.search(
            query_vector    = query_vector,
            project_id      = project_id,
            top_k           = dense_k,
            score_threshold = score_threshold * 0.5 if bm25_on else score_threshold,
            content_types   = content_types,
        )
    except Exception as exc:
        logger.error("Qdrant search failed for query '%s': %s", query[:80], exc)
        return []

    if not candidates:
        logger.info("Qdrant returned 0 candidates for '%s...' (project=%s, threshold=%.3f)",
                    query[:50], project_id or "ALL", score_threshold * 0.5 if bm25_on else score_threshold)
        return []

    if bm25_on and len(candidates) > 1:
        candidates = _bm25_rerank(query, candidates, weight=config.BM25_WEIGHT)

    results = [c for c in candidates if c.get("score", 0) >= score_threshold][:top_k]

    logger.info(
        "Retrieved %d chunks for '%s...' (project=%s, bm25=%s)",
        len(results), query[:50], project_id or "ALL", bm25_on,
    )
    return results


def format_context(results: list[dict], max_chars: int = None) -> str:
    """Format retrieved chunks as context for the LLM prompt."""
    if max_chars is None:
        max_chars = config.FORMAT_CONTEXT_MAX_CHARS
    if not results:
        return "Nessun documento rilevante trovato nel database."

    parts = []
    total = 0
    type_labels = {
        "text": "Testo", "ocr": "Testo OCR",
        "table": "Tabella", "image_caption": "Descrizione immagine",
    }

    for i, res in enumerate(results, 1):
        meta       = res.get("metadata", {})
        chunk_text = res.get("text", "").strip()
        if not chunk_text:
            continue
        label  = type_labels.get(meta.get("content_type", ""), "Testo")
        source = meta.get("source", "?")
        score  = res.get("score", 0.0)
        block  = f"[{i}] {label} | {source} | Score: {score:.3f}\n{chunk_text}"

        if len(block) <= max_chars - total:
            parts.append(block)
            total += len(block)
        else:
            break

    return "\n\n---\n\n".join(parts)


def get_sources(results: list[dict]) -> list[dict]:
    """Extract deduplicated source citations from retrieval results."""
    seen, sources = set(), []
    for res in results:
        meta = res.get("metadata", {})
        key  = (meta.get("project_id", ""), meta.get("filename", ""), meta.get("page_number", 0))
        if key not in seen:
            seen.add(key)
            sources.append({
                "project":  meta.get("project_id", ""),
                "filename": meta.get("filename", ""),
                "page":     meta.get("page_number", ""),
                "source":   meta.get("source", ""),
                "score":    round(res.get("score", 0.0), 3),
                "type":     meta.get("content_type", ""),
            })
    return sources

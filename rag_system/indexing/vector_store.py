"""
vector_store.py - Qdrant client wrapper for storing and searching chunks.

Design:
  - One Qdrant collection per project (e.g. "rag_project_14413_sicilia").
  - Each point stores: the embedding vector + all metadata as payload.
  - Cosine similarity is used for retrieval.
  - Collections are created automatically on first use.
  - Supports both project-scoped and cross-project (global) search.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from rag_system import config

logger = logging.getLogger(__name__)

# Singleton client - shared across the process
_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Return (or create) the shared Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            timeout=60,
        )
        logger.info("Qdrant client connected to %s:%d", config.QDRANT_HOST, config.QDRANT_PORT)
    return _client


def ensure_collection(collection_name: str):
    """Create the Qdrant collection if it doesn't already exist."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.EMBED_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection: %s", collection_name)
    return collection_name


def collection_for_project(project_id: str) -> str:
    """Return the Qdrant collection name for a given project ID."""
    return config.sanitize_collection_name(project_id)


def upsert_documents(docs_with_vectors: list[tuple[Document, list[float]]], project_id: str):
    """
    Upsert (insert or update) a batch of documents into the project's collection.

    Args:
        docs_with_vectors: List of (Document, embedding_vector) tuples.
        project_id:        The project this batch belongs to.
    """
    if not docs_with_vectors:
        return

    collection_name = collection_for_project(project_id)
    ensure_collection(collection_name)
    client = get_client()

    points = []
    for doc, vector in docs_with_vectors:
        # Use a deterministic ID based on content so re-indexing the same chunk
        # is idempotent (upsert replaces the existing point).
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{doc.metadata.get('pdf_path', '')}:{doc.metadata.get('page_number', 0)}:{doc.metadata.get('chunk_index', 0)}:{doc.metadata.get('content_type', '')}"
        ))

        payload = {**doc.metadata, "text": doc.page_content}
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    # Upload in batches of 100 to avoid large request payloads
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i: i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    logger.info(
        "Upserted %d points into collection '%s'",
        len(points), collection_name
    )


def delete_file_points(pdf_path: str, project_id: str):
    """
    Remove all points belonging to a specific PDF file from its collection.
    Useful when a file is re-indexed after modification.
    """
    collection_name = collection_for_project(project_id)
    client = get_client()

    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="pdf_path", match=MatchValue(value=pdf_path))]
            ),
        )
        logger.info("Deleted old points for %s from %s", pdf_path, collection_name)
    except Exception as exc:
        logger.warning("Could not delete old points for %s: %s", pdf_path, exc)


def search(
    query_vector: list[float],
    project_id: Optional[str] = None,
    top_k: int = None,
    score_threshold: float = None,
    content_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Search for similar chunks.

    Args:
        query_vector:    Embedding of the user's query.
        project_id:      If set, search only this project's collection.
                         If None, search ALL project collections and merge results.
        top_k:           Number of results to return.
        score_threshold: Minimum similarity score to include.
        content_types:   Optional filter e.g. ["text", "table"] to restrict content types.

    Returns:
        List of result dicts with keys: text, score, metadata.
    """
    top_k           = top_k or config.RETRIEVAL_TOP_K
    score_threshold = score_threshold if score_threshold is not None else config.RETRIEVAL_SCORE_THRESHOLD
    client          = get_client()

    # Build content-type filter if requested
    query_filter = None
    if content_types:
        from qdrant_client.http.models import Filter, FieldCondition, MatchAny
        query_filter = Filter(
            must=[FieldCondition(key="content_type", match=MatchAny(any=content_types))]
        )

    # Determine which collections to search
    if project_id:
        collections = [collection_for_project(project_id)]
    else:
        try:
            all_collections = client.get_collections().collections
            collections = [
                c.name for c in all_collections
                if c.name.startswith(config.QDRANT_COLLECTION_PREFIX)
            ]
        except Exception as exc:
            logger.error("Qdrant get_collections() failed: %s", exc)
            return []

    if not collections:
        logger.warning("No Qdrant collections found with prefix '%s'", config.QDRANT_COLLECTION_PREFIX)
        return []

    all_results = []
    for coll in collections:
        try:
            # qdrant-client ≥ 2.x removed client.search() - use query_points instead.
            # query_points returns a QueryResponse with a .points list of ScoredPoint.
            response = client.query_points(
                collection_name = coll,
                query           = query_vector,
                limit           = top_k,
                score_threshold = score_threshold,
                query_filter    = query_filter,
                with_payload    = True,
            )
            for hit in response.points:
                all_results.append({
                    "text":     hit.payload.get("text", ""),
                    "score":    hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                })
        except Exception as exc:
            logger.warning("Search failed in collection %s: %s", coll, exc)

    # Sort merged results by score descending and cap at top_k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def list_projects() -> list[str]:
    """Return all project IDs that have at least one indexed collection."""
    client = get_client()
    prefix = config.QDRANT_COLLECTION_PREFIX
    names = [c.name for c in client.get_collections().collections if c.name.startswith(prefix)]
    # Strip prefix to recover project-like identifiers
    return [n[len(prefix):] for n in names]


def collection_info(project_id: str) -> dict:
    """Return point count and status for a project's collection."""
    client = get_client()
    coll_name = collection_for_project(project_id)
    try:
        info = client.get_collection(coll_name)
        return {
            "collection": coll_name,
            "points":     info.points_count,
            "status":     info.status,
        }
    except Exception:
        return {"collection": coll_name, "points": 0, "status": "not_found"}

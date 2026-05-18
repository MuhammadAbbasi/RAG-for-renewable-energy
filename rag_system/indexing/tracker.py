"""
tracker.py — SQLite-based file tracker for incremental indexing.

Tracks which PDF files have been indexed, using MD5 content hashes to detect
changes. Only new or modified files are processed on subsequent runs.

Tables:
  processed_files  — one row per indexed PDF
  skipped_projects — project folders skipped due to being empty/unreadable
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from rag_system import config

logger = logging.getLogger(__name__)

DB_PATH = config.DB_DIR / "tracker.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    config.ensure_dirs()
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS processed_files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path   TEXT    NOT NULL UNIQUE,
                file_hash   TEXT    NOT NULL,
                project_id  TEXT    NOT NULL,
                indexed_at  TEXT    NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                status      TEXT    DEFAULT 'ok',
                error       TEXT    DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS skipped_projects (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id   TEXT    NOT NULL,
                folder_path  TEXT    NOT NULL,
                reason       TEXT    NOT NULL,
                pdf_count    INTEGER DEFAULT 0,
                detected_at  TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_processed_path
                ON processed_files(file_path);
            CREATE INDEX IF NOT EXISTS idx_processed_project
                ON processed_files(project_id);
        """)
        # Migration: add error column if upgrading from older schema
        try:
            conn.execute("ALTER TABLE processed_files ADD COLUMN error TEXT DEFAULT ''")
            conn.commit()
        except Exception:
            pass  # Column already exists
    logger.debug("Tracker DB initialised at %s", DB_PATH)


def compute_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file's contents."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_already_indexed(file_path: Path) -> bool:
    """
    Return True if this exact file (same path AND same content hash) has
    already been indexed. Returns False if new or changed.
    """
    if config.FORCE_REINDEX:
        return False

    path_str = str(file_path)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT file_hash FROM processed_files WHERE file_path = ? AND status = 'ok'",
            (path_str,)
        ).fetchone()

    if row is None:
        return False  # Never seen, or previously failed — allow retry

    # Re-hash to detect content changes
    current_hash = compute_hash(file_path)
    return row["file_hash"] == current_hash


def mark_indexed(file_path: Path, project_id: str, chunk_count: int):
    """Record a successfully indexed file."""
    path_str     = str(file_path)
    file_hash    = compute_hash(file_path)
    indexed_at   = datetime.utcnow().isoformat()

    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO processed_files (file_path, file_hash, project_id, indexed_at, chunk_count, status)
            VALUES (?, ?, ?, ?, ?, 'ok')
            ON CONFLICT(file_path) DO UPDATE SET
                file_hash   = excluded.file_hash,
                indexed_at  = excluded.indexed_at,
                chunk_count = excluded.chunk_count,
                status      = 'ok'
        """, (path_str, file_hash, project_id, indexed_at, chunk_count))


def mark_failed(file_path: Path, project_id: str, reason: str):
    """Record a file that failed to index."""
    path_str   = str(file_path)
    file_hash  = compute_hash(file_path) if file_path.exists() else ""
    indexed_at = datetime.utcnow().isoformat()

    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO processed_files (file_path, file_hash, project_id, indexed_at, chunk_count, status, error)
            VALUES (?, ?, ?, ?, 0, 'failed', ?)
            ON CONFLICT(file_path) DO UPDATE SET
                file_hash  = excluded.file_hash,
                indexed_at = excluded.indexed_at,
                status     = 'failed',
                error      = excluded.error
        """, (path_str, file_hash, project_id, indexed_at, str(reason)[:500]))
    logger.warning("Marked as failed: %s — %s", file_path.name, reason)


def record_skipped_project(project_id: str, folder_path: Path, reason: str, pdf_count: int = 0):
    """Record a project folder that was skipped (empty, unreadable, etc.)."""
    detected_at = datetime.utcnow().isoformat()
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO skipped_projects (project_id, folder_path, reason, pdf_count, detected_at)
            VALUES (?, ?, ?, ?, ?)
        """, (project_id, str(folder_path), reason, pdf_count, detected_at))
    logger.warning("Skipped project '%s': %s", project_id, reason)


def get_stats() -> dict:
    """Return summary statistics from the tracker DB."""
    with _get_conn() as conn:
        total      = conn.execute("SELECT COUNT(*) FROM processed_files WHERE status='ok'").fetchone()[0]
        failed     = conn.execute("SELECT COUNT(*) FROM processed_files WHERE status='failed'").fetchone()[0]
        skipped    = conn.execute("SELECT COUNT(*) FROM skipped_projects").fetchone()[0]
        projects   = conn.execute("SELECT COUNT(DISTINCT project_id) FROM processed_files WHERE status='ok'").fetchone()[0]
        total_chunks = conn.execute("SELECT SUM(chunk_count) FROM processed_files WHERE status='ok'").fetchone()[0] or 0

    return {
        "indexed_files": total,
        "failed_files":  failed,
        "skipped_projects": skipped,
        "active_projects":  projects,
        "total_chunks":     total_chunks,
    }


def get_indexed_files_for_project(project_id: str) -> list[str]:
    """Return all successfully indexed file paths for a project."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT file_path FROM processed_files WHERE project_id=? AND status='ok'",
            (project_id,)
        ).fetchall()
    return [row["file_path"] for row in rows]

"""
store.py — SQLite persistence for wiki project records.

Database: <DB_DIR>/wiki.db
Tables:
  projects        — one row per project_id (merged from all PDFs)
  doc_extractions — one row per (project_id, filename) audit trail

Merge policy
------------
On update, non-null values from the new extraction WIN only if:
  - the existing field is null / empty, OR
  - the new value is "more specific" (e.g. higher precision MW value)
List fields (municipalities, provinces) are UNION-merged (no duplicates).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rag_system import config
from rag_system.wiki.schema import ProjectRecord, DocExtraction

logger = logging.getLogger(__name__)

DB_PATH = config.DB_DIR / "wiki.db"


# ─────────────────────────────────────────────────────────────────────────────
# DB initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    config.ensure_dirs()
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id      TEXT PRIMARY KEY,
                project_name    TEXT,
                type            TEXT,
                summary         TEXT,
                power_mw        REAL,
                power_dc_mw     REAL,
                area_ha         REAL,
                power_source    TEXT,
                region          TEXT,
                municipalities  TEXT DEFAULT '[]',
                provinces       TEXT DEFAULT '[]',
                proponent       TEXT,
                designer        TEXT,
                procedure       TEXT,
                procedure_refs  TEXT,
                status          TEXT,
                approval_date   TEXT,
                approval_ref    TEXT,
                grid_connection TEXT,
                docs_count      INTEGER DEFAULT 0,
                last_updated    TEXT
            );

            CREATE TABLE IF NOT EXISTS doc_extractions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    TEXT    NOT NULL,
                filename      TEXT    NOT NULL,
                doc_type      TEXT,
                extracted     TEXT,
                extracted_at  TEXT,
                UNIQUE(project_id, filename)
            );

            CREATE INDEX IF NOT EXISTS idx_proj_type
                ON projects(type);
            CREATE INDEX IF NOT EXISTS idx_proj_power
                ON projects(power_mw);
            CREATE INDEX IF NOT EXISTS idx_doc_proj
                ON doc_extractions(project_id);
        """)
    logger.debug("Wiki DB initialised at %s", DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Merge helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_lists(existing_json: str, new_items: list[str]) -> str:
    """Union-merge two JSON list strings, preserve order, no duplicates."""
    existing = json.loads(existing_json or "[]")
    merged   = list(existing)
    for item in new_items:
        if item and item.strip() and item.strip() not in merged:
            merged.append(item.strip())
    return json.dumps(merged, ensure_ascii=False)


def _coalesce(old, new):
    """Return new if old is falsy, else old (keep existing data)."""
    return new if (new is not None and new != "" and old in (None, "")) else old


def _better_power(old_mw: Optional[float], new_mw: Optional[float]) -> Optional[float]:
    """
    Accept a new MW value if it's more precise (more decimal digits) or
    if the old value is missing.
    """
    if old_mw is None:
        return new_mw
    if new_mw is None:
        return old_mw
    # Prefer higher precision
    old_str = f"{old_mw:.6f}".rstrip("0")
    new_str = f"{new_mw:.6f}".rstrip("0")
    return new_mw if len(new_str) > len(old_str) else old_mw


# ─────────────────────────────────────────────────────────────────────────────
# Public write API
# ─────────────────────────────────────────────────────────────────────────────

def upsert_project(record: ProjectRecord):
    """
    Insert or merge a ProjectRecord into the projects table.
    Existing non-null fields are preserved unless the new value is better.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM projects WHERE project_id = ?", (record.project_id,)
        ).fetchone()

        if existing is None:
            # First insert — straight write
            d = record.to_dict()
            d["last_updated"] = now
            cols   = ", ".join(d.keys())
            placeholders = ", ".join("?" for _ in d)
            conn.execute(
                f"INSERT INTO projects ({cols}) VALUES ({placeholders})",
                list(d.values()),
            )
        else:
            # Merge — preserve existing data, fill blanks, union lists
            ex = dict(existing)
            updates = {
                "project_name":    _coalesce(ex["project_name"],    record.project_name),
                "type":            _coalesce(ex["type"],            record.type),
                "summary":         _coalesce(ex["summary"],         record.summary),
                "power_mw":        _better_power(ex["power_mw"],    record.power_mw),
                "power_dc_mw":     _better_power(ex["power_dc_mw"], record.power_dc_mw),
                "area_ha":         _coalesce(ex["area_ha"],         record.area_ha),
                "power_source":    ex["power_source"] if ex["power_mw"] == _better_power(ex["power_mw"], record.power_mw)
                                   else (record.power_source or ex["power_source"]),
                "region":          _coalesce(ex["region"],          record.region),
                "municipalities":  _merge_lists(ex["municipalities"], record.municipalities),
                "provinces":       _merge_lists(ex["provinces"],      record.provinces),
                "proponent":       _coalesce(ex["proponent"],       record.proponent),
                "designer":        _coalesce(ex["designer"],        record.designer),
                "procedure":       _coalesce(ex["procedure"],       record.procedure),
                "procedure_refs":  _coalesce(ex["procedure_refs"],  record.procedure_refs),
                "status":          _coalesce(ex["status"],          record.status),
                "approval_date":   _coalesce(ex["approval_date"],   record.approval_date),
                "approval_ref":    _coalesce(ex["approval_ref"],    record.approval_ref),
                "grid_connection": _coalesce(ex["grid_connection"], record.grid_connection),
                "docs_count":      (ex["docs_count"] or 0) + 1,
                "last_updated":    now,
            }
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(
                f"UPDATE projects SET {set_clause} WHERE project_id = ?",
                list(updates.values()) + [record.project_id],
            )


def save_doc_extraction(extraction: DocExtraction):
    """Save or replace a per-document extraction record."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO doc_extractions (project_id, filename, doc_type, extracted, extracted_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project_id, filename) DO UPDATE SET
                doc_type     = excluded.doc_type,
                extracted    = excluded.extracted,
                extracted_at = excluded.extracted_at
        """, (
            extraction.project_id,
            extraction.filename,
            extraction.doc_type,
            extraction.extracted,
            now,
        ))


def is_doc_extracted(project_id: str, filename: str) -> bool:
    """Return True if this file has already been wiki-extracted."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM doc_extractions WHERE project_id=? AND filename=?",
            (project_id, filename),
        ).fetchone()
    return row is not None


# ─────────────────────────────────────────────────────────────────────────────
# Public read API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_projects() -> list[dict]:
    """Return all project records as dicts, sorted by project_id."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY project_id"
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["municipalities"] = json.loads(d.get("municipalities") or "[]")
        d["provinces"]      = json.loads(d.get("provinces")      or "[]")
        results.append(d)
    return results


def get_project(project_id: str) -> Optional[dict]:
    """Return a single project record or None."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE project_id = ?", (project_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["municipalities"] = json.loads(d.get("municipalities") or "[]")
    d["provinces"]      = json.loads(d.get("provinces")      or "[]")
    return d


def get_stats() -> dict:
    """Return wiki coverage statistics."""
    with _get_conn() as conn:
        total_projects  = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        with_power      = conn.execute("SELECT COUNT(*) FROM projects WHERE power_mw IS NOT NULL").fetchone()[0]
        with_name       = conn.execute("SELECT COUNT(*) FROM projects WHERE project_name IS NOT NULL").fetchone()[0]
        total_docs      = conn.execute("SELECT COUNT(*) FROM doc_extractions").fetchone()[0]
        total_mw        = conn.execute("SELECT SUM(power_mw) FROM projects WHERE power_mw IS NOT NULL").fetchone()[0]

    return {
        "total_projects":    total_projects,
        "with_power_data":   with_power,
        "with_name_data":    with_name,
        "total_docs_parsed": total_docs,
        "total_mw_portfolio": round(total_mw or 0, 2),
    }


def execute_sql(sql: str, params: tuple = ()) -> list[dict]:
    """
    Execute a read-only SQL query against the wiki DB.
    ONLY SELECT statements are allowed.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed in wiki SQL mode.")
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "ATTACH"]
    for kw in forbidden:
        if kw in sql_stripped:
            raise ValueError(f"Forbidden SQL keyword: {kw}")

    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        for key in ("municipalities", "provinces"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        results.append(d)
    return results

"""
server.py - FastAPI server exposing an OpenAI-compatible API.

Endpoints:
  POST /auth/login                     → log in, get session token
  POST /auth/logout                    → invalidate token
  GET  /auth/me                        → current user info

  GET  /auth/users          [admin]    → list users
  POST /auth/users          [admin]    → create user
  PUT  /auth/users/{id}     [admin]    → update user
  DEL  /auth/users/{id}     [admin]    → delete user
  POST /auth/users/{id}/password [admin] → change password
  POST /auth/users/{id}/api-key  [admin] → regenerate API key

  GET  /v1/models                      → list RAG models
  GET  /v1/projects                    → indexed project stats
  GET  /v1/projects/{id}/files         → files in a project
  POST /v1/chat/completions            → RAG chat (streaming + non-streaming)

  GET  /v1/data-sources     [admin]    → registered data folders
  POST /v1/data-sources     [admin]    → register a new folder
  DEL  /v1/data-sources/{id}[admin]    → remove a folder
  POST /v1/index            [admin]    → trigger background re-indexing

  GET  /health                         → liveness check
  GET  /                               → dashboard
  GET  /app                            → full chat web interface
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from rag_system import config
from rag_system.api import auth as authlib
from rag_system.generation.chain import answer, stream_answer
from rag_system.indexing import vector_store, tracker
from rag_system.indexing.priority import QUERY_GATE

# Wiki module - optional
try:
    from rag_system.wiki import store as wiki_store
    from rag_system.wiki import extractor as wiki_extractor
    from rag_system.wiki.query import wiki_query, wiki_query_direct
    from rag_system.wiki.router import route as wiki_route, route_label as wiki_route_label
    wiki_store.init_db()
    _WIKI_AVAILABLE = True
except Exception as _wiki_err:
    _WIKI_AVAILABLE = False

def _watchdog_release_gate():
    """Safety net: if QUERY_GATE is held for >5 min, auto-release it."""
    import threading
    def _watch():
        import time
        time.sleep(300)
        if not QUERY_GATE.is_set():
            logger.warning("QUERY_GATE auto-released after 5-min watchdog")
            QUERY_GATE.set()
    threading.Thread(target=_watch, daemon=True).start()

logger = logging.getLogger(__name__)

# ── Query log (jsonl) ─────────────────────────────────────────────────────────
_QUERY_LOG = config.LOGS_DIR / "query_log.jsonl"

def _log_query(username: str, question: str, project_id: Optional[str],
               model: str, elapsed_ms: float, num_sources: int):
    """Append a query event to the JSONL analytics log."""
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts":          datetime.utcnow().isoformat(timespec="seconds"),
            "user":        username,
            "project":     project_id or "all",
            "model":       model,
            "question":    question[:200],
            "elapsed_ms":  round(elapsed_ms),
            "num_sources": num_sources,
        }
        with open(_QUERY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Query log write failed: %s", exc)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="A176LAB - RAG API",
    description="OpenAI-compatible RAG API for Italian PDF document retrieval",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    authlib.init_auth_db()
    logger.info("Auth DB initialised - default admin: admin / admin123")
    logger.info("Query analytics log: %s", _QUERY_LOG)


# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers (FastAPI dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _token_from_request(request: Request) -> Optional[str]:
    """Extract bearer token or x-api-key from the request."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return request.headers.get("x-api-key") or request.cookies.get("rag_session")


def get_current_user(request: Request) -> Optional[dict]:
    """Return session dict or None (soft auth - doesn't raise)."""
    token = _token_from_request(request)
    if not token:
        return None
    session = authlib.get_session(token)
    if session:
        return session
    return authlib.get_session_by_api_key(token)


def require_user(request: Request) -> dict:
    """Dependency - raises 401 if not logged in."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def require_admin(request: Request) -> dict:
    """Dependency - raises 401/403 if not admin."""
    user = require_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateUserRequest(BaseModel):
    username:  str
    full_name: str = ""
    email:     str = ""
    password:  str
    role:      str = "user"

class UpdateUserRequest(BaseModel):
    full_name: str = ""
    email:     str = ""
    role:      str = "user"

class ChangePasswordRequest(BaseModel):
    new_password: str

class AddDataSourceRequest(BaseModel):
    path:  str
    label: str = ""

class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    model:       str              = "rag-all"
    llm_model:   Optional[str]    = None   # actual Ollama LLM to use for generation
    messages:    list[Message]
    stream:      bool             = False
    temperature: Optional[float]  = None
    max_tokens:  Optional[int]    = None
    project_id:  Optional[str]    = None   # explicit project override from the web UI
    top_k:       Optional[int]    = None   # number of chunks to retrieve (default from config)


# ─────────────────────────────────────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/auth/login")
def login(req: LoginRequest):
    token = authlib.login(req.username, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    session = authlib.get_session(token)
    return {
        "token":     token,
        "username":  session["username"],
        "full_name": session.get("full_name", ""),
        "role":      session["role"],
    }


@app.post("/auth/logout")
def logout(request: Request):
    token = _token_from_request(request)
    if token:
        authlib.logout(token)
    return {"ok": True}


@app.get("/auth/me")
def me(user: dict = Depends(require_user)):
    return {
        "username":  user["username"],
        "full_name": user.get("full_name", ""),
        "role":      user["role"],
    }


# ─── User management (admin) ──────────────────────────────────────────────────

@app.get("/auth/users")
def list_users(admin: dict = Depends(require_admin)):
    users = authlib.list_users()
    # Never return password hashes
    for u in users:
        u.pop("password_hash", None)
        u.pop("salt", None)
    return {"users": users}


@app.post("/auth/users")
def create_user(req: CreateUserRequest, admin: dict = Depends(require_admin)):
    ok, err = authlib.create_user(req.username, req.full_name, req.email, req.password, req.role)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    users = [u for u in authlib.list_users() if u["username"] == req.username]
    user = users[0] if users else {}
    user.pop("password_hash", None)
    user.pop("salt", None)
    return {"user": user}


@app.put("/auth/users/{user_id}")
def update_user(user_id: int, req: UpdateUserRequest, admin: dict = Depends(require_admin)):
    authlib.update_user(user_id, req.full_name, req.email, req.role)
    return {"ok": True}


@app.delete("/auth/users/{user_id}")
def delete_user(user_id: int, admin: dict = Depends(require_admin)):
    ok, err = authlib.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    return {"ok": True}


@app.post("/auth/users/{user_id}/password")
def change_password(user_id: int, req: ChangePasswordRequest, admin: dict = Depends(require_admin)):
    authlib.change_password(user_id, req.new_password)
    return {"ok": True}


@app.post("/auth/users/{user_id}/api-key")
def regenerate_api_key(user_id: int, admin: dict = Depends(require_admin)):
    key = authlib.regenerate_api_key(user_id)
    return {"api_key": key}


# ─────────────────────────────────────────────────────────────────────────────
# Data source management (admin)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/data-sources")
def list_data_sources(admin: dict = Depends(require_admin)):
    return {"data_sources": authlib.list_data_sources()}


@app.post("/v1/data-sources")
def add_data_source(req: AddDataSourceRequest, admin: dict = Depends(require_admin)):
    ok, err = authlib.add_data_source(req.path, req.label, admin["username"])
    if not ok:
        raise HTTPException(status_code=400, detail=err)
    return {"ok": True, "data_sources": authlib.list_data_sources()}


@app.delete("/v1/data-sources/{source_id}")
def remove_data_source(source_id: int, admin: dict = Depends(require_admin)):
    authlib.remove_data_source(source_id)
    return {"ok": True}


@app.get("/v1/stats")
def get_stats(user: dict = Depends(require_user)):
    """Aggregated stats + per-project file tree for the web UI."""
    import sqlite3
    db_stats = tracker.get_stats()

    # ── Read tracker DB: original project IDs and their files ────────────────
    db = config.DB_DIR / "tracker.db"
    files_by_project: dict = {}   # original_project_id → [filename, ...]
    if db.exists():
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT project_id, file_path FROM processed_files WHERE status='ok' ORDER BY project_id, file_path"
        ).fetchall()
        conn.close()
        for r in rows:
            pid = r["project_id"]
            files_by_project.setdefault(pid, []).append(Path(r["file_path"]).name)

    # ── Build project list using original IDs from tracker DB ────────────────
    # Qdrant collection names are sanitized versions of the original project IDs.
    # Use config.sanitize_collection_name to translate for the Qdrant lookup.
    all_original_ids = sorted(files_by_project.keys())

    # Also include any Qdrant collections not yet in the tracker DB
    qdrant_projects = set(vector_store.list_projects())  # sanitized, prefix stripped
    tracked_sanitized = {
        config.sanitize_collection_name(pid).replace(config.QDRANT_COLLECTION_PREFIX, "", 1): pid
        for pid in all_original_ids
    }
    for qp in qdrant_projects:
        if qp not in tracked_sanitized:
            all_original_ids.append(qp)   # fallback: use sanitized name as display name

    projects_out = []
    for orig_pid in sorted(set(all_original_ids)):
        # Resolve Qdrant collection name
        if orig_pid in files_by_project:
            coll_suffix = config.sanitize_collection_name(orig_pid).replace(config.QDRANT_COLLECTION_PREFIX, "", 1)
        else:
            coll_suffix = orig_pid   # already sanitized (fallback)
        info = vector_store.collection_info(coll_suffix)
        docs = files_by_project.get(orig_pid, [])
        projects_out.append({
            "name":       orig_pid,
            "doc_count":  len(docs),
            "chunks":     info.get("points", 0),
            "documents":  docs,
        })

    total_docs   = sum(len(v) for v in files_by_project.values())
    total_chunks = db_stats.get("total_chunks", 0)
    n_collections = len(vector_store.list_projects())

    return {
        "total_projects":    len(projects_out),
        "total_documents":   total_docs,
        "total_chunks":      total_chunks,
        "total_collections": n_collections,
        "projects":          projects_out,
    }


# Indexing status (simple in-memory flag)
_indexing_state = {
    "running": False,
    "message": "In attesa",
    "started_at": None,
    "finished_at": None,
    "files_processed": 0,
    "total_files": 0,
    "projects_processed": 0,
    "total_projects": 0,
}


@app.get("/v1/index/status")
def indexing_status(user: dict = Depends(require_user)):
    progress = ""
    if _indexing_state["total_files"] > 0:
        progress = f"{_indexing_state['projects_processed']}/{_indexing_state['total_projects']} projects - {_indexing_state['files_processed']}/{_indexing_state['total_files']} files"
    elif _indexing_state["running"]:
        progress = f"0/{_indexing_state.get('total_projects', 0)} projects - 0/{_indexing_state.get('total_files', 0)} files (scanning...)"
    return {
        "status":      "running" if _indexing_state["running"] else "idle",
        "message":     _indexing_state["message"],
        "progress":    progress,
        "started_at":  _indexing_state.get("started_at"),
        "finished_at": _indexing_state.get("finished_at"),
    }


class IndexRequest(BaseModel):
    force_reindex: bool         = False
    project_id:    Optional[str] = None   # if set, only re-index this project


@app.post("/v1/index")
async def trigger_indexing(
    req: IndexRequest = IndexRequest(),
    user: dict = Depends(require_user),
):
    """
    Trigger indexing in the background.
    - Omit project_id to re-index everything (admin-only).
    - Supply project_id to re-index a single project (any logged-in user).
    Returns 202 immediately.
    """
    # Full re-index requires admin; per-project re-index is open to all users
    if req.project_id is None and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Full re-index requires admin role")

    if _indexing_state["running"]:
        scope = req.project_id or "ALL"
        raise HTTPException(status_code=409, detail=f"Indexing already in progress ({scope})")

    project_filter = req.project_id   # None → all projects

    async def _run():
        from rag_system.indexing.pipeline import run_indexing
        from rag_system import config as _cfg
        _indexing_state["running"]           = True
        _indexing_state["started_at"]        = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")
        _indexing_state["finished_at"]       = None
        _indexing_state["message"]           = f"In corso: {project_filter or 'tutti i progetti'}…"
        _indexing_state["files_processed"]   = 0
        _indexing_state["projects_processed"] = 0
        _indexing_state["total_files"]       = 0
        _indexing_state["total_projects"]    = 0
        if req.force_reindex:
            _cfg.FORCE_REINDEX = True
        try:
            logger.info(
                "Indexing triggered by %s - project=%s force=%s",
                user["username"], project_filter or "ALL", req.force_reindex
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: run_indexing(project_filter=project_filter)
            )
            _indexing_state["message"]     = "Completato"
            _indexing_state["finished_at"] = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")
        except Exception as exc:
            logger.error("Indexing failed: %s", exc)
            _indexing_state["message"] = f"Errore: {exc}"
        finally:
            _indexing_state["running"] = False
            if req.force_reindex:
                _cfg.FORCE_REINDEX = False   # reset so next run is incremental

    asyncio.create_task(_run())
    scope_msg = f'Progetto "{project_filter}"' if project_filter else "Tutti i progetti"
    return {"ok": True, "message": f"{scope_msg} - indicizzazione avviata in background"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_project_from_model(model_name: str) -> Optional[str]:
    import re
    prefix = "rag-"
    if not model_name.startswith(prefix):
        return None
    suffix = model_name[len(prefix):]
    if suffix == "all":
        return None
    projects = vector_store.list_projects()
    for p in projects:
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", p)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_").lower()
        if sanitized == suffix:
            return p
    return None


def _extract_project_from_messages(messages: list[Message]) -> Optional[str]:
    import re
    for msg in messages:
        if msg.role == "system":
            m = re.search(r"\[project:([^\]]+)\]", msg.content)
            if m:
                return m.group(1).strip()
    return None


def _build_openai_response(answer_text: str, model: str, sources: list[dict]) -> dict:
    sources_md = ""
    if sources:
        lines = ["\n\n---\n**Fonti:**"]
        for s in sources:
            lines.append(f"- {s['source']} (rilevanza: {s['score']})")
        sources_md = "\n".join(lines)
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer_text + sources_md},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_chunk(content: str, model: str) -> str:
    data = {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Wiki endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/wiki/projects")
def wiki_list_projects(user: dict = Depends(require_user)):
    """Return all structured project records from the wiki database."""
    if not _WIKI_AVAILABLE:
        raise HTTPException(503, "Wiki module not available")
    return {"projects": wiki_store.get_all_projects()}


@app.get("/v1/wiki/projects/{project_id:path}")
def wiki_get_project(project_id: str, user: dict = Depends(require_user)):
    """Return a single project's wiki record."""
    if not _WIKI_AVAILABLE:
        raise HTTPException(503, "Wiki module not available")
    rec = wiki_store.get_project(project_id)
    if rec is None:
        raise HTTPException(404, f"Project '{project_id}' not found in wiki")
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Log / project status endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/log/projects")
def log_projects(user: dict = Depends(require_user)):
    """
    Returns per-project indexing status by cross-referencing the data folder
    (disk PDFs) with the tracker database.

    Each project entry contains:
      - total_on_disk:  PDF count found in the data folder
      - indexed:        files successfully indexed
      - failed:         files that errored
      - pending:        files on disk but not yet in the tracker
      - total_chunks:   sum of all chunk counts
      - last_activity:  most recent indexed_at timestamp
      - health:         "ok" | "partial" | "failed" | "empty"
      - files:          list of all files with their status
    """
    import sqlite3

    # 1. Read tracker DB
    db = config.DB_DIR / "tracker.db"
    tracked: dict = {}   # file_path → row dict
    skipped_projects: set = set()

    if db.exists():
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT project_id, file_path, chunk_count, status, error, indexed_at "
            "FROM processed_files ORDER BY file_path"
        ).fetchall()
        for r in rows:
            tracked[r["file_path"]] = dict(r)
        skip_rows = conn.execute("SELECT project_id FROM skipped_projects").fetchall()
        skipped_projects = {s["project_id"] for s in skip_rows}
        conn.close()

    # 2. Scan data folder for PDF files grouped by project
    data_dir = config.DATA_DIR
    projects: dict = {}

    if data_dir.exists():
        for proj_dir in sorted(data_dir.iterdir()):
            if not proj_dir.is_dir():
                continue
            pid = proj_dir.name
            pdfs = sorted(proj_dir.glob("*.pdf"))
            if not pdfs and pid not in skipped_projects:
                # check tracker for this project anyway
                proj_tracked = [v for v in tracked.values() if v["project_id"] == pid]
                if not proj_tracked:
                    continue
            projects[pid] = {"pdfs": pdfs, "skipped": pid in skipped_projects}

    # 3. Build response
    result = []
    for pid, info in projects.items():
        files_out = []
        total_chunks = 0
        last_activity = ""
        n_ok = n_failed = n_pending = 0

        disk_paths = {str(p): p.name for p in info["pdfs"]}

        # Files on disk
        for fpath, fname in disk_paths.items():
            tr = tracked.get(fpath)
            if tr:
                status     = tr["status"]
                chunks     = tr["chunk_count"] or 0
                indexed_at = (tr["indexed_at"] or "")[:16].replace("T", " ")
                error      = tr["error"] or ""
                total_chunks += chunks
                if status == "ok":
                    n_ok += 1
                else:
                    n_failed += 1
                if indexed_at > last_activity:
                    last_activity = indexed_at
            else:
                status = "pending"
                chunks = 0
                indexed_at = ""
                error = ""
                n_pending += 1

            files_out.append({
                "name":       fname,
                "status":     status,
                "chunks":     chunks,
                "indexed_at": indexed_at,
                "error":      error,
            })

        # Files in tracker but no longer on disk (orphaned)
        for fpath, tr in tracked.items():
            if tr["project_id"] == pid and fpath not in disk_paths:
                files_out.append({
                    "name":       Path(fpath).name,
                    "status":     "orphaned",
                    "chunks":     tr["chunk_count"] or 0,
                    "indexed_at": (tr["indexed_at"] or "")[:16].replace("T", " "),
                    "error":      "File no longer on disk",
                })

        total_on_disk = len(disk_paths)

        if info["skipped"]:
            health = "skipped"
        elif n_failed > 0:
            health = "failed"
        elif n_pending > 0:
            health = "partial"
        elif n_ok == 0:
            health = "empty"
        else:
            health = "ok"

        result.append({
            "project_id":    pid,
            "total_on_disk": total_on_disk,
            "indexed":       n_ok,
            "failed":        n_failed,
            "pending":       n_pending,
            "total_chunks":  total_chunks,
            "last_activity": last_activity,
            "health":        health,
            "skipped":       info["skipped"],
            "files":         sorted(files_out, key=lambda f: (
                0 if f["status"]=="failed" else
                1 if f["status"]=="pending" else
                2 if f["status"]=="orphaned" else 3,
                f["name"]
            )),
        })

    # Sort: failed first, then partial, then ok
    order = {"failed": 0, "partial": 1, "skipped": 2, "empty": 3, "ok": 4}
    result.sort(key=lambda p: (order.get(p["health"], 9), p["project_id"]))

    total_files    = sum(p["total_on_disk"] for p in result)
    total_indexed  = sum(p["indexed"]       for p in result)
    total_failed   = sum(p["failed"]        for p in result)
    total_pending  = sum(p["pending"]       for p in result)
    total_chunks   = sum(p["total_chunks"]  for p in result)

    return {
        "summary": {
            "projects":     len(result),
            "total_files":  total_files,
            "indexed":      total_indexed,
            "failed":       total_failed,
            "pending":      total_pending,
            "total_chunks": total_chunks,
        },
        "projects": result,
    }


@app.get("/v1/wiki/stats")
def wiki_stats(user: dict = Depends(require_user)):
    """Return wiki extraction coverage statistics."""
    if not _WIKI_AVAILABLE:
        raise HTTPException(503, "Wiki module not available")
    return wiki_store.get_stats()


class WikiQueryRequest(BaseModel):
    question: str
    raw_sql:  Optional[str] = None   # if provided, execute directly (admin only)


@app.post("/v1/wiki/query")
def wiki_nl_query(req: WikiQueryRequest, user: dict = Depends(require_user)):
    """
    Natural-language query against the wiki structured database.
    Converts the question to SQL, executes it, formats the result in Italian.
    """
    if not _WIKI_AVAILABLE:
        raise HTTPException(503, "Wiki module not available")
    if req.raw_sql:
        # Direct SQL execution for admin/debug use
        if user.get("role") != "admin":
            raise HTTPException(403, "Direct SQL requires admin role")
        try:
            rows = wiki_query_direct(req.raw_sql)
            return {"rows": rows, "sql": req.raw_sql, "source": "wiki"}
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    result = wiki_query(req.question)
    return result


@app.post("/v1/wiki/reindex")
async def wiki_reindex(
    project_id: Optional[str] = None,
    admin: dict = Depends(require_admin),
):
    """
    Trigger wiki re-extraction for all files in a project (or all projects).
    Runs in background thread so it doesn't block the response.
    """
    if not _WIKI_AVAILABLE:
        raise HTTPException(503, "Wiki module not available")

    import threading
    from pathlib import Path

    def _run_extraction(pid_filter: Optional[str]):
        data_dir = config.DATA_DIR
        folders = sorted(d for d in data_dir.iterdir() if d.is_dir())
        if pid_filter:
            folders = [f for f in folders if f.name == pid_filter]
        wiki_store.init_db()
        for folder in folders:
            project = folder.name
            for pdf in sorted(folder.glob("*.pdf")):
                try:
                    wiki_extractor.extract_and_store(pdf, project, force=True)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Wiki reindex failed for %s: %s", pdf.name, exc
                    )

    t = threading.Thread(target=_run_extraction, args=(project_id,), daemon=True)
    t.start()
    return {
        "status": "started",
        "project_filter": project_id or "all",
        "message": "Wiki re-extraction running in background. Check /v1/wiki/stats for progress.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "a176lab-rag"}


@app.get("/debug/query")
def debug_query(q: str = "requisiti tecnici", user: dict = Depends(require_user), request: Request = None):
    """
    Diagnostic endpoint - tests every step of the retrieval pipeline and reports
    exactly where it fails.  Open in browser:
      http://localhost:8000/debug/query?q=your+question
    """
    import traceback
    from rag_system.indexing import embedder as emb

    report = {}

    # 1. Embedding test
    try:
        vec = emb.embed_text(q)
        if vec is None:
            report["embed"] = "FAIL - embed_text returned None (Ollama unreachable or model not loaded)"
        else:
            report["embed"] = f"OK - {len(vec)}-dim vector, first 3 values: {vec[:3]}"
    except Exception as e:
        report["embed"] = f"ERROR - {e}\n{traceback.format_exc()}"
        vec = None

    # 2. Qdrant connection
    try:
        cols = vector_store.get_client().get_collections().collections
        col_names = [c.name for c in cols if c.name.startswith(config.QDRANT_COLLECTION_PREFIX)]
        report["qdrant"] = f"OK - {len(col_names)} RAG collections: {col_names}"
    except Exception as e:
        report["qdrant"] = f"ERROR - {e}"
        col_names = []

    # 3. Point counts
    counts = {}
    for name in col_names:
        try:
            info = vector_store.get_client().get_collection(name)
            counts[name] = info.points_count
        except Exception as e:
            counts[name] = f"error: {e}"
    report["point_counts"] = counts

    # 4. Raw search (threshold=0 so everything comes back)
    if vec is not None and col_names:
        try:
            response = vector_store.get_client().query_points(
                collection_name = col_names[0],
                query           = vec,
                limit           = 3,
                score_threshold = 0.0,
                with_payload    = False,
            )
            scores = [round(h.score, 4) for h in response.points]
            report["raw_search"] = (
                f"OK - top-3 scores from '{col_names[0]}': {scores}. "
                f"Config threshold is {config.RETRIEVAL_SCORE_THRESHOLD}. "
                + ("⚠ All below threshold - lower RETRIEVAL_SCORE_THRESHOLD in .env"
                   if scores and max(scores) < config.RETRIEVAL_SCORE_THRESHOLD else "")
            )
        except Exception as e:
            report["raw_search"] = f"ERROR - {e}"
    else:
        report["raw_search"] = "SKIPPED (embed or Qdrant failed)"

    # 5. Full retrieval
    if vec is not None:
        try:
            from rag_system.retrieval.retriever import retrieve
            results = retrieve(q)
            report["retrieve"] = (
                f"OK - {len(results)} chunks returned above threshold {config.RETRIEVAL_SCORE_THRESHOLD}"
                if results else
                f"EMPTY - 0 chunks above threshold {config.RETRIEVAL_SCORE_THRESHOLD}"
            )
        except Exception as e:
            report["retrieve"] = f"ERROR - {e}"

    return report


@app.get("/v1/models")
def list_models():
    import re
    projects = vector_store.list_projects()
    model_list = [{
        "id": "rag-all", "object": "model", "created": 0, "owned_by": "a176lab",
        "description": "Cerca in tutti i progetti",
    }]
    for proj in sorted(projects):
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", proj)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_").lower()
        model_list.append({
            "id": f"rag-{sanitized}", "object": "model", "created": 0, "owned_by": "a176lab",
            "description": f"Progetto: {proj}",
        })
    return {"object": "list", "data": model_list}


@app.get("/v1/llm-models")
def list_llm_models():
    """
    Proxy Ollama's GET /api/tags to return available LLM models.
    The configured LLM_MODEL is always listed first.
    Falls back to a minimal list if Ollama is unreachable.
    """
    try:
        import httpx as _httpx
        resp = _httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        # Extract model names, put configured default first
        names = [m["name"] for m in tags if "name" in m]
        if config.LLM_MODEL in names:
            names.remove(config.LLM_MODEL)
        names.insert(0, config.LLM_MODEL)
        return {"models": names, "default": config.LLM_MODEL}
    except Exception as exc:
        logger.warning("Could not fetch Ollama model list: %s", exc)
        return {"models": [config.LLM_MODEL], "default": config.LLM_MODEL}


@app.get("/v1/projects")
def list_projects():
    projects = vector_store.list_projects()
    result = []
    for proj in sorted(projects):
        info = vector_store.collection_info(proj)
        result.append({
            "project_id": proj,
            "chunks":     info.get("points", 0),
            "status":     info.get("status", "unknown"),
            "collection": info.get("collection", ""),
        })
    db_stats = tracker.get_stats()
    return {"projects": result, "stats": db_stats}


@app.get("/v1/projects/{project_id}/files")
def list_project_files(project_id: str):
    import sqlite3
    db = config.DB_DIR / "tracker.db"
    if not db.exists():
        return {"project_id": project_id, "files": []}
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT file_path, chunk_count, status, indexed_at "
        "FROM processed_files WHERE project_id = ? ORDER BY file_path",
        (project_id,)
    ).fetchall()
    conn.close()
    return {
        "project_id": project_id,
        "files": [
            {
                "name":       Path(r["file_path"]).name,
                "chunks":     r["chunk_count"] or 0,
                "status":     r["status"],
                "indexed_at": (r["indexed_at"] or "")[:16].replace("T", " "),
            }
            for r in rows
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Auth is optional - works without a token for Open-WebUI compatibility.
    When a token is present the query is attributed to that user in the log.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    user = get_current_user(request)
    username = user["username"] if user else "anonymous"

    project_id = (
        req.project_id
        or _extract_project_from_messages(req.messages)
        or _extract_project_from_model(req.model)
    )

    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    question = user_messages[-1].content

    history = [
        {"role": m.role, "content": m.content}
        for m in req.messages[:-1]
        if m.role in ("user", "assistant")
    ]

    logger.info("Query [%s]: '%s…' | project=%s | stream=%s",
                username, question[:60], project_id or "ALL", req.stream)

    t0 = time.time()

    if req.stream:
        def generate():
            QUERY_GATE.clear(); _watchdog_release_gate()
            sources = []
            try:
                sources_json = None
                for token in stream_answer(  # noqa: E501
                    question,
                    project_id = project_id,
                    model      = req.llm_model or config.LLM_MODEL,
                    history    = history or None,
                ):
                    if token.startswith("\n\n__SOURCES__:"):
                        sources_json = token.replace("\n\n__SOURCES__:", "")
                        continue
                    yield _sse_chunk(token, req.model)
                if sources_json:
                    try:
                        sources = json.loads(sources_json)
                        if sources:
                            src_chunk = {
                                "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
                                "object":  "chat.completion.chunk",
                                "created": int(time.time()),
                                "model":   req.model,
                                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}],
                                "sources": [
                                    {
                                        "document_id": s.get("source", ""),
                                        "page":        s.get("page", "?"),
                                        "score":       s.get("score", 0),
                                    }
                                    for s in sources
                                ],
                            }
                            yield f"data: {json.dumps(src_chunk, ensure_ascii=False)}\n\n"
                    except Exception:
                        pass
                yield _sse_done()
            except Exception as _stream_exc:
                logger.error("Streaming generation failed for query '%s': %s",
                             question[:80], _stream_exc, exc_info=True)
                yield _sse_chunk(f"\n[Errore interno: {_stream_exc}]", req.model)
                yield _sse_done()
            finally:
                QUERY_GATE.set()
                # Always log - even zero-source queries help diagnose embedding failures
                _log_query(username, question, project_id, req.model,
                           (time.time() - t0) * 1000, len(sources))
        return StreamingResponse(generate(), media_type="text/event-stream")

    QUERY_GATE.clear(); _watchdog_release_gate()
    try:
        result = answer(question, project_id=project_id, history=history or None, model=req.llm_model or config.LLM_MODEL)
    finally:
        QUERY_GATE.set()
    _log_query(username, question, project_id, req.model,
               (time.time() - t0) * 1000, len(result.get("sources", [])))
    return _build_openai_response(result["answer"], req.model, result["sources"])


# ─────────────────────────────────────────────────────────────────────────────
# Direct RAG Query API  (spec-compliant, simpler than /v1/chat/completions)
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    project_id: Optional[str]   = None
    top_k:      int              = 5
    model:      Optional[str]   = None
    stream:     bool             = True


@app.post("/query")
async def direct_query(req: QueryRequest, request: Request):
    """
    Direct RAG query endpoint.

    POST /query
    {
        "question": "Qual è il valore del progetto?",
        "project_id": "NomeProgetto",   // optional
        "top_k": 5,
        "model": "qwen3.6:35b-a3b",     // optional, overrides config
        "stream": true                  // default true
    }

    Streaming response: text/event-stream with JSON tokens.
    Blocking response: {"answer": "...", "sources": [...], "model": "...", "project": "..."}
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    user     = get_current_user(request)
    username = user["username"] if user else "anonymous"
    model    = req.model or config.LLM_MODEL
    t0       = time.time()

    logger.info("Direct query [%s]: '%s…' | project=%s | stream=%s",
                username, req.question[:60], req.project_id or "ALL", req.stream)

    if req.stream:
        def _generate():
            QUERY_GATE.clear(); _watchdog_release_gate()          # pause indexing embedding batches
            try:
                sources_payload = None
                for token in stream_answer(
                    req.question,
                    project_id = req.project_id,
                    top_k      = req.top_k,
                    model      = model,
                ):
                    if token.startswith("\n\n__SOURCES__:"):
                        sources_payload = token.replace("\n\n__SOURCES__:", "")
                        continue
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

                if sources_payload:
                    try:
                        sources = json.loads(sources_payload)
                        yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
                        _log_query(username, req.question, req.project_id, model,
                                   (time.time() - t0) * 1000, len(sources))
                    except Exception:
                        pass

                yield "data: [DONE]\n\n"
            finally:
                QUERY_GATE.set()        # resume indexing

        return StreamingResponse(_generate(), media_type="text/event-stream")

    # ── Blocking mode ──────────────────────────────────────────────────────────
    QUERY_GATE.clear(); _watchdog_release_gate()                  # pause indexing embedding batches
    try:
        result = answer(req.question, project_id=req.project_id, top_k=req.top_k, model=model)
    finally:
        QUERY_GATE.set()                # resume indexing
    _log_query(username, req.question, req.project_id, model,
               (time.time() - t0) * 1000, len(result.get("sources", [])))
    return {
        "answer":  result["answer"],
        "sources": result["sources"],
        "model":   model,
        "project": req.project_id or "all",
    }


@app.get("/status")
def system_status(user: dict = Depends(require_user)):
    """
    Combined system status: ingestion progress + vector store stats + config.

    GET /status → {
        "ingestion":    {...},
        "vector_store": {...},
        "system":       {...}
    }
    """
    db_stats     = tracker.get_stats()
    projects_raw = vector_store.list_projects()

    project_details = []
    total_chunks    = 0
    for p in sorted(projects_raw):
        info  = vector_store.collection_info(p)
        pts   = info.get("points", 0)
        total_chunks += pts
        project_details.append({
            "project_id": p,
            "chunks":     pts,
            "status":     info.get("status", "unknown"),
        })

    return {
        "ingestion": {
            "status":          "running" if _indexing_state["running"] else "idle",
            "message":         _indexing_state["message"],
            "indexed_files":   db_stats.get("indexed_files", 0),
            "failed_files":    db_stats.get("failed_files", 0),
            "active_projects": db_stats.get("active_projects", 0),
        },
        "vector_store": {
            "total_collections": len(projects_raw),
            "total_chunks":      total_chunks,
            "projects":          project_details,
        },
        "system": {
            "llm_model":    config.LLM_MODEL,
            "embed_model":  config.EMBED_MODEL,
            "top_k":        config.RETRIEVAL_TOP_K,
            "bm25_enabled": config.BM25_ENABLED,
            "bm25_weight":  config.BM25_WEIGHT,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard + web UI
# ─────────────────────────────────────────────────────────────────────────────

def _tracker_all_projects_files() -> dict:
    import sqlite3
    db = config.DB_DIR / "tracker.db"
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT project_id, file_path, chunk_count, status, indexed_at "
        "FROM processed_files ORDER BY project_id, file_path"
    ).fetchall()
    skipped = conn.execute(
        "SELECT project_id, reason, detected_at FROM skipped_projects"
    ).fetchall()
    conn.close()
    projects: dict = {}
    for r in rows:
        pid = r["project_id"]
        if pid not in projects:
            projects[pid] = {"files": [], "skipped": False, "reason": ""}
        projects[pid]["files"].append({
            "name":       Path(r["file_path"]).name,
            "chunks":     r["chunk_count"] or 0,
            "status":     r["status"],
            "indexed_at": (r["indexed_at"] or "")[:16].replace("T", " "),
        })
    for s in skipped:
        pid = s["project_id"]
        if pid not in projects:
            projects[pid] = {"files": [], "skipped": True, "reason": s["reason"]}
        else:
            projects[pid]["skipped"] = True
            projects[pid]["reason"]  = s["reason"]
    return projects


@app.get("/app", response_class=HTMLResponse)
def chat_app():
    """Full web interface."""
    html_file = Path(__file__).parent / "app.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="app.html not found")
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """System dashboard - project browser."""
    import re as _re
    stats    = tracker.get_stats()
    projects = _tracker_all_projects_files()

    cards_html = ""
    for pid in sorted(projects.keys()):
        info  = projects[pid]
        files = info["files"]
        n_ok     = sum(1 for f in files if f["status"] == "ok")
        n_failed = sum(1 for f in files if f["status"] == "failed")
        n_chunks = sum(f["chunks"] for f in files)
        skipped_badge = (
            f'<span class="badge skip">SKIPPED - {info["reason"]}</span>'
            if info["skipped"] and not files else ""
        )
        rows = ""
        for f in files:
            sc = "ok" if f["status"] == "ok" else "fail"
            sl = "✓ indexed" if f["status"] == "ok" else "✗ failed"
            rows += f'<tr><td class="fname">{f["name"]}</td><td class="center">{f["chunks"]}</td><td class="center"><span class="badge {sc}">{sl}</span></td><td class="center muted">{f["indexed_at"]}</td></tr>'
        sanitized = _re.sub(r"[^a-zA-Z0-9_]", "_", pid)
        sanitized = _re.sub(r"_+", "_", sanitized).strip("_").lower()
        no_row = "" if rows else '<tr><td colspan="4" class="center muted">Nessun file</td></tr>'
        cards_html += f"""
        <div class="card" id="proj-{sanitized}">
          <div class="card-header">
            <span class="proj-name">{pid}</span>
            {skipped_badge}
            <span class="pill">{n_ok} ok</span>
            {"<span class='pill fail'>" + str(n_failed) + " fail</span>" if n_failed else ""}
            <span class="pill muted">{n_chunks} chunks</span>
          </div>
          <table class="file-table">
            <thead><tr><th>File</th><th class="center">Chunks</th><th class="center">Stato</th><th class="center">Indicizzato</th></tr></thead>
            <tbody>{rows}{no_row}</tbody>
          </table>
        </div>"""

    total_p = stats.get("total_projects", 0)
    total_f = stats.get("total_files", 0)
    total_c = stats.get("total_chunks", 0)

    html = (
        "<!DOCTYPE html>"
        "<html lang='it'><head><meta charset='utf-8'>"
        "<title>RAG Dashboard</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;background:#f5f5f5;margin:0;padding:20px}"
        "h1{color:#333;margin-bottom:4px}"
        ".subtitle{color:#888;font-size:.9em;margin-bottom:20px}"
        ".stats-bar{display:flex;gap:16px;margin-bottom:24px}"
        ".stat-box{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:12px 20px;text-align:center}"
        ".stat-box .num{font-size:1.8em;font-weight:700;color:#2563eb}"
        ".stat-box .lbl{font-size:.8em;color:#888;margin-top:2px}"
        ".card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;margin-bottom:16px;overflow:hidden}"
        ".card-header{padding:12px 16px;border-bottom:1px solid #f0f0f0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}"
        ".proj-name{font-weight:600;color:#1e293b;flex:1}"
        ".pill{background:#eff6ff;color:#2563eb;border-radius:4px;padding:2px 8px;font-size:.78em;font-weight:600}"
        ".pill.fail{background:#fef2f2;color:#dc2626}"
        ".pill.muted{background:#f1f5f9;color:#64748b}"
        ".badge{border-radius:4px;padding:2px 8px;font-size:.78em;font-weight:600}"
        ".badge.ok{background:#dcfce7;color:#16a34a}"
        ".badge.fail{background:#fef2f2;color:#dc2626}"
        ".badge.skip{background:#fef9c3;color:#b45309}"
        ".file-table{width:100%;border-collapse:collapse;font-size:.85em}"
        ".file-table th{background:#f8fafc;color:#64748b;font-weight:600;padding:8px 12px;text-align:left;border-bottom:1px solid #e2e8f0}"
        ".file-table td{padding:7px 12px;border-bottom:1px solid #f1f5f9;color:#334155}"
        ".file-table tr:last-child td{border-bottom:none}"
        ".fname{font-family:monospace;font-size:.82em;color:#475569}"
        ".center{text-align:center}.muted{color:#94a3b8}"
        "</style></head><body>"
    )
    html += f"<h1>\U0001f4ca RAG System Dashboard</h1>"
    html += "<p class='subtitle'>Panoramica dei progetti indicizzati</p>"
    html += "<div class='stats-bar'>"
    html += f"<div class='stat-box'><div class='num'>{total_p}</div><div class='lbl'>Progetti</div></div>"
    html += f"<div class='stat-box'><div class='num'>{total_f}</div><div class='lbl'>Documenti</div></div>"
    html += f"<div class='stat-box'><div class='num'>{total_c}</div><div class='lbl'>Chunks totali</div></div>"
    html += "</div>"
    if cards_html:
        html += cards_html
    else:
        html += "<p style='color:#888;text-align:center;padding:40px'>Nessun progetto indicizzato.</p>"
    html += "</body></html>"
    return HTMLResponse(content=html)

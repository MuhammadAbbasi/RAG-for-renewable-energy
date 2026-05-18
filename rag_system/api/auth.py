"""
auth.py — Authentication, user management, and data source registry.

Uses only Python stdlib (hashlib, hmac, secrets) — no extra packages.
Passwords: PBKDF2-HMAC-SHA256 with random salt (260 000 iterations).
Sessions:  Random URL-safe token stored in SQLite with 24h expiry.
API keys:  Per-user static key for Open-WebUI / curl access.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rag_system import config

AUTH_DB = config.DB_DIR / "auth.db"
_SESSION_TTL_HOURS = 24


# ─── DB helpers ──────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(AUTH_DB))
    c.row_factory = sqlite3.Row
    return c


def init_auth_db():
    """Create tables and seed the default admin account on first run."""
    AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            full_name     TEXT    DEFAULT '',
            email         TEXT    DEFAULT '',
            password_hash TEXT    NOT NULL,
            salt          TEXT    NOT NULL,
            role          TEXT    DEFAULT 'user',   -- 'admin' | 'user'
            api_key       TEXT    UNIQUE,
            created_at    TEXT    NOT NULL,
            last_login    TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT    PRIMARY KEY,
            user_id     INTEGER NOT NULL,
            username    TEXT    NOT NULL,
            full_name   TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            expires_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS data_sources (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT    UNIQUE NOT NULL,
            label       TEXT    DEFAULT '',
            added_by    TEXT    DEFAULT '',
            added_at    TEXT    NOT NULL,
            enabled     INTEGER DEFAULT 1
        );
        """)

        # Seed default admin if DB is brand new
        count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            _create_user_internal(db, "admin", "Administrator", "", "admin123", "admin")


# ─── Password hashing ─────────────────────────────────────────────────────────

def _hash(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex()


def _new_api_key() -> str:
    return "rag-" + secrets.token_urlsafe(32)


# ─── Auth ────────────────────────────────────────────────────────────────────

def login(username: str, password: str) -> Optional[str]:
    """
    Verify credentials. Returns a session token on success, None on failure.
    """
    with _conn() as db:
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not user:
        return None
    expected = _hash(password, user["salt"])
    if not hmac.compare_digest(expected, user["password_hash"]):
        return None

    token     = secrets.token_urlsafe(32)
    expires   = (datetime.utcnow() + timedelta(hours=_SESSION_TTL_HOURS)).isoformat()
    now       = datetime.utcnow().isoformat()

    with _conn() as db:
        db.execute(
            "INSERT INTO sessions (token,user_id,username,full_name,role,expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (token, user["id"], user["username"], user["full_name"], user["role"], expires),
        )
        db.execute("UPDATE users SET last_login=? WHERE id=?", (now, user["id"]))

    return token


def get_session(token: str) -> Optional[dict]:
    """Return session dict or None if missing/expired."""
    if not token:
        return None
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM sessions WHERE token=?", (token,)
        ).fetchone()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        logout(token)
        return None
    return dict(row)


def get_session_by_api_key(api_key: str) -> Optional[dict]:
    """Resolve an API key to a synthetic session dict (for Open-WebUI / curl)."""
    if not api_key:
        return None
    with _conn() as db:
        user = db.execute(
            "SELECT * FROM users WHERE api_key=?", (api_key,)
        ).fetchone()
    if not user:
        return None
    return {
        "token": api_key,
        "user_id": user["id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "role": user["role"],
    }


def logout(token: str):
    with _conn() as db:
        db.execute("DELETE FROM sessions WHERE token=?", (token,))


def logout_all(user_id: int):
    with _conn() as db:
        db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))


# ─── User CRUD ────────────────────────────────────────────────────────────────

def _create_user_internal(db, username, full_name, email, password, role):
    salt    = secrets.token_hex(16)
    pw_hash = _hash(password, salt)
    api_key = _new_api_key()
    now     = datetime.utcnow().isoformat()
    db.execute(
        "INSERT INTO users (username,full_name,email,password_hash,salt,role,api_key,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (username, full_name, email, pw_hash, salt, role, api_key, now),
    )


def list_users() -> list[dict]:
    with _conn() as db:
        rows = db.execute(
            "SELECT id,username,full_name,email,role,api_key,created_at,last_login "
            "FROM users ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def create_user(username: str, full_name: str, email: str,
                password: str, role: str = "user") -> tuple[bool, str]:
    """Returns (success, error_message)."""
    if not username or not password:
        return False, "Username and password are required"
    try:
        with _conn() as db:
            _create_user_internal(db, username, full_name, email, password, role)
        return True, ""
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists"


def update_user(user_id: int, full_name: str, email: str, role: str):
    with _conn() as db:
        db.execute(
            "UPDATE users SET full_name=?,email=?,role=? WHERE id=?",
            (full_name, email, role, user_id),
        )


def delete_user(user_id: int) -> tuple[bool, str]:
    with _conn() as db:
        row = db.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            return False, "User not found"
        if row["username"] == "admin":
            return False, "Cannot delete the built-in admin account"
        db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        db.execute("DELETE FROM users WHERE id=?", (user_id,))
    return True, ""


def change_password(user_id: int, new_password: str):
    salt    = secrets.token_hex(16)
    pw_hash = _hash(new_password, salt)
    with _conn() as db:
        db.execute(
            "UPDATE users SET password_hash=?,salt=? WHERE id=?",
            (pw_hash, salt, user_id),
        )
    logout_all(user_id)


def regenerate_api_key(user_id: int) -> str:
    key = _new_api_key()
    with _conn() as db:
        db.execute("UPDATE users SET api_key=? WHERE id=?", (key, user_id))
    return key


# ─── Data sources ─────────────────────────────────────────────────────────────

def list_data_sources() -> list[dict]:
    with _conn() as db:
        rows = db.execute(
            "SELECT * FROM data_sources WHERE enabled=1 ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def add_data_source(path: str, label: str, added_by: str) -> tuple[bool, str]:
    """
    Register a new folder path for indexing. Returns (success, error).

    Path resolution:
    - Accepts absolute container paths  (e.g. /app/data)
    - Accepts the default data folder   (/app/data is always valid)
    - Windows host paths are NOT visible inside Docker; users must supply the
      container-internal path (see docker-compose.yml volume mounts).
    - If the path doesn't exist yet (e.g. it will be mounted later) the source
      is still registered — indexing will skip it gracefully if absent at run time.
    """
    # Normalise: strip trailing slashes / backslashes
    path = path.strip().rstrip("/\\")

    # If the path looks like a Windows absolute path (e.g. C:\...) and we're
    # running on Linux (inside Docker), translate it to the likely mount point.
    import sys
    if sys.platform != "win32" and len(path) >= 2 and path[1] == ":":
        # Best-effort: map to /app/data (the standard data mount)
        # This handles the most common case where the user pastes their host path.
        translated = "/app/data"
        import logging
        logging.getLogger(__name__).warning(
            "Windows path '%s' detected on Linux container — using '%s' instead",
            path, translated,
        )
        path = translated

    p = Path(path)
    # Warn but don't reject if the path doesn't exist yet
    if p.exists() and not p.is_dir():
        return False, f"Path exists but is not a directory: {path}"

    try:
        with _conn() as db:
            db.execute(
                "INSERT INTO data_sources (path,label,added_by,added_at) VALUES (?,?,?,?)",
                (str(p), label or p.name, added_by, datetime.utcnow().isoformat()),
            )
        return True, ""
    except sqlite3.IntegrityError:
        return False, "This path is already registered"


def remove_data_source(source_id: int):
    with _conn() as db:
        db.execute("DELETE FROM data_sources WHERE id=?", (source_id,))


def toggle_data_source(source_id: int, enabled: bool):
    with _conn() as db:
        db.execute(
            "UPDATE data_sources SET enabled=? WHERE id=?",
            (1 if enabled else 0, source_id),
        )

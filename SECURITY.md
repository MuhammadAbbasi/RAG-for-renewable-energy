# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| v2.x (current) | ✅ Yes |
| v1.x | ❌ No |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Send a description of the vulnerability to **info@a176lab.it** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any suggested fix if you have one

You will receive a response within **48 hours**. If the issue is confirmed, a fix will be released as soon as possible depending on severity.

## Scope

This system runs **entirely on-premises** with no external network calls (except to a local Ollama instance). The main attack surfaces are:

- The FastAPI HTTP API (`rag_system/api/server.py`) — authenticated with Bearer tokens
- The SQLite auth database (`db/auth.db`) — keep the `db/` directory outside web root
- PDF ingestion pipeline — malicious PDFs could exploit parsing libraries

## Security configuration

- Change the default admin password (`admin` / `admin123`) immediately after first login
- Run behind a reverse proxy (nginx/Caddy) with TLS in any multi-user deployment
- Keep `db/`, `logs/`, and `qdrant_storage/` outside any publicly accessible directory

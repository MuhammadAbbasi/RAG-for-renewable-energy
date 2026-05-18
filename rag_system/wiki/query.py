"""
query.py — NL2SQL query engine for the wiki knowledge base.

How it works
------------
1. Receive a natural-language question (Italian or English).
2. Build a prompt that describes the projects table schema + sample data.
3. Ask the LLM to generate a single SELECT SQL statement.
4. Execute it against wiki.db via store.execute_sql() (SELECT-only guard).
5. Ask the LLM to format the SQL results as a human-readable Italian answer.

The two-step LLM pattern (generate SQL → format answer) keeps each call
small and fast, and lets us validate the SQL before touching the DB.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import httpx

from rag_system import config
from rag_system.wiki import store

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Schema description injected into the SQL-generation prompt
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_DESC = """
Tabella SQLite: projects

Colonne:
  project_id      TEXT    -- cartella progetto (es. "9579 - Sicilia")
  project_name    TEXT    -- nome impianto (es. "MARGHERITO", "VILLALBA")
  type            TEXT    -- tipo: fotovoltaico | agrovoltaico | eolico | idroelettrico | rete | altro
  power_mw        REAL    -- potenza nominale AC in MW (NULL se non trovata)
  power_dc_mw     REAL    -- potenza DC/installata in MW (NULL se non distinta)
  area_ha         REAL    -- superficie in ettari
  region          TEXT    -- regione (es. "Sicilia")
  municipalities  TEXT    -- JSON array (es. '["Ramacca","Caltagirone"]') — usa LIKE per ricercare
  provinces       TEXT    -- JSON array (es. '["CT","CL"]') — usa LIKE per ricercare
  proponent       TEXT    -- committente/società proponente
  designer        TEXT    -- progettista/società di ingegneria
  procedure       TEXT    -- VIA | PAUR | Verifica assoggettabilità a VIA | AIA | Autorizzazione Unica
  procedure_refs  TEXT    -- riferimenti normativi
  status          TEXT    -- approvato | in corso VIA | in corso autorizzazione | proposta | ottemperanza
  approval_date   TEXT    -- data parere (YYYY-MM-DD)
  approval_ref    TEXT    -- riferimento decreto/parere
  grid_connection TEXT    -- tipo connessione rete
  summary         TEXT    -- descrizione breve del progetto
  docs_count      INTEGER -- numero documenti elaborati
  last_updated    TEXT    -- timestamp ultimo aggiornamento

Note importanti:
- municipalities e provinces sono stringhe JSON — usa: municipalities LIKE '%Ramacca%'
- power_mw può essere NULL per i progetti non ancora estratti
- Per sommare o confrontare MW usa: WHERE power_mw IS NOT NULL
"""

_SQL_SYSTEM = f"""Sei un esperto SQL. Dato uno schema di database e una domanda in italiano o inglese,
genera UNA SOLA query SELECT SQLite valida che risponda alla domanda.

{_SCHEMA_DESC}

REGOLE:
- Genera SOLO la query SQL, senza spiegazioni, senza markdown, senza ```
- Usa solo la tabella 'projects'
- Solo query SELECT — nessun INSERT/UPDATE/DELETE/DROP
- Per confronti su potenza usa: CAST(power_mw AS REAL)
- Per ricerche in array JSON usa LIKE '%valore%'
- Ordina i risultati in modo sensato (ORDER BY power_mw DESC per domande su potenza)
- Limita a massimo 50 righe (LIMIT 50) se non specificato
- Se la domanda non può essere risposta con questi dati, genera: SELECT 'NO_DATA' AS result
"""

_FORMAT_SYSTEM = """Sei un assistente che presenta dati strutturati in italiano chiaro.
Ricevi risultati SQL e la domanda originale. Formatta la risposta in italiano con tabelle Markdown
o elenchi puntati secondo il contesto. Sii conciso e preciso.
Se i risultati sono vuoti, di' "Nessun dato disponibile per questa query."
Se i risultati contengono 'NO_DATA', di' "Questa domanda richiede dati non disponibili nel database strutturato."
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _llm_call(system: str, user: str, max_tokens: int = 512) -> str:
    model   = config.WIKI_EXTRACT_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            "num_ctx":     4096,
        },
    }
    if "qwen3" in model.lower():
        payload["options"]["think"] = False

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:
        logger.error("Wiki query LLM call failed: %s", exc)
        return ""


def _generate_sql(question: str) -> Optional[str]:
    """Ask the LLM to generate a SELECT SQL for the question."""
    user = f"Domanda: {question}\n\nSQL:"
    raw  = _llm_call(_SQL_SYSTEM, user, max_tokens=256)

    # Strip any markdown fences
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()

    # Extract first SQL statement
    match = re.search(r"(SELECT\s+.+?)(?:;|$)", raw, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        # Safety: strip trailing semicolon, ensure no dangerous keywords
        sql = sql.rstrip(";").strip()
        return sql

    logger.warning("Could not extract SQL from: %.200s", raw)
    return None


def _format_results(question: str, sql: str, rows: list[dict]) -> str:
    """Ask the LLM to format SQL results as a human-readable Italian answer."""
    rows_preview = rows[:30]  # limit context
    user = (
        f"Domanda originale: {question}\n\n"
        f"SQL eseguita: {sql}\n\n"
        f"Risultati ({len(rows)} righe):\n"
        f"{json.dumps(rows_preview, ensure_ascii=False, indent=2)}\n\n"
        "Rispondi in italiano:"
    )
    answer = _llm_call(_FORMAT_SYSTEM, user, max_tokens=1024)
    return answer or "Errore nella formattazione della risposta."


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def wiki_query(question: str) -> dict:
    """
    Execute a full wiki NL2SQL pipeline for a question.

    Returns:
        {
            "answer":   str,          # formatted Italian answer
            "sql":      str,          # generated SQL (for transparency)
            "rows":     list[dict],   # raw SQL results
            "source":   "wiki",
            "error":    str | None,
        }
    """
    # Step 1: Generate SQL
    sql = _generate_sql(question)
    if not sql:
        return {
            "answer": "Non riesco a generare una query SQL per questa domanda.",
            "sql":    "",
            "rows":   [],
            "source": "wiki",
            "error":  "sql_generation_failed",
        }

    logger.info("Wiki NL2SQL: %s → %s", question[:60], sql[:80])

    # Step 2: Execute SQL
    try:
        rows = store.execute_sql(sql)
    except ValueError as exc:
        logger.warning("Wiki SQL safety guard triggered: %s", exc)
        return {
            "answer": f"Query non consentita: {exc}",
            "sql":    sql,
            "rows":   [],
            "source": "wiki",
            "error":  "sql_forbidden",
        }
    except Exception as exc:
        logger.error("Wiki SQL execution error: %s — SQL: %s", exc, sql)
        return {
            "answer": f"Errore nell'esecuzione della query: {exc}",
            "sql":    sql,
            "rows":   [],
            "source": "wiki",
            "error":  "sql_error",
        }

    # Step 3: Format results
    answer = _format_results(question, sql, rows)

    return {
        "answer": answer,
        "sql":    sql,
        "rows":   rows,
        "source": "wiki",
        "error":  None,
    }


def wiki_query_direct(sql: str) -> list[dict]:
    """Execute a raw SQL query directly (for API/debug use). SELECT only."""
    return store.execute_sql(sql)

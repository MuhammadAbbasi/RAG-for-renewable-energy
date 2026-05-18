"""
extractor.py — LLM-based structured extraction from PDF documents.

Strategy
--------
1. Extract text from the first WIKI_EXTRACT_PAGES pages via pdfplumber (fast, CPU).
2. If the text is too sparse (scanned page), try up to 3 more pages.
3. Send the extracted text to the LLM with a strict JSON-output prompt.
4. Parse the JSON, build a ProjectRecord, merge into wiki.db via store.upsert_project().

The extraction model is intentionally separate from the answering model:
  - config.WIKI_EXTRACT_MODEL  (default: same as LLM_MODEL)
  - A smaller/faster model (e.g. qwen2.5:7b) works well for structured extraction.

Call pattern (from pipeline.py after step 6):
    from rag_system.wiki import extractor
    extractor.extract_and_store(pdf_path, project_id)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import httpx

from rag_system import config
from rag_system.wiki import store
from rag_system.wiki.schema import ProjectRecord, DocExtraction, infer_doc_type

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Extraction prompt
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """Sei un assistente specializzato nell'estrazione di dati strutturati da documenti tecnici italiani per progetti di energia rinnovabile.
Rispondi SOLO con un oggetto JSON valido. Non aggiungere testo prima o dopo il JSON. Non usare markdown.

Estrai questi campi (usa null se non trovato):
{
  "project_name": "string — nome dell'impianto o del progetto (es. VILLALBA, MARGHERITO)",
  "type": "string — fotovoltaico | agrovoltaico | eolico | idroelettrico | accumulo | rete | altro",
  "power_mw": number — potenza nominale AC in MW (converti kW/1000, ignora potenza DC se diversa),
  "power_dc_mw": number — potenza DC o installata in MW se esplicitamente distinta,
  "area_ha": number — superficie totale area di progetto in ettari,
  "municipalities": ["string"] — lista dei comuni interessati,
  "provinces": ["string"] — sigle province (es. CT, CL, RG),
  "region": "string — regione (es. Sicilia)",
  "proponent": "string — nome del committente/proponente/società richiedente",
  "designer": "string — nome del progettista/società di ingegneria",
  "procedure": "string — tipo procedura: VIA | PAUR | Verifica assoggettabilità a VIA | AIA | Autorizzazione Unica | altro",
  "procedure_refs": "string — riferimenti normativi citati (es. art. 19 D.Lgs. 152/2006)",
  "status": "string — approvato | in corso VIA | in corso autorizzazione | proposta | ottemperanza | altro",
  "approval_date": "string — data parere/decreto in formato YYYY-MM-DD, null se non presente",
  "approval_ref": "string — riferimento parere o decreto (es. parere n. 255 del 25/01/2024)",
  "grid_connection": "string — tipo connessione rete (es. RTN 150kV, AT 36kV, MT 20kV)",
  "doc_type": "string — SPA | SIA | VIA | SNT | PMA | RT | VINCA | MASE | altro",
  "summary": "string — descrizione del progetto in 2-3 frasi (tipo, potenza, luogo, proponente)"
}"""

_EXTRACT_USER = """Documento: {filename}
Cartella progetto: {project_id}

--- TESTO PRIME PAGINE ---
{text}
--- FINE TESTO ---

Estrai i dati strutturati. Rispondi solo con JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pages_text(pdf_path: Path, max_pages: int) -> str:
    """
    Extract text from the first max_pages pages using pdfplumber.
    If the first page is sparse (scanned), tries up to 3 additional pages.
    Returns concatenated text, capped at 6000 chars.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not available — wiki extraction skipped for %s", pdf_path.name)
        return ""

    text_parts = []
    total_chars = 0

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            n_pages = len(pdf.pages)
            target  = min(max_pages + 3, n_pages)  # read a few extra to skip blank pages

            for i in range(target):
                if total_chars >= 6000:
                    break
                try:
                    page_text = pdf.pages[i].extract_text() or ""
                    page_text = page_text.strip()
                    if page_text:
                        header = f"\n[Pagina {i+1}]\n"
                        text_parts.append(header + page_text)
                        total_chars += len(page_text)
                        if i < max_pages and total_chars >= 800:
                            # Have enough from first N pages — only continue for sparse docs
                            pass
                except Exception as exc:
                    logger.debug("Page %d extraction error in %s: %s", i+1, pdf_path.name, exc)
                    continue
    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s", pdf_path.name, exc)
        return ""

    full = "\n".join(text_parts)
    return full[:6000]


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm_extract(filename: str, project_id: str, text: str) -> Optional[dict]:
    """
    Call the extraction LLM with the page text.
    Returns parsed dict or None on failure.
    """
    if not text or len(text.strip()) < 100:
        logger.debug("Insufficient text for wiki extraction of %s — skipping LLM call", filename)
        return None

    model   = config.WIKI_EXTRACT_MODEL
    user_msg = _EXTRACT_USER.format(
        filename=filename,
        project_id=project_id,
        text=text,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
            "num_ctx":     8192,
        },
    }

    # Disable thinking for extraction — we want deterministic JSON, not reasoning
    if "qwen3" in model.lower():
        payload["options"]["think"] = False

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
    except Exception as exc:
        logger.warning("Wiki LLM call failed for %s: %s", filename, exc)
        return None

    # Strip markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()

    # Find the JSON object
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        logger.warning("No JSON found in wiki extraction response for %s", filename)
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error for %s: %s — raw: %.200s", filename, exc, content)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Record builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_record(project_id: str, filename: str, data: dict) -> ProjectRecord:
    """Convert raw LLM JSON dict → ProjectRecord, with type coercion."""

    def _float(val) -> Optional[float]:
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _str(val) -> Optional[str]:
        if val is None or val == "":
            return None
        return str(val).strip() or None

    def _list(val) -> list[str]:
        if not val:
            return []
        if isinstance(val, list):
            return [str(v).strip() for v in val if v]
        if isinstance(val, str):
            # Handle comma-separated strings
            return [v.strip() for v in val.split(",") if v.strip()]
        return []

    power_mw = _float(data.get("power_mw"))

    return ProjectRecord(
        project_id      = project_id,
        project_name    = _str(data.get("project_name")),
        type            = _str(data.get("type")),
        summary         = _str(data.get("summary")),
        power_mw        = power_mw,
        power_dc_mw     = _float(data.get("power_dc_mw")),
        area_ha         = _float(data.get("area_ha")),
        power_source    = filename if power_mw is not None else None,
        region          = _str(data.get("region")),
        municipalities  = _list(data.get("municipalities")),
        provinces       = _list(data.get("provinces")),
        proponent       = _str(data.get("proponent")),
        designer        = _str(data.get("designer")),
        procedure       = _str(data.get("procedure")),
        procedure_refs  = _str(data.get("procedure_refs")),
        status          = _str(data.get("status")),
        approval_date   = _str(data.get("approval_date")),
        approval_ref    = _str(data.get("approval_ref")),
        grid_connection = _str(data.get("grid_connection")),
        docs_count      = 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_store(pdf_path: Path, project_id: str, force: bool = False) -> bool:
    """
    Run wiki extraction for one PDF file:
      1. Skip if already extracted (unless force=True).
      2. Extract text from first WIKI_EXTRACT_PAGES pages.
      3. Call LLM → parse JSON.
      4. Merge ProjectRecord into projects table.
      5. Save DocExtraction audit record.

    Returns True if extraction produced data, False otherwise.
    """
    if not config.WIKI_ENABLED:
        return False

    filename = pdf_path.name

    if not force and store.is_doc_extracted(project_id, filename):
        logger.debug("Wiki: already extracted %s — skipping", filename)
        return False

    t0 = time.perf_counter()

    # Step 1: Extract page text
    text = _extract_pages_text(pdf_path, max_pages=config.WIKI_EXTRACT_PAGES)

    # Step 2: LLM extraction
    data = _call_llm_extract(filename, project_id, text)

    elapsed = time.perf_counter() - t0

    # Step 3: Infer doc_type from filename (as fallback/supplement)
    doc_type = infer_doc_type(filename)
    if data and data.get("doc_type"):
        doc_type = data["doc_type"]

    # Step 4: Save audit record regardless of outcome
    extraction = DocExtraction(
        project_id  = project_id,
        filename    = filename,
        doc_type    = doc_type,
        extracted   = json.dumps(data, ensure_ascii=False) if data else None,
    )
    store.save_doc_extraction(extraction)

    if data is None:
        logger.info(
            "  Wiki: no data extracted from %s (%.1fs) — text_len=%d",
            filename, elapsed, len(text),
        )
        return False

    # Step 5: Build record and merge into projects table
    record = _build_record(project_id, filename, data)
    store.upsert_project(record)

    logger.info(
        "  Wiki: extracted %s → name=%s mw=%.1f (%.1fs)",
        filename,
        record.project_name or "?",
        record.power_mw or 0,
        elapsed,
    )
    return True

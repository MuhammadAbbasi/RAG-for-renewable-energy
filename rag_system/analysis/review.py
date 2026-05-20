"""
review.py — Cross-project comparative review engine for new proposals.

Given a new proposal document (PDF bytes or raw text), this engine:

  1. Extracts and understands the proposal content
  2. Searches ALL indexed projects for:
       - Failure patterns  (prescriptions issued, integrations demanded, rejections)
       - Success patterns  (approved projects, complete docs, best practices)
       - Common requirements for that project type / region
  3. Compares the proposal against findings
  4. Streams a consultant-grade comparative report ending with
     clarifying questions for multi-turn dialogue

The report structure:
  ## 🔎 Comprensione del documento
  ## 📊 Confronto con progetti simili nel database
  ## ❌ Errori comuni che hanno causato problemi
  ## ✅ Elementi sempre presenti nei progetti approvati
  ## 📋 Analisi del tuo documento
  ## ⚠️ Problemi rilevati nel tuo documento
  ## 📌 Raccomandazioni prioritarie
  ## ❓ Domande di approfondimento   ← triggers follow-up UI buttons
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from rag_system.retrieval.retriever import retrieve, format_context, get_sources
from rag_system import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Comparative probe queries — run across ALL projects (no project_id filter)
# ─────────────────────────────────────────────────────────────────────────────

FAILURE_PROBES = [
    {"id": "prescriptions",
     "label": "Prescrizioni imposte",
     "query": "prescrizioni condizioni aggiuntive imposte decreto autorizzazione richieste"},

    {"id": "integrations",
     "label": "Integrazioni documentali richieste",
     "query": "richiesta integrazione documentazione aggiuntiva incompleta carente mancante"},

    {"id": "negative",
     "label": "Esiti negativi / sospensioni",
     "query": "rigetto diniego parere negativo improcedibile sospeso archiviato non approvato"},

    {"id": "common_gaps",
     "label": "Lacune documentali frequenti",
     "query": "documento mancante assente non allegato non presente non prodotto incompleto"},

    {"id": "vinca_issues",
     "label": "Problemi VINCA / Natura 2000",
     "query": "valutazione incidenza negativa habitat specie protette impatto significativo ZSC ZPS"},

    {"id": "grid_issues",
     "label": "Problemi connessione rete",
     "query": "connessione rete rigettata non fattibile soluzione tecnica alternativa cavidotto problemi"},
]

SUCCESS_PROBES = [
    {"id": "approved_projects",
     "label": "Progetti approvati",
     "query": "approvato decreto VIA positivo autorizzazione unica rilasciata favorevole"},

    {"id": "complete_sia",
     "label": "SIA completo e approvato",
     "query": "studio impatto ambientale completo esaustivo approvato quadro progettuale ambientale"},

    {"id": "good_mitigation",
     "label": "Misure di mitigazione efficaci",
     "query": "misure mitigazione compensazione efficace accettata approvata impatto ridotto"},

    {"id": "good_monitoring",
     "label": "Piano di monitoraggio approvato",
     "query": "piano monitoraggio ambientale PMA approvato misure parametri indicatori"},

    {"id": "good_grid",
     "label": "Connessione rete approvata",
     "query": "soluzione tecnica connessione approvata STMG accettata Terna GSE"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Proposal analysis angles — searched inside the uploaded doc's extracted text
# by using that text as additional context (not as a query to the DB)
# ─────────────────────────────────────────────────────────────────────────────

PROPOSAL_PROBES = [
    {"id": "identity",
     "query": "tipo impianto potenza MW localizzazione comune provincia proponente committente"},
    {"id": "env_impact",
     "query": "impatto ambientale misure mitigazione compensazione biodiversità fauna flora"},
    {"id": "grid",
     "query": "connessione rete elettrica soluzione tecnica STMG cavidotto AT MT"},
    {"id": "land",
     "query": "vincoli paesaggistici urbanistici PRG destinazione suolo compatibilità"},
    {"id": "vinca",
     "query": "valutazione incidenza Natura 2000 habitat ZSC ZPS SIC flora fauna"},
    {"id": "tech",
     "query": "relazione tecnica layout planimetria schema elettrico moduli inverter"},
    {"id": "timeline",
     "query": "cronoprogramma cantiere costruzione collaudo fine lavori"},
    {"id": "decommission",
     "query": "dismissione smontaggio ripristino garanzie finanziarie fideiussione"},
]

# ─────────────────────────────────────────────────────────────────────────────
# LLM system prompt
# ─────────────────────────────────────────────────────────────────────────────

_REVIEW_SYSTEM = """Sei un consulente senior specializzato in autorizzazioni per impianti rinnovabili in Italia (VIA, PAUR, AU).
Hai analizzato decine di progetti fotovoltaici, agrovoltaici ed eolici.

Il tuo compito: confrontare una NUOVA PROPOSTA con i precedenti del database (sia successi che fallimenti)
e fornire una revisione critica che aiuti il proponente a migliorarla PRIMA di presentarla.

REGOLE:
- Scrivi in italiano, tono consulenziale diretto.
- Sii SPECIFICO: cita valori, nomi di documenti, enti, articoli normativi se presenti nel contesto.
- Confronta ESPLICITAMENTE: "nel database X progetti hanno avuto il problema Y", "i progetti approvati avevano sempre Z".
- Non inventare dati non presenti nel contesto.
- Sii CRITICO ma costruttivo.

FORMATO OBBLIGATORIO (usa esattamente questi header Markdown):

## 🔎 Comprensione del documento proposto
[Riassumi: tipo impianto, potenza, localizzazione, procedura applicabile, proponente]

## 📊 Confronto con il database progetti
[Quanti progetti simili esistono, quali procedure usano, range di potenza comuni]

## ❌ Errori e problemi ricorrenti nei progetti simili
[Lista numerata — problemi che hanno causato prescrizioni, integrazioni o rigetti in passato]

## ✅ Elementi sempre presenti nei progetti approvati
[Lista — cosa hanno in comune i progetti che hanno ottenuto l'autorizzazione]

## 📋 Valutazione del documento proposto
[Tabella markdown: Aspetto | Stato nel documento | Giudizio (✅/⚠️/❌) | Note]

## ⚠️ Problemi critici rilevati nel tuo documento
[Lista numerata con priorità alta — cosa potrebbe causare rigetto o richiesta integrazioni]

## 📌 Raccomandazioni prioritarie
[Azioni concrete ordinate per impatto — cosa fare SUBITO prima della presentazione]

## ❓ Domande di approfondimento
[Esattamente 3 domande specifiche che ti aiuterebbero a dare consigli ancora più precisi.
Formato: "1. [domanda]" — una per riga]
"""


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes, max_chars: int = 40000) -> str:
    """Extract plain text from PDF bytes using pdfplumber."""
    try:
        import pdfplumber
        text_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t.strip())
        full = "\n\n".join(text_parts)
        if len(full) > max_chars:
            full = full[:max_chars] + "\n\n[... documento troncato per limiti di contesto ...]"
        logger.info("PDF extraction: %d chars from %d pages", len(full), len(text_parts))
        return full
    except Exception as exc:
        logger.error("PDF text extraction failed: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_proposal(
    proposal_text: str,
    project_type:  str = "",
    region:        str = "",
    concerns:      str = "",
    model:         Optional[str] = None,
    top_k:         int = 8,
) -> dict:
    """
    Run comparative analysis of a new proposal against the full project database.

    Returns a dict with synthesis_prompt, all_chunks, model.
    """
    model = model or config.LLM_MODEL
    logger.info("Starting proposal review: type=%s region=%s text_len=%d",
                project_type, region, len(proposal_text))

    all_chunks: list[dict] = []
    failure_ctx: dict[str, str] = {}
    success_ctx: dict[str, str] = {}
    proposal_db_ctx: dict[str, str] = {}

    # ── Step 1: Search DB for failure patterns (all projects, no filter) ──────
    for probe in FAILURE_PROBES:
        q = probe["query"]
        if project_type:
            q += f" {project_type}"
        results = retrieve(q, project_id=None, top_k=top_k)
        failure_ctx[probe["id"]] = format_context(results)
        all_chunks.extend(results)
        logger.debug("  failure[%s]: %d chunks", probe["id"], len(results))

    # ── Step 2: Search DB for success patterns ────────────────────────────────
    for probe in SUCCESS_PROBES:
        q = probe["query"]
        if project_type:
            q += f" {project_type}"
        results = retrieve(q, project_id=None, top_k=top_k)
        success_ctx[probe["id"]] = format_context(results)
        all_chunks.extend(results)
        logger.debug("  success[%s]: %d chunks", probe["id"], len(results))

    # ── Step 3: Search DB using proposal-specific angles ──────────────────────
    # Use the first 500 chars of each probe topic augmented with proposal keywords
    proposal_keywords = _extract_keywords(proposal_text)
    for probe in PROPOSAL_PROBES:
        q = probe["query"] + " " + proposal_keywords
        results = retrieve(q, project_id=None, top_k=6)
        proposal_db_ctx[probe["id"]] = format_context(results)
        all_chunks.extend(results)

    # ── Step 4: Assemble synthesis prompt ─────────────────────────────────────
    prompt = _build_review_prompt(
        proposal_text=proposal_text,
        project_type=project_type,
        region=region,
        concerns=concerns,
        failure_ctx=failure_ctx,
        success_ctx=success_ctx,
        proposal_db_ctx=proposal_db_ctx,
    )

    return {
        "synthesis_prompt": prompt,
        "all_chunks":       all_chunks,
        "model":            model,
    }


def stream_review(analysis: dict):
    """
    Stream the LLM synthesis. Same token/sources protocol as chain.py.
    """
    import json
    import httpx

    model   = analysis["model"]
    prompt  = analysis["synthesis_prompt"]
    sources = get_sources(analysis["all_chunks"])

    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": _REVIEW_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "stream": True,
        "options": {
            "temperature": 0.25,
            "num_predict": config.LLM_MAX_TOKENS,
            "num_ctx":     config.LLM_CONTEXT_SIZE,
        },
    }

    try:
        with httpx.Client(timeout=600.0) as client:
            with client.stream("POST", f"{config.OLLAMA_BASE_URL}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data  = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.error("Review LLM streaming failed: %s", exc)
        yield f"\n[Errore streaming: {exc}]"

    yield "\n\n__SOURCES__:" + json.dumps(sources, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keywords(text: str, max_words: int = 30) -> str:
    """Pull the most distinctive words from proposal text for DB queries."""
    import re
    # Remove common stopwords and short words
    stopwords = {
        "di","il","la","le","lo","gli","un","una","che","e","per","con","del",
        "della","dei","delle","degli","in","su","da","a","al","alla","alle",
        "agli","ai","non","si","è","ha","ho","hanno","sono","essere","avere",
        "questo","questa","questi","queste","loro","esso","essa","essi",
        "the","of","and","in","to","a","for","is","are","was","were","with",
    }
    words = re.findall(r'\b[a-zA-ZàèéìòùÀÈÉÌÒÙ]{4,}\b', text.lower())
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        if w not in stopwords and w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= max_words:
            break
    return " ".join(result)


def _build_review_prompt(
    proposal_text: str,
    project_type:  str,
    region:        str,
    concerns:      str,
    failure_ctx:   dict[str, str],
    success_ctx:   dict[str, str],
    proposal_db_ctx: dict[str, str],
) -> str:
    parts: list[str] = []

    # Header
    parts.append("=== NUOVO DOCUMENTO DA RIVEDERE ===")
    if project_type:
        parts.append(f"Tipo impianto dichiarato: {project_type}")
    if region:
        parts.append(f"Regione/Provincia: {region}")
    if concerns:
        parts.append(f"Preoccupazioni del proponente: {concerns}")

    # The proposal itself (capped)
    proposal_excerpt = proposal_text[:12000] if len(proposal_text) > 12000 else proposal_text
    parts.append("\n[TESTO DEL DOCUMENTO PROPOSTO]")
    parts.append(proposal_excerpt)
    parts.append("[FINE DOCUMENTO]")

    # Failure patterns from DB
    parts.append("\n=== PATTERN DI FALLIMENTO DAL DATABASE PROGETTI ===")
    for probe in FAILURE_PROBES:
        ctx = failure_ctx.get(probe["id"], "").strip()
        if ctx:
            parts.append(f"\n[{probe['label'].upper()}]")
            parts.append(ctx[:2500])

    # Success patterns from DB
    parts.append("\n=== PATTERN DI SUCCESSO DAL DATABASE PROGETTI ===")
    for probe in SUCCESS_PROBES:
        ctx = success_ctx.get(probe["id"], "").strip()
        if ctx:
            parts.append(f"\n[{probe['label'].upper()}]")
            parts.append(ctx[:2000])

    # Proposal-specific DB context
    parts.append("\n=== CONTESTO DATABASE SPECIFICO PER QUESTO DOCUMENTO ===")
    for probe in PROPOSAL_PROBES:
        ctx = proposal_db_ctx.get(probe["id"], "").strip()
        if ctx:
            parts.append(f"\n[SIMILI NEL DB — {probe['id'].upper()}]")
            parts.append(ctx[:1500])

    parts.append("\n---")
    parts.append(
        "Produci ora il rapporto completo nel formato richiesto. "
        "Confronta ESPLICITAMENTE il documento proposto con i pattern del database. "
        "Termina SEMPRE con la sezione '## ❓ Domande di approfondimento' con esattamente 3 domande."
    )

    return "\n".join(parts)

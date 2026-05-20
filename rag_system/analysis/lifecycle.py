"""
lifecycle.py — Project permit-lifecycle analysis engine.

Runs a structured multi-step RAG interrogation of a single project's
documents and synthesises a consultant-grade report that tells the user:

  1. What the project IS (type, power, location)
  2. Which permit procedure applies (VIA / PAUR / AU)
  3. What documents are present and what is missing
  4. What requirements / prescrizioni have been identified
  5. What has changed between document versions
  6. What the project is doing WRONG or is at risk of

Designed to be called from the /v1/process endpoint.
The analysis is returned as a structured dict; the endpoint streams
the final LLM narrative.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag_system.retrieval.retriever import retrieve, format_context
from rag_system import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Italian renewable-energy permit checklist
# These are the document/topic categories we probe for in every project.
# ─────────────────────────────────────────────────────────────────────────────

CHECKLIST: list[dict] = [
    # ── Identity ──────────────────────────────────────────────────────────────
    {"id": "identity",       "label": "Identità progetto",
     "query": "nome progetto proponente committente tipo impianto potenza MW localizzazione comune provincia"},

    # ── Permit procedure ──────────────────────────────────────────────────────
    {"id": "procedure",      "label": "Procedura autorizzativa",
     "query": "procedura autorizzativa VIA PAUR autorizzazione unica DEC decreto ministeriale screening"},

    # ── Environmental study ───────────────────────────────────────────────────
    {"id": "via",            "label": "Studio di Impatto Ambientale (SIA/VIA)",
     "query": "studio impatto ambientale SIA quadro progettuale ambientale programmatico VIA"},

    # ── Technical documents ───────────────────────────────────────────────────
    {"id": "technical",      "label": "Relazione tecnica e progetto definitivo",
     "query": "relazione tecnica progetto definitivo layout planimetria schema elettrico sezioni"},

    # ── Grid connection ───────────────────────────────────────────────────────
    {"id": "grid",           "label": "Connessione alla rete elettrica",
     "query": "connessione rete elettrica soluzione tecnica STMG Terna GSE cavidotto AT MT"},

    # ── Land use / compatibility ───────────────────────────────────────────────
    {"id": "land",           "label": "Compatibilità urbanistica e paesaggistica",
     "query": "vincoli urbanistici paesaggistici PRG destinazione uso suolo compatibilità"},

    # ── Flora / fauna / Vinca ────────────────────────────────────────────────
    {"id": "vinca",          "label": "Valutazione di Incidenza (VINCA)",
     "query": "valutazione incidenza VINCA Natura 2000 habitat specie flora fauna ZSC ZPS SIC"},

    # ── Prescriptions / conditions ─────────────────────────────────────────────
    {"id": "prescriptions",  "label": "Prescrizioni e condizioni autorizzative",
     "query": "prescrizioni condizioni autorizzazione richieste integrazioni ministeriali"},

    # ── Document versions / changes ───────────────────────────────────────────
    {"id": "versions",       "label": "Varianti e aggiornamenti documentali",
     "query": "variante modifica aggiornamento revisione integrazione documentazione risposta osservazioni"},

    # ── Timeline / schedule ───────────────────────────────────────────────────
    {"id": "timeline",       "label": "Cronoprogramma e scadenze",
     "query": "cronoprogramma tempi realizzazione cantiere scadenze costruzione collaudo"},

    # ── Decommissioning / bonds ───────────────────────────────────────────────
    {"id": "decommission",   "label": "Piano di dismissione e garanzie finanziarie",
     "query": "dismissione smontaggio ripristino garanzie finanziarie fideiussione fine vita"},

    # ── Agricultural / agrivoltaic specifics ─────────────────────────────────
    {"id": "agrivoltaic",    "label": "Aspetti agrovoltaici / agricoli",
     "query": "agrovoltaico attività agricola monitoraggio colture pascolo integrazione"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Known required-document types per procedure
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_DOCS = {
    "VIA": [
        "Studio di Impatto Ambientale (SIA)",
        "Progetto definitivo",
        "Relazione tecnica descrittiva",
        "Planimetria generale layout",
        "Schema elettrico unifilare",
        "Soluzione tecnica connessione rete (STMG)",
        "Relazione paesaggistica",
        "Valutazione di Incidenza (se area Natura 2000)",
        "Piano di monitoraggio ambientale (PMA)",
        "Piano di dismissione e ripristino",
        "Garanzie finanziarie / fideiussione",
    ],
    "PAUR": [
        "Studio di Impatto Ambientale (SIA)",
        "Istanza autorizzazione unica",
        "Progetto definitivo",
        "Relazione tecnica",
        "Schema elettrico",
        "STMG / soluzione connessione",
        "Valutazione di Incidenza",
        "Relazione paesaggistica",
        "Piano dismissione",
    ],
    "AU": [
        "Istanza autorizzazione unica",
        "Progetto definitivo",
        "Relazione tecnica",
        "Schema elettrico unifilare",
        "STMG",
        "Pareri preliminari enti",
        "Piano dismissione",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# System prompt for the synthesis LLM call
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """Sei un consulente esperto in autorizzazioni per impianti rinnovabili in Italia.
Il tuo compito è analizzare i documenti di un progetto e produrre un rapporto critico e dettagliato.

ISTRUZIONI:
- Scrivi in italiano, tono professionale e diretto.
- Sii specifico: cita nomi di documenti, valori numerici, date, enti coinvolti.
- NON generare informazioni non presenti nel contesto.
- Evidenzia CHIARAMENTE i problemi, le lacune e i rischi.
- Usa il formato richiesto con sezioni Markdown.

FORMATO RISPOSTA OBBLIGATORIO:

## 📋 Scheda progetto
[tipo, potenza, localizzazione, proponente, procedura identificata]

## 📁 Inventario documentale
[tabella: categoria | stato (✅ presente / ⚠️ parziale / ❌ mancante)]

## ⚠️ Problemi identificati
[lista numerata dei problemi critici, con documento di riferimento]

## 🔴 Documenti mancanti o incompleti
[lista di documenti richiesti dalla procedura ma non trovati]

## 🟡 Rischi e osservazioni
[aspetti che potrebbero causare ritardi o richieste di integrazione]

## ✅ Elementi in regola
[cosa è presente e conforme]

## 📌 Raccomandazioni prioritarie
[azioni concrete da intraprendere, in ordine di priorità]
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyse_project(
    project_id: str,
    current_phase: str = "unknown",
    main_concern: str = "",
    known_issues: str = "",
    model: Optional[str] = None,
    top_k_per_query: int = 8,
) -> dict:
    """
    Run a full lifecycle analysis on *project_id*.

    Returns a dict with:
      - sections: {checklist_id: {"label": str, "context": str, "chunks": list}}
      - procedure: detected permit procedure (VIA / PAUR / AU / unknown)
      - synthesis_prompt: the assembled user message for the LLM
    """
    model = model or config.LLM_MODEL
    logger.info("Starting lifecycle analysis for project=%s phase=%s", project_id, current_phase)

    sections: dict[str, dict] = {}
    all_chunks: list[dict] = []

    # Step 1: probe each checklist category
    for item in CHECKLIST:
        results = retrieve(item["query"], project_id=project_id, top_k=top_k_per_query)
        ctx = format_context(results)
        sections[item["id"]] = {
            "label":   item["label"],
            "context": ctx,
            "found":   bool(results),
            "chunks":  results,
        }
        all_chunks.extend(results)
        logger.debug("  [%s] retrieved %d chunks", item["id"], len(results))

    # Step 2: detect permit procedure from the "procedure" section
    procedure = _detect_procedure(sections.get("procedure", {}).get("context", ""))

    # Step 3: build the synthesis prompt
    prompt = _build_synthesis_prompt(
        project_id=project_id,
        sections=sections,
        procedure=procedure,
        current_phase=current_phase,
        main_concern=main_concern,
        known_issues=known_issues,
    )

    return {
        "project_id":       project_id,
        "procedure":        procedure,
        "sections":         sections,
        "synthesis_prompt": prompt,
        "model":            model,
        "all_chunks":       all_chunks,
    }


def stream_analysis(analysis: dict):
    """
    Stream the LLM synthesis for a pre-built analysis dict.
    Yields text tokens then a __SOURCES__ sentinel (same protocol as chain.py).
    """
    import json
    import httpx
    from rag_system.retrieval.retriever import get_sources

    model   = analysis["model"]
    prompt  = analysis["synthesis_prompt"]
    sources = get_sources(analysis["all_chunks"])

    payload = {
        "model":   model,
        "messages": [
            {"role": "system",  "content": _SYNTHESIS_SYSTEM},
            {"role": "user",    "content": prompt},
        ],
        "stream": True,
        "options": {
            "temperature": 0.2,
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
        logger.error("Lifecycle LLM streaming failed: %s", exc)
        yield f"\n[Errore streaming: {exc}]"

    yield "\n\n__SOURCES__:" + json.dumps(sources, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_procedure(procedure_context: str) -> str:
    """Infer permit procedure from retrieved text."""
    txt = procedure_context.lower()
    if "paur" in txt:
        return "PAUR"
    if "autorizzazione unica" in txt and "via" in txt:
        return "PAUR"
    if "via" in txt and ("decreto" in txt or "ministeriale" in txt or "mite" in txt or "mase" in txt):
        return "VIA"
    if "autorizzazione unica" in txt:
        return "AU"
    if "screening" in txt:
        return "Screening VIA"
    return "Non determinata"


def _build_synthesis_prompt(
    project_id: str,
    sections: dict,
    procedure: str,
    current_phase: str,
    main_concern: str,
    known_issues: str,
) -> str:
    """Assemble the full context block for the LLM synthesis call."""
    parts: list[str] = []

    parts.append(f"PROGETTO: {project_id}")
    parts.append(f"FASE DICHIARATA: {current_phase}")
    parts.append(f"PROCEDURA RILEVATA: {procedure}")
    if main_concern:
        parts.append(f"PREOCCUPAZIONE PRINCIPALE: {main_concern}")
    if known_issues:
        parts.append(f"PROBLEMI GIÀ NOTI: {known_issues}")

    # Required docs checklist for the detected procedure
    req = REQUIRED_DOCS.get(procedure.split()[0] if procedure != "Non determinata" else "", [])
    if req:
        parts.append("\nDOCUMENTI RICHIESTI DALLA PROCEDURA " + procedure + ":")
        for d in req:
            parts.append(f"  - {d}")

    # Each checklist section as a context block
    for item in CHECKLIST:
        sec = sections.get(item["id"], {})
        ctx = sec.get("context", "").strip()
        if ctx:
            parts.append(f"\n[SEZIONE: {item['label'].upper()}]")
            parts.append(ctx[:3000])  # cap per section to stay within context window

    parts.append("\n---")
    parts.append("Sulla base di TUTTO il contesto sopra, produci il rapporto completo nel formato richiesto.")
    parts.append("Sii CRITICO e SPECIFICO. Se un documento obbligatorio non appare nel contesto, segnalalo come MANCANTE.")

    return "\n".join(parts)

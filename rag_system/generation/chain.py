"""
chain.py - RAG answer generation via Ollama.

Supports both blocking and streaming generation.
Query routing: WIKI (NL2SQL), HYBRID (wiki+RAG), RAG (default vector search).
"""

from __future__ import annotations

import json
import logging
from typing import Iterator, Optional

import httpx

from rag_system import config
from rag_system.retrieval.retriever import retrieve, format_context, get_sources

try:
    from rag_system.wiki.router import route, QueryRoute
    from rag_system.wiki.query import wiki_query
    _WIKI_AVAILABLE = True
except Exception:
    _WIKI_AVAILABLE = False
    class QueryRoute:
        RAG = "rag"
        WIKI = "wiki"
        HYBRID = "hybrid"
    def route(q): return QueryRoute.RAG

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Sei un assistente esperto nell'analisi di documenti tecnici e normativi italiani.
Rispondi SOLO in italiano, in modo chiaro, preciso e tecnico.

ISTRUZIONI FONDAMENTALI:
- Basa le tue risposte SUL CONTESTO fornito tra i tag [CONTESTO].
- Non inventare dati, cifre o fatti non presenti nel contesto.
- Non usare conoscenze esterne al contesto per presentare fatti come certi.
- Cita sempre il documento sorgente e il numero di pagina quando riporti dati specifici.
- Per dati numerici, date, nomi tecnici: riporta i valori esatti dal documento, senza arrotondamenti.

PER DOMANDE AGGREGATIVE (liste, confronti, panoramiche su piu' progetti):
- OBBLIGO: estrai e presenta TUTTE le informazioni rilevanti presenti nel contesto, anche se parziali.
- Cerca attivamente potenza, MW, MWp, MWe, kWp, capacita' installata, taglia impianto e termini equivalenti.
- Se il contesto copre solo alcuni progetti: elenca quelli con i dati disponibili, poi aggiungi
  "Nota: il contesto non contiene informazioni su tutti i progetti richiesti."
- Organizza la risposta in tabelle Markdown o elenchi puntati quando la domanda lo richiede.
- DIVIETO ASSOLUTO di rispondere "Non ho trovato" se nel contesto esiste QUALSIASI informazione
  parzialmente rilevante. Usa sempre quei dati parziali e indica esplicitamente cosa manca.

QUANDO RISPONDERE "Nessuna informazione trovata":
- Solo se il contesto e' completamente privo di qualsiasi dato rilevante alla domanda.
- In quel caso scrivi: "Nessuna informazione rilevante trovata nei documenti disponibili."

Formato risposta:
- Risposta diretta e strutturata (tabelle Markdown per confronti/liste numeriche).
- Citazioni in linea: (Doc: nome_file, p. N)
- Se incerti su un dato: "I documenti indicano che..." oppure "Non specificato nei documenti."
"""

_USER_TEMPLATE = """[CONTESTO]
{context}
[/CONTESTO]

Domanda: {question}

Risposta:"""


def _build_payload(system: str, user: str, model: str, stream: bool) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": stream,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_MAX_TOKENS,
            "num_ctx":     config.LLM_CONTEXT_SIZE,
        },
    }


def _call_ollama_blocking(system: str, user: str, model: str) -> str:
    payload = _build_payload(system, user, model, stream=False)
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
    except httpx.TimeoutException:
        logger.error("LLM call timed out (model=%s)", model)
        return "Errore: il modello di linguaggio ha impiegato troppo tempo."
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"Errore nella generazione della risposta: {exc}"


def _stream_rag(user_msg: str, model: str) -> Iterator[str]:
    payload = _build_payload(_SYSTEM_PROMPT, user_msg, model, stream=True)
    try:
        with httpx.Client(timeout=300.0) as client:
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
        logger.error("Streaming failed: %s", exc)
        yield f"\n[Errore streaming: {exc}]"


def _build_user_msg(context: str, question: str, history: Optional[list]) -> str:
    if history:
        hist_text = "\n".join(
            f"{'Utente' if m['role']=='user' else 'Assistente'}: {m['content']}"
            for m in history[-6:]
        )
        return "CONVERSAZIONE PRECEDENTE:\n" + hist_text + "\n\n" + _USER_TEMPLATE.format(context=context, question=question)
    return _USER_TEMPLATE.format(context=context, question=question)


def answer(
    question: str,
    project_id: Optional[str] = None,
    top_k: int = None,
    history: Optional[list] = None,
    model: Optional[str] = None,
) -> dict:
    model    = model or config.LLM_MODEL
    results  = retrieve(question, project_id=project_id, top_k=top_k)
    context  = format_context(results)
    sources  = get_sources(results)
    user_msg = _build_user_msg(context, question, history)
    answer_text = _call_ollama_blocking(_SYSTEM_PROMPT, user_msg, model)
    return {"answer": answer_text, "sources": sources, "chunks": results}


def stream_answer(
    question: str,
    project_id: Optional[str] = None,
    top_k: int = None,
    model: Optional[str] = None,
    history: Optional[list] = None,
) -> Iterator[str]:
    """
    Routing pipeline: WIKI -> NL2SQL, HYBRID -> wiki+RAG, RAG -> vector only.
    Yields text tokens then a __SOURCES__:<json> sentinel.
    """
    model = model or config.LLM_MODEL

    query_route = QueryRoute.RAG
    if _WIKI_AVAILABLE and config.WIKI_ENABLED:
        query_route = route(question)
        logger.info("Query route: %s -> %s", question[:60], query_route)

    # WIKI: pure structured query
    if query_route == QueryRoute.WIKI:
        try:
            result = wiki_query(question)
            # Only return wiki answer if it actually has rows / real content.
            # "Nessun dato" or empty answers fall through to RAG.
            rows = result.get("rows") or []
            answer_text = (result.get("answer") or "").strip()
            has_real_answer = (
                rows                                          # SQL returned rows
                and answer_text
                and not result.get("error")
                and "nessun" not in answer_text.lower()[:60]  # not an empty-result msg
                and "no data" not in answer_text.lower()[:60]
            )
            if has_real_answer:
                wiki_src = [{"source": "wiki_database", "type": "structured", "sql": result.get("sql", "")}]
                yield answer_text
                yield "\n\n__SOURCES__:" + json.dumps(wiki_src, ensure_ascii=False)
                return
            else:
                # Wiki has no data for this question - fall back to RAG
                logger.info("Wiki returned empty result, falling back to RAG for: %s", question[:60])
                query_route = QueryRoute.RAG
        except Exception as exc:
            logger.error("Wiki query failed, falling back to RAG: %s", exc)
            query_route = QueryRoute.RAG

    # HYBRID: wiki result prepended to RAG context
    if query_route == QueryRoute.HYBRID:
        wiki_context = ""
        wiki_src     = []
        try:
            result = wiki_query(question)
            if result.get("answer") and not result.get("error"):
                wiki_context = ("\n\n[DATI STRUTTURATI DAL DATABASE PROGETTI]\n"
                                + result["answer"] + "\n[FINE DATI STRUTTURATI]\n")
                wiki_src = [{"source": "wiki_database", "type": "structured", "sql": result.get("sql", "")}]
        except Exception as exc:
            logger.warning("Wiki part of hybrid query failed: %s", exc)
        results  = retrieve(question, project_id=project_id, top_k=top_k)
        context  = wiki_context + format_context(results)
        sources  = wiki_src + get_sources(results)
        user_msg = _build_user_msg(context, question, history)
        yield from _stream_rag(user_msg, model)
        yield "\n\n__SOURCES__:" + json.dumps(sources, ensure_ascii=False)
        return

    # RAG: standard vector retrieval
    results  = retrieve(question, project_id=project_id, top_k=top_k)
    context  = format_context(results)
    sources  = get_sources(results)
    user_msg = _build_user_msg(context, question, history)
    yield from _stream_rag(user_msg, model)
    yield "\n\n__SOURCES__:" + json.dumps(sources, ensure_ascii=False)

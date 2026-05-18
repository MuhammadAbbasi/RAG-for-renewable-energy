"""
router.py - Query router: wiki SQL vs RAG vector search.

Routes questions to the appropriate backend:
  WIKI  - aggregative, comparative, list, count, filter-by-value questions
  RAG   - explanatory, detail, document-content questions
  HYBRID- questions that need both (project detail + context)

Routing is pattern-based (fast, no extra LLM call). The patterns cover
Italian and English phrasing since users may query in either language.
"""

from __future__ import annotations

import re
from enum import Enum


class QueryRoute(str, Enum):
    WIKI   = "wiki"    # → wiki NL2SQL
    RAG    = "rag"     # → vector retrieval
    HYBRID = "hybrid"  # → wiki + RAG combined


# ─────────────────────────────────────────────────────────────────────────────
# Pattern lists
# ─────────────────────────────────────────────────────────────────────────────

# WIKI patterns - aggregative / structured / list / compare
_WIKI_PATTERNS = [
    # Count
    r"\bquanti\s+progetti\b",
    r"\bhow\s+many\s+projects?\b",
    r"\bnumber\s+of\s+projects?\b",
    r"\bconta\s+i\s+progetti\b",

    # List / enumerate
    r"\belenca\s+(tutti\s+i\s+)?progetti\b",
    r"\blista\s+(di\s+)?(tutti\s+i\s+)?progetti\b",
    r"\belenco\s+progetti\b",
    r"\blist\s+(all\s+)?projects?\b",
    r"\bmostra\s+(tutti\s+i\s+)?progetti\b",
    r"\bshow\s+(all\s+)?projects?\b",
    r"\bdammi\s+(la\s+lista|l[''']elenco)\b",

    # Power / MW comparisons
    r"\b(maggiore|superiore|più grande)\s+(di|a)\s+\d",
    r"\b(minore|inferiore|più piccol)\s+(di|a)\s+\d",
    r"\bgreater\s+than\s+\d",
    r"\bless\s+than\s+\d",
    r"\b>\s*\d+\s*(mw|MW|megawatt)\b",
    r"\b<\s*\d+\s*(mw|MW|megawatt)\b",
    r"\bpotenza\s+(di|del|dei)\s+progett",
    r"\bquant[io]\s+MW\b",
    r"\bhow\s+many\s+MW\b",
    r"\btotale\s+(di\s+)?MW\b",
    r"\btotal\s+(of\s+)?MW\b",
    r"\bpotenza\s+totale\b",
    r"\btotal\s+power\b",
    r"\bportfolio\b",

    # Sorting / ranking
    r"\bpiù grande\b",
    r"\bpiù piccol[oa]\b",
    r"\blargest\b",
    r"\bsmallest\b",
    r"\bmaggior[e]?\s+impianto\b",
    r"\bminor[e]?\s+impianto\b",
    r"\bordinat[io]\s+per\b",
    r"\bsort(ed)?\s+by\b",

    # Filter by attribute
    r"\bprogetti\s+(di tipo|fotovoltai|agrovoltai|eolic|idroelettr)",
    r"\bimpianti\s+(fotovoltai|agrovoltai|eolic)",
    r"\bprojects?\s+(of type|fotovoltai|agrovoltai|eolic|wind|solar)\b",
    r"\bproponente\s+(di|del|dei|per)\b",
    r"\bproponent\s+of\b",
    r"\bchiedere\s+chi\s+è\s+il\s+proponente\b",
    r"\bcommittente\b",
    r"\bdove\s+si\s+trova\b",
    r"\bwhere\s+is\b",
    r"\bcomune\s+di\b",
    r"\bmunicipality\b",
    r"\bprovincia\s+di\b",

    # Compare / summary
    r"\bconfronta\b",
    r"\bcompare\b",
    r"\bpanoramica\b",
    r"\boverview\b",
    r"\bsintesi\s+(dei\s+)?progetti\b",
    r"\bsummary\s+of\s+projects?\b",
    r"\bquali\s+progetti\b",
    r"\bwhich\s+projects?\b",

    # Status / procedure
    r"\bstatus\s+(dei\s+)?progetti\b",
    r"\bprocedura\s+(VIA|PAUR|autorizzazione)\b",
    r"\bquanti\s+(hanno|sono\s+stati)\s+(approvati|in corso)\b",
    r"\bhow\s+many\s+(are|were)\s+(approved|ongoing)\b",
]

# RAG patterns - explanatory, content-lookup, document-specific, and
# technical/construction queries that the wiki schema cannot answer.
_RAG_PATTERNS = [
    r"\bspiega\b",
    r"\bexplain\b",
    r"\bcosa\s+(dice|contiene|riporta)\b",
    r"\bwhat\s+does\b",
    r"\bdettagli\s+tecnici\b",
    r"\btechnical\s+details?\b",
    r"\bimpatto\s+ambientale\b",
    r"\benvironmental\s+impact\b",
    r"\bcome\s+funziona\b",
    r"\bhow\s+does\s+it\s+work\b",
    r"\bdocumento\b",
    r"\bdocument\b",
    r"\brelazione\b",
    r"\bstudio\b",
    r"\bpagina\b",
    r"\bpage\b",
    r"\bparagrafo\b",
    r"\bsezione\b",
    r"\bsection\b",
    r"\bdescrive\b",
    r"\bdescribes?\b",
    r"\banalisi\b",
    r"\banalysis\b",
    r"\bmonitoraggio\b",
    r"\bmonitoring\b",
    r"\bVinca\b",
    r"\bVIA\s+del\s+progetto\b",
    r"\bcondizioni\s+ambientali\b",

    # ── Technical / construction details (wiki schema has none of these) ──
    r"\bcavi?\b",                   # cable(s)
    r"\bconduttori?\b",             # conductor(s)
    r"\bconduttor[ei]\b",
    r"\bmaterial[ei]\b",            # material(s)
    r"\bdimensioni\b",              # dimensions / sizes
    r"\bdiametro\b",                # diameter
    r"\bsezione\s+\d",              # cross-section with a number e.g. "sezione 95mm²"
    r"\bmm[²2]?\b",                 # mm / mm²
    r"\bcomponenti\b",              # components
    r"\bspecifich[ea]\b",           # specifications
    r"\bfondazioni\b",              # foundations
    r"\bmisure?\b",                 # measurement(s)
    r"\bcaratteristich[ea]\b",      # characteristics
    r"\buttilizzan[oa]\b",          # "use/utilize" (present pl.)
    r"\buttilizzat[io]\b",          # used/utilized
    r"\bimpiegat[io]\b",            # employed
    r"\badottat[io]\b",             # adopted
    r"\binstallat[io]\b",           # installed
    r"\btecnologia\b",              # technology
    r"\bequipaggiament\b",          # equipment
    r"\bdispositiv\b",              # device(s)
    r"\bapparecc\b",                # apparatus
    r"\bmacchinari\b",              # machinery
    r"\bpannelli\s+fotovoltaici\b", # PV panels (technical context)
    r"\bmoduli\s+(fotovoltaici|solari)\b",
    r"\btrasformatori?\b",          # transformer(s)
    r"\binverter\b",
    r"\bgeneratori?\b",             # generator(s)
    r"\bpali\b",                    # poles / posts
    r"\bcabina\b"
    r"\bsbarre\b",                  # busbars
    r"\bcollegamento\b",            # connection detail
    r"\bisolamento\b",              # insulation
    r"\bguaina\b",                  # cable sheath
    r"\bacciaio\b",                 # steel
    r"\balluminio\b",               # aluminium
    r"\brame\b",                    # copper
    r"\btaglia\b",                  # size/rating (technical)
    r"\bportata\b",                 # current rating / capacity
    r"\btensione\b",                # voltage
    r"\bcorrente\s+di\b",          # electric current (avoids false match on "corrente" as adj)
    r"\bfrequenza\b",               # frequency
    r"\bpotenza\s+unitaria\b",     # unit power
    r"\brendimento\b",              # efficiency / yield
    r"\bperdite\b",                 # losses
]

# Compiled once at module load
_WIKI_RE  = [re.compile(p, re.IGNORECASE) for p in _WIKI_PATTERNS]
_RAG_RE   = [re.compile(p, re.IGNORECASE) for p in _RAG_PATTERNS]


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

def route(question: str) -> QueryRoute:
    """
    Determine the query route for a question.

    Returns QueryRoute.WIKI, QueryRoute.RAG, or QueryRoute.HYBRID.

    Algorithm:
      1. Count WIKI signal matches.
      2. Count RAG signal matches.
      3. If WIKI > 0 and RAG == 0 → WIKI
         If WIKI > 0 and RAG > 0  → HYBRID
         Else                      → RAG (default)
    """
    wiki_hits = sum(1 for pat in _WIKI_RE if pat.search(question))
    rag_hits  = sum(1 for pat in _RAG_RE  if pat.search(question))

    if wiki_hits > 0 and rag_hits == 0:
        return QueryRoute.WIKI
    if wiki_hits > 0 and rag_hits > 0:
        return QueryRoute.HYBRID
    return QueryRoute.RAG


def route_label(question: str) -> str:
    """Return human-readable route label for logging/debugging."""
    r = route(question)
    return {
        QueryRoute.WIKI:   "wiki (SQL)",
        QueryRoute.RAG:    "rag (vector)",
        QueryRoute.HYBRID: "hybrid (SQL + vector)",
    }[r]

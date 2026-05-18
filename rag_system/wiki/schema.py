"""
schema.py — Dataclasses for wiki extracted project records.

ProjectRecord   — one per project folder (merged from all its PDFs)
DocExtraction   — one per PDF file (raw extracted JSON, for audit)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ProjectRecord:
    """
    Aggregated structured data for one energy project (one folder in data/).
    Fields are merged across all PDF documents in the folder — the most
    specific / highest-confidence value wins on each update.
    """
    project_id:      str                       # folder name, e.g. "9579 - Sicilia"

    # Core identity
    project_name:    Optional[str]  = None     # "MARGHERITO", "VILLALBA", …
    type:            Optional[str]  = None     # fotovoltaico | agrovoltaico | eolico | …
    summary:         Optional[str]  = None     # 2-3 sentence description

    # Power / size
    power_mw:        Optional[float] = None    # nominal AC power in MW
    power_dc_mw:     Optional[float] = None    # DC / installed power in MW
    area_ha:         Optional[float] = None    # project area in hectares
    power_source:    Optional[str]  = None     # filename where power was found

    # Location
    region:          Optional[str]  = None     # "Sicilia"
    municipalities:  list[str]      = field(default_factory=list)   # ["Ramacca", …]
    provinces:       list[str]      = field(default_factory=list)   # ["CT", "CL", …]

    # Stakeholders
    proponent:       Optional[str]  = None     # committente / società proponente
    designer:        Optional[str]  = None     # progettista / società di ingegneria

    # Procedure / authorization
    procedure:       Optional[str]  = None     # VIA | PAUR | Verifica assoggettabilità | AIA
    procedure_refs:  Optional[str]  = None     # "art. 19 D.Lgs 152/2006"
    status:          Optional[str]  = None     # approvato | in corso VIA | proposta
    approval_date:   Optional[str]  = None     # ISO date string "YYYY-MM-DD"
    approval_ref:    Optional[str]  = None     # "parere n. 255 del 25/01/2024"

    # Infrastructure
    grid_connection: Optional[str]  = None     # "RTN 150kV", "MT 30kV", …

    # Bookkeeping
    docs_count:      int            = 0        # number of docs processed
    last_updated:    Optional[str]  = None     # ISO datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure list fields are stored as JSON strings for SQLite
        d["municipalities"] = json.dumps(self.municipalities, ensure_ascii=False)
        d["provinces"]      = json.dumps(self.provinces,      ensure_ascii=False)
        return d

    @classmethod
    def from_row(cls, row: dict) -> "ProjectRecord":
        """Reconstruct from a SQLite row dict."""
        r = dict(row)
        r["municipalities"] = json.loads(r.get("municipalities") or "[]")
        r["provinces"]      = json.loads(r.get("provinces")      or "[]")
        return cls(**{k: v for k, v in r.items() if k in cls.__dataclass_fields__})


@dataclass
class DocExtraction:
    """Raw extraction result for one PDF file."""
    project_id:   str
    filename:     str
    doc_type:     Optional[str]  = None   # SPA | SIA | VIA | SNT | PMA | RT | …
    extracted:    Optional[str]  = None   # JSON string of all extracted fields
    extracted_at: Optional[str]  = None   # ISO datetime


# ── Document type normalisation ──────────────────────────────────────────────

_DOC_TYPE_HINTS = {
    "SPA":  ["SPA", "STUDIO PRELIMINARE AMBIENTALE", "studio preliminare"],
    "SIA":  ["SIA", "STUDIO DI IMPATTO AMBIENTALE", "studio impatto"],
    "VIA":  ["VIA", "VALUTAZIONE DI IMPATTO AMBIENTALE"],
    "SNT":  ["SNT", "SINTESI NON TECNICA", "sintesi non tecnica"],
    "PMA":  ["PMA", "PIANO DI MONITORAGGIO AMBIENTALE", "monitoraggio ambientale"],
    "RT":   ["relazione tecnica", "RELAZIONE TECNICA", "RT_", "-RT-"],
    "VINCA":["VINCA", "Vinca", "vinca", "VALUTAZIONE INCIDENZA"],
    "MASE": ["MASE-"],
}

def infer_doc_type(filename: str) -> str:
    """Infer document type from filename heuristics."""
    upper = filename.upper()
    for dtype, hints in _DOC_TYPE_HINTS.items():
        if any(h.upper() in upper for h in hints):
            return dtype
    return "other"

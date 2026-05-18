"""
main.py - CLI entry point for the Offline RAG System.

Commands:
  python -m rag_system.main index              → index all projects
  python -m rag_system.main index --project "14413 - Sicilia"  → single project
  python -m rag_system.main serve              → start the API server
  python -m rag_system.main status             → show indexing statistics
  python -m rag_system.main query "domanda"    → quick query from terminal
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Log directory ─────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Formatter ─────────────────────────────────────────────────────────────────
_LOG_FMT  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

# ── Console handler (always active) ───────────────────────────────────────────
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)

# ── Rotating file handler ─────────────────────────────────────────────────────
# One log file per day, named rag_YYYYMMDD.log.
# Each file rotates at 50 MB, keeping 30 backup files (~1.5 GB max total).
_today_str   = datetime.now().strftime("%Y%m%d")
_log_file    = _LOG_DIR / f"rag_{_today_str}.log"
_file_handler = RotatingFileHandler(
    _log_file,
    maxBytes=50 * 1024 * 1024,   # 50 MB per file
    backupCount=30,
    encoding="utf-8",
)
_file_handler.setFormatter(_formatter)

# ── Root logger setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FMT,
    datefmt=_DATE_FMT,
    handlers=[_console_handler, _file_handler],
)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("Logs → %s  (console + rotating file)", _log_file)


def cmd_index(args):
    """Run the indexing pipeline."""
    from rag_system.indexing.pipeline import run_indexing
    from rag_system import config

    project = getattr(args, "project", None)
    logger.info("Starting indexing pipeline… (project filter: %s)", project or "ALL")
    stats = run_indexing(project_filter=project)
    if stats:
        skipped_csv = config.SKIPPED_PROJECTS_CSV
        if skipped_csv.exists():
            logger.info("Skipped projects report: %s", skipped_csv)


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    from rag_system import config

    logger.info("Starting API server on http://%s:%d", config.API_HOST, config.API_PORT)
    logger.info("Open-WebUI: add http://%s:%d/v1 as a custom OpenAI endpoint",
                "localhost" if config.API_HOST == "0.0.0.0" else config.API_HOST,
                config.API_PORT)
    uvicorn.run(
        "rag_system.api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="info",
    )


def cmd_status(args):
    """Print indexing statistics."""
    from rag_system.indexing import tracker, vector_store
    from rag_system import config

    tracker.init_db()
    db_stats = tracker.get_stats()

    print("\n" + "═" * 50)
    print("  RAG SYSTEM STATUS")
    print("═" * 50)
    print(f"  Indexed files:     {db_stats['indexed_files']}")
    print(f"  Failed files:      {db_stats['failed_files']}")
    print(f"  Skipped projects:  {db_stats['skipped_projects']}")
    print(f"  Active projects:   {db_stats['active_projects']}")
    print(f"  Total chunks:      {db_stats['total_chunks']:,}")

    # Qdrant per-project stats
    projects = vector_store.list_projects()
    if projects:
        print(f"\n  Qdrant collections ({len(projects)}):")
        for proj in sorted(projects):
            info = vector_store.collection_info(proj)
            print(f"    • {proj}: {info.get('points', 0):,} chunks")

    # Skipped projects CSV
    csv_path = config.SKIPPED_PROJECTS_CSV
    if csv_path.exists():
        print(f"\n  Skipped projects log: {csv_path}")

    print("═" * 50 + "\n")


def cmd_query(args):
    """Run a quick query from the terminal."""
    from rag_system.generation.chain import answer

    question   = args.question
    project_id = getattr(args, "project", None)

    print(f"\nQuery: {question}")
    print(f"Project: {project_id or 'ALL'}\n")
    print("─" * 60)

    result = answer(question, project_id=project_id)

    print(result["answer"])

    if result["sources"]:
        print("\n─── Fonti ───────────────────────────────────────────────")
        for s in result["sources"]:
            print(f"  • {s['source']}  (score: {s['score']})")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="rag-system",
        description="Offline RAG System for Italian PDF documents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index PDF documents")
    p_index.add_argument("--project", "-p", default=None,
                         help="Process only this project folder name")
    p_index.set_defaults(func=cmd_index)

    # serve
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.set_defaults(func=cmd_serve)

    # status
    p_status = sub.add_parser("status", help="Show indexing statistics")
    p_status.set_defaults(func=cmd_status)

    # query
    p_query = sub.add_parser("query", help="Run a quick terminal query")
    p_query.add_argument("question", help="Question to ask (in Italian)")
    p_query.add_argument("--project", "-p", default=None,
                         help="Restrict to this project")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

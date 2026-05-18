#!/usr/bin/env python3
"""
reindex_missing.py - Index all projects that have unindexed or failed PDF files.

Run INSIDE the Docker container:
    docker exec -it rag_system python reindex_missing.py

Or with force mode to re-process everything:
    docker exec -it rag_system python reindex_missing.py --force
"""
import os
import sys
import sqlite3
import argparse
from pathlib import Path

# ── Container paths ───────────────────────────────────────────────────────────
sys.path.insert(0, '/app')
os.environ.setdefault('QDRANT_HOST',      'qdrant')
os.environ.setdefault('QDRANT_PORT',      '6333')
os.environ.setdefault('OLLAMA_BASE_URL',  'http://host.docker.internal:11434')
os.environ.setdefault('EMBED_MODEL',      'bge-m3:latest')
os.environ.setdefault('LLM_MODEL',        'qwen3.6:35b-a3b')

DATA_DIR = Path('/app/data')
DB_PATH  = Path('/app/db/tracker.db')


def get_coverage() -> dict[str, tuple[int, int, int]]:
    """Returns {project_name: (n_pdfs, n_ok, n_failed)}."""
    coverage: dict[str, tuple[int, int, int]] = {}

    ok_counts: dict[str, int]   = {}
    fail_counts: dict[str, int] = {}

    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        for pid, cnt in conn.execute(
            "SELECT project_id, COUNT(*) FROM processed_files WHERE status='ok' GROUP BY project_id"
        ).fetchall():
            ok_counts[pid] = cnt
        for pid, cnt in conn.execute(
            "SELECT project_id, COUNT(*) FROM processed_files WHERE status='failed' GROUP BY project_id"
        ).fetchall():
            fail_counts[pid] = cnt
        conn.close()

    for proj_dir in sorted(DATA_DIR.iterdir()):
        if not proj_dir.is_dir():
            continue
        pdfs = list(proj_dir.glob('*.pdf')) + list(proj_dir.glob('*.PDF'))
        coverage[proj_dir.name] = (
            len(pdfs),
            ok_counts.get(proj_dir.name, 0),
            fail_counts.get(proj_dir.name, 0),
        )
    return coverage


def main():
    parser = argparse.ArgumentParser(description='Index missing RAG projects')
    parser.add_argument('--force', action='store_true',
                        help='Re-index ALL files, even already-indexed ones')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print report but do not run indexing')
    args = parser.parse_args()

    if args.force:
        os.environ['FORCE_REINDEX'] = 'true'

    coverage = get_coverage()

    print(f"\n{'Project':<35} {'PDFs':>6} {'OK':>6} {'Failed':>7}  Status")
    print("─" * 70)

    needs_indexing = []
    for name, (n_pdf, n_ok, n_fail) in coverage.items():
        gap = n_pdf - n_ok
        if args.force:
            status = f'⟳ force re-index ({n_pdf} files)'
            needs_indexing.append(name)
        elif gap > 0:
            status = f'⚠  {gap} unindexed'
            needs_indexing.append(name)
        else:
            status = '✓ complete'
        print(f"  {name:<33} {n_pdf:>6} {n_ok:>6} {n_fail:>7}  {status}")

    total = sum(n for n, _, _ in coverage.values())
    total_ok = sum(ok for _, ok, _ in coverage.values())
    print(f"\n  {'TOTAL':<33} {total:>6} {total_ok:>6}")
    print(f"\n  {len(needs_indexing)} project(s) need indexing.")

    if not needs_indexing:
        print("  Nothing to do.")
        return

    if args.dry_run:
        print("\n  --dry-run mode: skipping actual indexing.")
        return

    print("\n  Starting indexing pipeline - check container logs for progress.\n")
    from rag_system.indexing.pipeline import run_indexing
    run_indexing()
    print("\n  Done. Run this script again to verify coverage.")


if __name__ == '__main__':
    main()

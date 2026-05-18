# Offline RAG System — Setup Guide

## Prerequisites (already on your machine)
- NVIDIA 4060 Ti 16GB, 64GB RAM
- Ollama running with: `bge-m3:latest`, `qwen2.5vl:latest`, `qwen2.5:7b`
- Docker Desktop
- Python 3.10+

---

## Step 1 — Install Python dependencies

```bash
cd "RAG Implementation V2"

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install all packages
pip install -r requirements.txt
```

> **Surya OCR note:** If `surya-ocr` fails to install, it may need a specific torch version.
> Try: `pip install surya-ocr --extra-index-url https://download.pytorch.org/whl/cu121`

---

## Step 2 — Start Qdrant (vector database)

```bash
docker compose up -d qdrant
```

Verify it's running:
```
http://localhost:6333/dashboard
```

---

## Step 3 — Configure (optional)

```bash
copy .env.example .env
# Edit .env only if you want to change defaults
# The defaults work out of the box with your Ollama setup
```

---

## Step 4 — Index your PDFs (first time)

```bash
# Index ALL projects
./start.sh index
# or on Windows:
python -m rag_system.main index

# Index a single project only
python -m rag_system.main index --project "14413 - Sicilia"
```

**What happens:**
1. Scans `data/` for project folders
2. Empty or unreadable folders → logged to `logs/skipped_projects.csv`, pipeline continues
3. Each PDF is parsed, OCR'd (if scanned), tables extracted, images captioned
4. All content is embedded with bge-m3 and stored in Qdrant
5. Already-indexed unchanged files are skipped automatically

**Empty project CSV:**
If any project folders are empty or have unreadable PDFs, a report is saved to:
```
logs/skipped_projects.csv
```
Columns: `project_id`, `folder_path`, `reason`, `pdf_count`, `empty_pdf_names`, `detected_at`

---

## Step 5 — Start the API server

```bash
python -m rag_system.main serve
# or
./start.sh serve
```

Server runs at: `http://localhost:8000`

---

## Step 6 — Connect Open-WebUI

1. Open Open-WebUI in your browser
2. Go to **Settings → Connections**
3. Add a new OpenAI-compatible API:
   - URL: `http://localhost:8000/v1`
   - API Key: `any-value` (not checked)
4. Save — you'll now see models in the dropdown:
   - `rag-all` → searches all projects
   - `rag-14413_sicilia` → searches only project 14413
   - `rag-14432_sicilia` → etc.

**Tip:** To scope a query to a specific project from within Open-WebUI's system prompt, add:
```
[project:14413 - Sicilia]
```

---

## Daily workflow

```bash
# After adding new PDFs to any project folder:
python -m rag_system.main index

# New project folders are detected automatically.
# Unchanged files are skipped (fast incremental update).
```

---

## CLI reference

```bash
# Index everything
python -m rag_system.main index

# Index one project
python -m rag_system.main index --project "14413 - Sicilia"

# Start API server
python -m rag_system.main serve

# Check status (files indexed, chunks, skipped projects)
python -m rag_system.main status

# Quick terminal query (no Open-WebUI needed)
python -m rag_system.main query "Qual è la potenza installata del parco eolico?"
python -m rag_system.main query "Descrivi i vincoli idrogeologici" --project "14413 - Sicilia"
```

---

## Folder structure

```
RAG Implementation V2/
├── data/                        ← Your PDF projects (read-only)
│   ├── 14413 - Sicilia/
│   ├── 14432 - Sicilia/
│   └── ...
├── rag_system/
│   ├── config.py                ← All settings
│   ├── ingestion/
│   │   ├── pdf_parser.py        ← PyMuPDF text + image extraction
│   │   ├── ocr_engine.py        ← OCR via qwen2.5vl (Ollama) + Tesseract fallback
│   │   ├── image_captioner.py   ← qwen2.5vl deep image understanding
│   │   ├── table_extractor.py   ← pdfplumber table → Markdown
│   │   └── chunker.py           ← Semantic Italian chunking
│   ├── indexing/
│   │   ├── tracker.py           ← SQLite file tracker (MD5 hashing)
│   │   ├── embedder.py          ← bge-m3 via Ollama
│   │   ├── vector_store.py      ← Qdrant client
│   │   └── pipeline.py          ← Main orchestrator + empty-folder CSV
│   ├── retrieval/
│   │   └── retriever.py         ← Query-time search + formatting
│   ├── generation/
│   │   └── chain.py             ← LangChain RAG + qwen2.5:7b
│   ├── api/
│   │   └── server.py            ← FastAPI OpenAI-compatible server
│   └── main.py                  ← CLI entry point
├── db/
│   └── tracker.db               ← SQLite (auto-created)
├── logs/
│   └── skipped_projects.csv     ← Empty/failed project report (auto-created)
├── qdrant_storage/              ← Qdrant data (Docker volume)
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── start.sh
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Qdrant connection refused | Run `docker compose up -d qdrant` |
| Ollama not found | Ensure `ollama serve` is running |
| Empty skipped_projects.csv | All projects have valid PDFs — this is normal |
| Low quality answers | Increase `RETRIEVAL_TOP_K` in `.env` (try 8–10) |
| Images not captioned | Check Ollama has `qwen2.5vl:latest` pulled: `ollama pull qwen2.5vl:latest` |
| OCR quality poor | Increase `OCR_DPI=400` in `.env`; qwen2.5vl handles the actual text extraction |
| **Do NOT install surya-ocr** | surya-ocr requires Pillow<11 which cannot build on Python 3.14. OCR is handled by qwen2.5vl via Ollama instead — no surya needed. |

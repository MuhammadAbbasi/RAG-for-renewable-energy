# A176LAB - RAG System V2

Offline, GPU-accelerated Retrieval-Augmented Generation for Italian PDF documents.  
Runs entirely on local hardware with no cloud and no data leakage.

---

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8 GB VRAM (NVIDIA) | RTX 4060 Ti 16 GB |
| RAM | 32 GB | 64 GB |
| Storage | 20 GB free | 100 GB+ (for PDFs + vectors) |
| OS | Windows 10/11, Ubuntu 22+ | Ubuntu 22+ |

---

## Prerequisites

1. **Docker Desktop** (with WSL2 backend on Windows) - [docs.docker.com](https://docs.docker.com)
2. **Ollama** running locally - [ollama.com](https://ollama.com)
3. **NVIDIA Container Toolkit** (for GPU passthrough in Docker on Linux)

### Pull required models into Ollama

```bash
# Primary LLM (35B MoE - requires ~20 GB VRAM or runs on CPU with 64 GB RAM)
ollama pull qwen3.6:35b-a3b

# Vision model (image captioning, map descriptions)
ollama pull qwen2.5vl:latest

# Primary embedding model (multilingual, 1024-dim, optimised for Italian)
ollama pull bge-m3:latest

# Fallback embedding (faster, 768-dim - used when bge-m3 is unavailable)
ollama pull nomic-embed-text:latest
```

> **Tip for Windows:** enable parallel model loading so the LLM and embedding model stay resident simultaneously:
> ```powershell
> $env:OLLAMA_NUM_PARALLEL=2; $env:OLLAMA_MAX_LOADED_MODELS=2; ollama serve
> ```

---

## Quick start

```bash
# 1. Enter the project folder
cd "RAG Implementation V2"

# 2. Start all services (Qdrant + RAG API)
docker compose up -d

# 3. Open the web interface
#    http://localhost:8000/app
#    Default login: admin / admin123  ← change this immediately
```

---

## Adding documents

### Drop-folder (recommended)

Copy project folders into `./data/`. Each subfolder becomes a separate searchable collection.

```
data/
├── ProjectAlpha/
│   ├── relazione_tecnica.pdf
│   └── planimetria.pdf
└── ProjectBeta/
    └── disciplinare.pdf
```

### Via Admin UI

**Admin → Indicizzazione** - shows per-project coverage, lets you trigger indexing with or without force re-index.

### Via API

```bash
curl -X POST http://localhost:8000/v1/data-sources \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"path": "/mnt/nas/documenti", "label": "NAS Archivio"}'
```

---

## Starting ingestion

### Via web UI (Admin panel)

1. Log in as admin → open the **Indicizzazione** panel
2. Optionally tick **Forza ri-indicizzazione** to re-process all files
3. Click **▶ Avvia indicizzazione** - progress appears in the container logs

### Via terminal (inside container)

```bash
# Standard run - skips already-indexed files
docker exec -it rag_system python /app/rag_system/reindex_missing.py

# Dry run - shows what would be indexed without actually doing it
docker exec -it rag_system python /app/rag_system/reindex_missing.py --dry-run

# Force - re-processes everything even if unchanged
docker exec -it rag_system python /app/rag_system/reindex_missing.py --force
```

### Via API

```bash
# Trigger background indexing
curl -X POST "http://localhost:8000/v1/index" \
  -H "Authorization: Bearer <admin_token>"

# Force re-index all files
curl -X POST "http://localhost:8000/v1/index?force=true" \
  -H "Authorization: Bearer <admin_token>"

# Poll status
curl http://localhost:8000/v1/index/status \
  -H "Authorization: Bearer <token>"
```

---

## Querying

### Web interface

Open `http://localhost:8000/app`, log in, select a project or search all, and type your question in Italian or English.  
Sources with page numbers appear below each answer.

### Direct API (streaming)

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quali sono le date e le scadenze importanti?",
    "project_id": "ProjectAlpha",
    "top_k": 6,
    "stream": true
  }'
```

### Direct API (blocking)

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qual è la potenza installata?",
    "stream": false
  }'
```

Response:

```json
{
  "answer": "La potenza installata è di 42 MW...",
  "sources": [
    {"filename": "relazione.pdf", "page": 3, "score": 0.812, "project": "ProjectAlpha"},
    {"filename": "allegato_A.pdf", "page": 7, "score": 0.741, "project": "ProjectAlpha"}
  ],
  "model": "qwen3.6:35b-a3b",
  "project": "ProjectAlpha"
}
```

### OpenAI-compatible endpoint (Open-WebUI)

The system is fully OpenAI-compatible at `/v1/chat/completions`.

In Open-WebUI → **Settings → Connections → Add OpenAI connection**:
- Base URL: `http://localhost:8000/v1`
- API Key: your session token

Select model `rag-all` to search all projects, or `rag-<project>` for a specific one.

---

## API reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/login` | - | Get session token |
| POST | `/auth/logout` | user | Invalidate token |
| GET | `/auth/me` | user | Current user info |
| GET | `/auth/users` | admin | List all users |
| POST | `/auth/users` | admin | Create user |
| PUT | `/auth/users/{id}` | admin | Update user |
| DELETE | `/auth/users/{id}` | admin | Delete user |
| GET | `/health` | - | Liveness check |
| GET | `/` | - | System dashboard |
| GET | `/app` | - | Chat web interface |
| GET | `/docs` | - | Interactive API docs (Swagger) |
| POST | `/query` | user | Direct RAG query (streaming or blocking) |
| GET | `/v1/stats` | user | Aggregated stats + per-project file tree |
| GET | `/v1/projects` | user | List indexed projects with chunk counts |
| GET | `/v1/projects/{id}/files` | user | Files indexed for a project |
| GET | `/v1/models` | - | List available RAG models |
| GET | `/v1/llm-models` | - | List Ollama LLMs available for generation |
| POST | `/v1/chat/completions` | user | OpenAI-compatible streaming chat |
| POST | `/v1/index` | admin | Trigger background re-indexing (`?force=true`) |
| GET | `/v1/index/status` | user | Indexing running / idle |
| GET | `/v1/data-sources` | admin | List registered folders |
| POST | `/v1/data-sources` | admin | Register a folder |
| DELETE | `/v1/data-sources/{id}` | admin | Remove a folder |
| GET | `/debug/query` | user | Step-by-step retrieval diagnostic (`?q=your+question`) |

---

## Configuration

All settings live in `.env` at the project root (create from `.env.example`).  
Defaults work out of the box - only change what you need.

```env
# ── Models ────────────────────────────────────────────────────────────────────
LLM_MODEL=qwen3.6:35b-a3b
EMBED_MODEL=bge-m3:latest
EMBED_MODEL_FALLBACK=nomic-embed-text:latest
VISION_MODEL=qwen2.5vl:latest

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K=10
RETRIEVAL_SCORE_THRESHOLD=0.25   # lowered to compensate for BM25 score deflation
BM25_ENABLED=true
BM25_WEIGHT=0.3                  # 0 = pure dense, 1 = pure BM25
BM25_OVERRETRIEVE=20             # dense candidates fetched before BM25 reranks

# ── Generation ────────────────────────────────────────────────────────────────
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096
LLM_CONTEXT_SIZE=16384           # qwen3.6 supports 32K; 16K leaves headroom for output
FORMAT_CONTEXT_MAX_CHARS=36000   # max retrieved text passed to LLM

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE=1000                  # characters per chunk (richer context)
CHUNK_OVERLAP=150                # overlap between adjacent chunks
CHUNK_MIN_CHARS=80               # discard chunks shorter than this

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_LANGUAGE=it                  # EasyOCR language (Italian)
OCR_DPI=300

# ── Indexing ──────────────────────────────────────────────────────────────────
INDEXING_WORKERS=2               # parallel project workers (GPU ops serialised)
FORCE_REINDEX=false              # true = re-embed all files regardless of hash

# ── API server ────────────────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Architecture overview

```
data/
└── ProjectName/
    └── *.pdf
         │
         ▼
    ┌─────────────────────────────────────┐
    │           pipeline.py               │
    │  1. MD5 hash check (skip unchanged) │
    │  2. pdfplumber pre-validation       │
    │  3. PDF parse - PyMuPDF             │
    │  4. OCR scanned pages - EasyOCR GPU │
    │  5. Table extraction - pdfplumber   │
    │  6. Image captioning - qwen2.5vl    │
    │  7. Semantic chunking (1000 / 150)  │
    │  8. Batch embed - bge-m3 (1024-dim) │
    │  9. Upsert → Qdrant                 │
    │ 10. Mark indexed in tracker.db      │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │   Qdrant (Cosine, 1024-dim)         │
    │   One collection per project        │
    │   rag_project_<sanitized_name>      │
    └─────────────────────────────────────┘
         │
         ▼  POST /query  or  /v1/chat/completions
    ┌─────────────────────────────────────┐
    │           retriever.py              │
    │  1. Embed query - bge-m3            │
    │  2. Dense search (top_k × 20 wide)  │
    │  3. BM25 re-rank (weight 0.3)       │
    │  4. Zero-signal guard (pure dense   │
    │     fallback when no keyword match) │
    │  5. Score threshold filter (0.25)   │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │           chain.py                  │
    │  Build prompt with [CONTESTO]       │
    │  Inject conversation history        │
    │  Stream → qwen3.6:35b-a3b           │
    │  Emit __SOURCES__ sentinel          │
    └─────────────────────────────────────┘
```

### Key design decisions

**Query-priority gate** - when a user query arrives, a threading Event (`QUERY_GATE`) pauses all background embedding batches so Ollama can serve the LLM without resource contention. The gate auto-releases after 5 minutes via a watchdog thread to prevent permanent blocking.

**BM25 hybrid search** - after dense Qdrant retrieval, BM25 reranks candidates using keyword overlap. A zero-signal guard prevents score deflation when BM25 finds no keyword matches (which would artificially raise the effective threshold by ~43%).

**Incremental indexing** - each file is MD5-hashed before indexing. Unchanged files are skipped on subsequent runs. Failed files are always retried.

**Corrupted PDF guard** - pdfplumber validates each PDF before PyMuPDF or EasyOCR touch it, preventing native library crashes (SIGABRT) from malformed files.

---

## Logs

```
logs/
├── rag_YYYYMMDD.log     # Full server + indexing log (rotates daily, 50 MB max)
├── query_log.jsonl      # One JSON line per query - user, question, sources, ms
└── skipped_projects.csv # Projects with no processable PDFs
```

View live logs:
```bash
docker compose logs -f rag
```

---

## Troubleshooting

### Queries return 0 sources

Run the built-in diagnostic endpoint (login required):

```
http://localhost:8000/debug/query?q=your+question
```

It tests every step: embed → Qdrant connection → point counts → raw scores → full retrieval, and reports exactly where the pipeline fails.

Common causes:
- **bge-m3 not loaded in Ollama** - run `ollama pull bge-m3:latest` and retry
- **Score threshold too high** - lower `RETRIEVAL_SCORE_THRESHOLD` to `0.2` in `.env`
- **Qdrant empty** - check `http://localhost:6333/dashboard` and re-run indexing

### Indexing crashes mid-run

Corrupted PDFs can cause native library crashes. The pipeline pre-validates with pdfplumber, but some malformed files pass validation. Re-run indexing - successfully indexed files are skipped, so only failed ones are retried.

Check which files failed:
```bash
docker exec -it rag_system python /app/rag_system/reindex_missing.py --dry-run
```

### VRAM / Out-of-Memory errors

1. Reduce workers: `INDEXING_WORKERS=1`
2. Use a smaller LLM: `LLM_MODEL=qwen2.5:7b`
3. Disable image captioning: `IMAGE_MIN_WIDTH=99999`
4. After vision model runs, the system automatically unloads it and waits 20 s before loading the embedding model - this is intentional

### Indexing is slow

- Text-only PDFs: ~2–5 s/page
- Scanned PDFs (EasyOCR on RTX 4060 Ti): ~3–10 s/page
- Increase `INDEXING_WORKERS=4` for text-only datasets
- Watch progress: `docker compose logs -f rag`

### Ollama 500 errors during indexing

Ollama returns 500 when switching between models (VRAM swap). The system retries automatically with exponential backoff (8 attempts, up to 120 s between retries). If errors persist, ensure `OLLAMA_NUM_PARALLEL=2` and `OLLAMA_MAX_LOADED_MODELS=2` are set in the Ollama environment.

### Re-indexing a single project

Delete the project's Qdrant collection via `http://localhost:6333/dashboard`, then trigger indexing from the Admin UI or:

```bash
curl -X POST "http://localhost:8000/v1/index?force=true" \
  -H "Authorization: Bearer <admin_token>"
```

---

## Development

The `./rag_system` directory is live-mounted into the container - edit any `.py` file and restart to apply (no rebuild needed):

```bash
docker compose restart rag
```

Rebuild the image only when `requirements.txt` changes:

```bash
docker compose build rag && docker compose up -d
```

Install deps locally for IDE support:

```bash
pip install -r requirements.txt --break-system-packages
```

### Project structure

```
RAG Implementation V2/
├── docker-compose.yml          # Qdrant + RAG API services
├── Dockerfile                  # Python 3.11, EasyOCR, GPU passthrough
├── requirements.txt
├── .env                        # Runtime config (gitignore this)
├── data/                       # PDF project folders (mounted read-only)
├── db/                         # SQLite tracker (tracker.db)
├── logs/                       # Rotating logs + query analytics
├── qdrant_storage/             # Qdrant persistence (mounted)
└── rag_system/
    ├── config.py               # Central config - all env vars with defaults
    ├── main.py                 # CLI entry point (index / serve / query)
    ├── reindex_missing.py      # Coverage report + retry tool
    ├── api/
    │   ├── server.py           # FastAPI app - all endpoints
    │   ├── app.html            # Web UI (single-file, no build step)
    │   └── auth.py             # Session/API-key auth (SQLite-backed)
    ├── generation/
    │   └── chain.py            # RAG prompt builder + Ollama streaming
    ├── indexing/
    │   ├── pipeline.py         # Orchestrates all indexing steps
    │   ├── embedder.py         # bge-m3 batch + fallback embedding
    │   ├── vector_store.py     # Qdrant client wrapper
    │   ├── tracker.py          # SQLite file-hash tracker
    │   └── priority.py         # QUERY_GATE threading event
    ├── retrieval/
    │   └── retriever.py        # Dense + BM25 hybrid retrieval
    └── ingestion/
        ├── pdf_parser.py       # PyMuPDF page parsing
        ├── ocr_engine.py       # EasyOCR GPU integration
        ├── table_extractor.py  # pdfplumber table extraction
        ├── image_captioner.py  # qwen2.5vl image descriptions
        └── chunker.py          # Italian-aware text chunking
```

---

## Backup

The only three folders you need to preserve the full system state:

| Folder | Contents |
|--------|----------|
| `rag_system/` | All application code |
| `qdrant_storage/` | All embedded vectors (30 K+ chunks) |
| `db/` | Indexing tracker (which files are indexed) |

The Docker image itself is stateless - all state lives in these mounted folders.

---

## Contributing

We welcome contributions! Here's how to get started.

### Development setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MuhammadAbbasi/RAG-for-renewable-energy.git
   cd "RAG Implementation V2"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   # or
   .\venv\Scripts\activate        # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Copy `.env.example` to `.env`** and configure for your local setup
   ```bash
   cp .env.example .env
   # Edit .env with your model paths, API keys, etc.
   ```

5. **Start services**
   ```bash
   docker compose up -d
   ```

6. **Run the system**
   ```bash
   python -m rag_system.main serve      # Start API server
   # In another terminal:
   python -m rag_system.main index      # Index your data
   ```

### Code guidelines

- **Python style**: Follow PEP 8. Aim for clarity over brevity.
- **Type hints**: Use them liberally - they improve IDE support and catch bugs early.
- **Logging**: Use the structured logging in `config.py`; avoid bare `print()` statements.
- **Docstrings**: Keep them concise. Focus on the *why*, not the what (code should be self-documenting).
- **Testing**: Add tests for any new retrieval/indexing logic. Run with:
  ```bash
  pytest tests/
  ```

### Common areas to contribute

| Area | Files | What it does |
|------|-------|-------------|
| **Retrieval** | `retrieval/retriever.py` | Dense + BM25 ranking, score filtering |
| **Indexing** | `indexing/pipeline.py` | Orchestrates PDF → chunks → vectors |
| **PDF parsing** | `ingestion/pdf_parser.py` | Extracts text, tables, images from PDFs |
| **OCR** | `ingestion/ocr_engine.py` | Scanned page text extraction (GPU) |
| **API** | `api/server.py` | REST endpoints, auth, streaming responses |
| **UI** | `api/app.html` | Web chat interface (single-file HTML) |
| **Router** | `wiki/router.py` | Decides: wiki → structured DB, or RAG → dense search |

### Submitting changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-improvement
   ```

2. **Make your changes** - keep commits focused and atomic

3. **Test thoroughly**
   - Manual tests via the web UI or API
   - Check logs for errors: `docker compose logs -f rag`
   - Run diagnostics: `http://localhost:8000/debug/query?q=test`

4. **Push and open a PR**
   ```bash
   git push origin feature/my-improvement
   ```

5. **PR checklist**
   - [ ] Describe what changed and why
   - [ ] Link any related issues
   - [ ] Tested locally (web UI, API, or CLI)
   - [ ] No breaking changes to the API or config format

### Reporting issues

- **Bug**: Open an issue with a minimal reproduction, your OS, and container logs
- **Feature request**: Describe the use case and desired behavior
- **Question**: Check the [troubleshooting section](#troubleshooting) first

---

## License

MIT — see [LICENSE](LICENSE) for full terms.

---

## Support

For questions, issues, or feedback:
- **Email**: info@a176lab.it
- **Issues**: [GitHub Issues](https://github.com/MuhammadAbbasi/RAG-for-renewable-energy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MuhammadAbbasi/RAG-for-renewable-energy/discussions)

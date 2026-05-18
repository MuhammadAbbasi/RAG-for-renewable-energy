"""
config.py - Central configuration for the Offline RAG System.
All settings are controlled here or via environment variables / .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"                      # Root folder containing all project subfolders
LOGS_DIR = BASE_DIR / "logs"                      # Log files
DB_DIR   = BASE_DIR / "db"                        # SQLite tracker database
QDRANT_STORAGE_DIR = BASE_DIR / "qdrant_storage"  # Qdrant local persistence (if not using Docker)

# CSV report for skipped/empty project folders
SKIPPED_PROJECTS_CSV = LOGS_DIR / "skipped_projects.csv"

# ─────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Primary LLM for answer generation.
# qwen3.6:35b-a3b - best quality, runs via Ollama (fits in 64 GB RAM).
# Falls back to qwen3.5:9b or qwen2.5:7b when VRAM is tight.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6:35b-a3b")

# Vision-language model for image captioning and map description
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:latest")

# Primary embedding model (multilingual, 1024-dim, ideal for Italian)
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3:latest")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1024"))

# Fallback embedding model (faster, 768-dim, used when bge-m3 is saturated)
EMBED_MODEL_FALLBACK     = os.getenv("EMBED_MODEL_FALLBACK", "nomic-embed-text:latest")
EMBED_DIMENSION_FALLBACK = int(os.getenv("EMBED_DIMENSION_FALLBACK", "768"))

# ─────────────────────────────────────────────
# QDRANT
# ─────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_PREFIX = "rag_project_"   # Collection name = prefix + sanitized project id
QDRANT_DISTANCE = "Cosine"                  # Similarity metric

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))   # Target tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150")) # Overlap between adjacent chunks

# Minimum characters a chunk must have to be stored (filters noise)
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "80"))

# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "it")   # Italian
OCR_DPI      = int(os.getenv("OCR_DPI", "300"))  # DPI for rendering scanned pages

# A page is considered "scanned" (needs OCR) if extracted text is below this threshold
SCANNED_TEXT_THRESHOLD = int(os.getenv("SCANNED_TEXT_THRESHOLD", "50"))  # characters

# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────
RETRIEVAL_TOP_K           = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.25"))

# BM25 hybrid search
# When enabled: dense Qdrant retrieves top_k * BM25_OVERRETRIEVE candidates,
# BM25 re-ranks them, and the final top_k are passed to the LLM.
BM25_ENABLED      = os.getenv("BM25_ENABLED", "true").lower() == "true"
BM25_OVERRETRIEVE = int(os.getenv("BM25_OVERRETRIEVE", "20"))  # candidates per dense pass
# Weight for combining dense + BM25 scores (0 = pure dense, 1 = pure BM25)
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "4096"))
# qwen3.6:35b-a3b supports 32 K context; keep at 16 K to leave headroom for output
LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "16384"))
# Max chars of retrieved context passed to the LLM (~9000 tokens available after system prompt)
FORMAT_CONTEXT_MAX_CHARS = int(os.getenv("FORMAT_CONTEXT_MAX_CHARS", "36000"))

# ─────────────────────────────────────────────
# WIKI KNOWLEDGE EXTRACTION
# ─────────────────────────────────────────────
# Set WIKI_ENABLED=false to disable wiki extraction (RAG-only mode)
WIKI_ENABLED        = os.getenv("WIKI_ENABLED", "true").lower() == "true"
# Model used for extraction - can be a smaller/faster model than LLM_MODEL
# e.g. "qwen2.5:7b" for speed, or leave blank to use LLM_MODEL
WIKI_EXTRACT_MODEL  = os.getenv("WIKI_EXTRACT_MODEL", "") or os.getenv("LLM_MODEL", "qwen2.5:7b")
# Number of pages to read from the start of each PDF for extraction
WIKI_EXTRACT_PAGES  = int(os.getenv("WIKI_EXTRACT_PAGES", "5"))

# ─────────────────────────────────────────────
# API SERVER
# ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ─────────────────────────────────────────────
# IMAGE CAPTIONING
# ─────────────────────────────────────────────
IMAGE_MAX_DIM    = int(os.getenv("IMAGE_MAX_DIM", "1024"))
IMAGE_MIN_WIDTH  = int(os.getenv("IMAGE_MIN_WIDTH", "80"))
IMAGE_MIN_HEIGHT = int(os.getenv("IMAGE_MIN_HEIGHT", "80"))

# ─────────────────────────────────────────────
# INDEXING PIPELINE
# ─────────────────────────────────────────────
# Number of parallel project workers.
# GPU ops (OCR, captioning) are serialised by a shared lock so VRAM is safe.
# • 2 - recommended default (one worker overlaps CPU work with another's GPU)
# • 4 - text-only datasets with no OCR/captioning bottleneck
# • 1 - use when debugging or very tight on VRAM
INDEXING_WORKERS = int(os.getenv("INDEXING_WORKERS", "2"))

# Whether to re-index a file even if it hasn't changed (force mode)
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"

# Show tqdm progress bars in the terminal during indexing
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "true").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [DATA_DIR, LOGS_DIR, DB_DIR, QDRANT_STORAGE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def sanitize_collection_name(project_id: str) -> str:
    """Convert a project folder name to a valid Qdrant collection name."""
    import re
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", project_id)
    clean = re.sub(r"_+", "_", clean).strip("_").lower()
    return f"{QDRANT_COLLECTION_PREFIX}{clean}"
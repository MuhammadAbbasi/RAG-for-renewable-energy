# ── RAG System — CUDA-enabled Dockerfile ──────────────────────────────────────
#
# Base: python:3.11-slim  (clean, no conda conflicts)
# CUDA: PyTorch CUDA 12.1 wheels — runtime CUDA libs bundled inside the wheel,
#       so no nvidia/cuda base image is needed.
# OCR:  EasyOCR with GPU (CRAFT + CRNN models pre-downloaded at build time)
#
# This container handles BOTH:
#   • API server  →  started automatically (CMD)
#   • Indexing    →  run on demand:
#       docker compose exec rag python -m rag_system.main index
#
# Prerequisites on the host:
#   Windows / Docker Desktop  — enable WSL2 backend + GPU in Docker Desktop settings
#   Linux                     — install NVIDIA Container Toolkit:
#       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System packages ───────────────────────────────────────────────────────────
# libgl1, libglib2.0-0  → required by OpenCV (used by EasyOCR)
# libmupdf-dev          → required by PyMuPDF (fitz)
# tesseract-ocr-ita     → Italian Tesseract model (CPU fallback)
# curl                  → healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-ita \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: PyTorch with CUDA 12.1 ───────────────────────────────────────────
# The CUDA wheel bundles its own runtime libs (libcublas, libcudnn, etc.)
# so no cuda base image is needed.  ~2.5 GB download but cached by Docker.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.3.0 \
        torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cu121

# ── Step 2: EasyOCR (primary GPU OCR engine) ─────────────────────────────────
# torch is already installed above; EasyOCR finds it automatically.
RUN pip install --no-cache-dir easyocr==1.7.1 numpy

# ── Step 3: Pre-download EasyOCR models at build time ────────────────────────
# Models (~300 MB) are stored in /root/.EasyOCR inside the image so the first
# indexing run doesn't need internet access.  gpu=False is correct here
# (no GPU during docker build); at runtime EasyOCR auto-detects CUDA.
RUN python -c "\
import easyocr, sys; \
print('Downloading EasyOCR models for Italian + English…', flush=True); \
easyocr.Reader(['it', 'en'], gpu=False, verbose=True); \
print('Models ready.', flush=True)"

# ── Step 4: Remaining Python requirements ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 5: Application code ──────────────────────────────────────────────────
COPY rag_system/ ./rag_system/

# ── Runtime directories (bound as volumes in docker-compose.yml) ──────────────
RUN mkdir -p /app/data /app/db /app/logs /app/qdrant_storage

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default: start the API server ─────────────────────────────────────────────
# To run indexing instead:
#   docker compose exec rag python -m rag_system.main index
CMD ["python", "-m", "rag_system.main", "serve"]

"""
priority.py — Query-priority gate for the indexing pipeline.

How it works
------------
The embedding loop in embedder.py checks QUERY_GATE before every batch.
When a user submits a chat query, server.py clears the gate (blocks new
batches from starting) and sets it again once the response is fully delivered.
This ensures Ollama can swap from bge-m3 → the LLM without two competing
callers fighting over the same GPU.

For true parallel execution (both models resident at the same time), set
OLLAMA_NUM_PARALLEL=2 and OLLAMA_MAX_LOADED_MODELS=2 in the Ollama process
environment — see docker-compose.yml comments.

Thread safety
-------------
threading.Event is fully thread-safe. FastAPI's async handlers run in the
same process and share this event with the indexing thread pool.
"""

import threading

# QUERY_GATE is SET   → indexing embedding batches may proceed
# QUERY_GATE is CLEAR → a user query is in progress; batches must wait
QUERY_GATE = threading.Event()
QUERY_GATE.set()   # default: no query active, indexing is free to run

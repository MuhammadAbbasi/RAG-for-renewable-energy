"""
wiki/ - Structured knowledge extraction layer for RAG.

Extracts structured project records from PDF documents using an LLM,
stores them in a SQLite database, and routes aggregative queries
(how many projects, list all, compare MW) to SQL instead of vector search.

Public API
----------
from rag_system.wiki import extractor, store, query, router
"""

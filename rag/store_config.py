"""Persist paths and embedding model names — keep in sync across ingest and query."""

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "suvit_help"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_EMBED_DIM = 384

"""Embedding model setup for APEX RAG — Voyage AI API (lightweight, no PyTorch)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import voyageai

# Ensure .env is loaded (needed when running rag module standalone)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

_client = voyageai.Client()  # Uses VOYAGE_API_KEY env var

MODEL = "voyage-3-lite"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of document texts and return float vectors."""
    result = _client.embed(texts, model=MODEL, input_type="document")
    return result.embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query text and return float vector."""
    result = _client.embed([text], model=MODEL, input_type="query")
    return result.embeddings[0]

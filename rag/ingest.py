"""Ingestion script — scan knowledge/ folders, chunk, embed, store in ChromaDB."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .store import add_documents, get_collection, AGENT_ROLES

# Delay between embedding API calls to respect rate limits
EMBED_DELAY = 1  # seconds between calls (paid tier: 300+ RPM)

# Chunking parameters
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def _read_file(path: Path) -> str:
    """Read a text file (.txt or .md)."""
    return path.read_text(encoding="utf-8")


def ingest_role(role: str, force: bool = False) -> int:
    """Ingest all documents for a specific role. Returns count of chunks added."""
    if role not in AGENT_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {AGENT_ROLES}")

    role_dir = KNOWLEDGE_DIR / role
    if not role_dir.exists():
        print(f"  No knowledge directory for {role}, skipping.")
        return 0

    # Check for existing data
    collection = get_collection(role)
    if collection.count() > 0 and not force:
        print(f"  Collection '{role}' already has {collection.count()} docs. Use force=True to re-ingest.")
        return 0

    # Clear existing if force
    if force and collection.count() > 0:
        # Delete all existing documents
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        print(f"  Cleared {len(existing['ids'])} existing docs from '{role}'.")

    # Scan for .txt and .md files
    files = list(role_dir.glob("*.md")) + list(role_dir.glob("*.txt"))
    if not files:
        print(f"  No .md or .txt files found in {role_dir}")
        return 0

    total_chunks = 0
    for filepath in files:
        text = _read_file(filepath)
        chunks = _chunk_text(text)

        # Create stable IDs based on file + chunk index
        file_hash = hashlib.md5(filepath.name.encode()).hexdigest()[:8]
        ids = [f"{role}_{file_hash}_{i}" for i in range(len(chunks))]
        metadatas = [{"role": role, "source": filepath.name, "chunk_index": i} for i in range(len(chunks))]

        # Batch embed — Voyage AI supports up to 128 texts per call
        BATCH_SIZE = 64
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            if batch_start > 0:
                time.sleep(EMBED_DELAY)
            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            add_documents(
                role,
                chunks[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
                ids=ids[batch_start:batch_end],
            )

        total_chunks += len(chunks)
        print(f"  {filepath.name}: {len(chunks)} chunks")

    return total_chunks


def ingest_all(force: bool = False) -> dict[str, int]:
    """Ingest knowledge documents for all roles. Returns counts per role."""
    results = {}
    for role in AGENT_ROLES:
        print(f"\nIngesting '{role}'...")
        results[role] = ingest_role(role, force=force)

    total = sum(results.values())
    print(f"\nTotal: {total} chunks ingested across {len(results)} collections.")
    return results


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    ingest_all(force=force)

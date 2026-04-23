"""ChromaDB vector store — one persistent collection per agent role.

Role identifiers (`cso`, `cto`, `cmo`, `cbo`, `ip_attorney`, `shared`) are
kept as internal Python identifiers for compatibility with the rest of the
codebase. The external ChromaDB collection names are mapped to the public
scheme in `config.COLLECTION_NAMES` (e.g., `cso` -> `apex_scientific`).
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from config import COLLECTION_NAMES
from .embeddings import embed_texts, embed_query

# Persistent storage directory
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chromadb_data"

# Valid agent roles that get their own collection
AGENT_ROLES = ("cso", "cto", "cmo", "cbo", "ip_attorney", "shared")

# Map internal role id -> external collection name (from config)
_ROLE_TO_COLLECTION = {
    "cso":          COLLECTION_NAMES["scientific"],
    "cto":          COLLECTION_NAMES["technical"],
    "cmo":          COLLECTION_NAMES["clinical"],
    "cbo":          COLLECTION_NAMES["commercial"],
    "ip_attorney":  COLLECTION_NAMES["ip"],
    "shared":       COLLECTION_NAMES["shared"],
}


def _get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(role: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection for the given agent role."""
    if role not in AGENT_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {AGENT_ROLES}")
    client = _get_client()
    return client.get_or_create_collection(
        name=_ROLE_TO_COLLECTION[role],
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    role: str,
    texts: list[str],
    metadatas: list[dict] | None = None,
    ids: list[str] | None = None,
) -> int:
    """Add document chunks to a role's collection. Returns count added."""
    if not texts:
        return 0

    collection = get_collection(role)
    embeddings = embed_texts(texts)

    if ids is None:
        # Generate deterministic IDs from role + index
        existing_count = collection.count()
        ids = [f"{role}_{existing_count + i}" for i in range(len(texts))]

    if metadatas is None:
        metadatas = [{"role": role}] * len(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return len(texts)


def query_collection_by_embedding(
    role: str, embedding: list[float], k: int = 5
) -> list[dict]:
    """Query a role's collection using a pre-computed embedding vector."""
    collection = get_collection(role)

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


def query_collection(role: str, query: str, k: int = 5) -> list[dict]:
    """Query a role's collection and return top-k results with metadata."""
    collection = get_collection(role)

    if collection.count() == 0:
        return []

    query_embedding = [embed_query(query)]
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits

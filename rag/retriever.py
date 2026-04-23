"""RAG retriever — query interface for executive agents."""

from __future__ import annotations

from .store import query_collection


def retrieve_context(role: str, query: str, k: int = 5) -> str:
    """Retrieve relevant knowledge base context for an agent.

    Queries both the role-specific collection and the shared collection,
    then merges and deduplicates results.

    Returns a formatted string ready to inject into the prompt.
    """
    # Query role-specific knowledge
    role_hits = query_collection(role, query, k=k)

    # Query shared knowledge
    shared_hits = query_collection("shared", query, k=max(2, k // 2))

    # Merge, preferring role-specific hits
    all_hits = role_hits + shared_hits

    # Deduplicate by text content
    seen = set()
    unique_hits = []
    for hit in all_hits:
        text_key = hit["text"][:100]  # first 100 chars as dedup key
        if text_key not in seen:
            seen.add(text_key)
            unique_hits.append(hit)

    # Take top k by relevance (lowest distance = most similar)
    unique_hits.sort(key=lambda h: h["distance"])
    top_hits = unique_hits[:k]

    if not top_hits:
        return ""

    # Format as injectable context block
    lines = ["INSTITUTIONAL KNOWLEDGE BASE:"]
    for i, hit in enumerate(top_hits, 1):
        source = hit["metadata"].get("source", "knowledge base")
        lines.append(f"\n[{i}] Source: {source}")
        lines.append(hit["text"])

    return "\n".join(lines)

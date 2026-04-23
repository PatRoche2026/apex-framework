"""HyDE (Hypothetical Document Embeddings) wrapper for APEX RAG retrieval.

Generates a hypothetical ideal PubMed abstract via Claude Haiku, embeds it
with Voyage AI, then searches ChromaDB with that embedding instead of the
raw query — producing semantically closer matches for novel targets.

Falls back to standard retrieve_context() on any failure.
"""

from __future__ import annotations

import logging

import anthropic

from agents import ANTHROPIC_API_KEY, LLM_MODEL_FAST
from .embeddings import embed_texts
from .store import query_collection_by_embedding
from .retriever import retrieve_context

logger = logging.getLogger(__name__)

HYDE_PROMPT = """\
You are a biomedical researcher. Write a hypothetical PubMed abstract for a \
paper that would be the ideal reference for evaluating {query} as a drug target. \
Include gene function, disease mechanism, key experimental findings, and \
therapeutic implications. Write only the abstract, no title or metadata."""


def generate_hypothetical_abstract(query: str) -> str:
    """Call Claude Haiku to generate a hypothetical PubMed abstract for the query."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=LLM_MODEL_FAST,
        max_tokens=512,
        temperature=0.4,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
    )
    return response.content[0].text


def hyde_retrieve_context(role: str, query: str, k: int = 5) -> str:
    """HyDE-enhanced retrieval — drop-in replacement for retrieve_context().

    1. Generate a hypothetical abstract via Haiku
    2. Embed it as a document (Voyage AI)
    3. Search role-specific + shared ChromaDB collections
    4. Deduplicate and format

    On ANY failure in steps 1-2, logs a warning and falls back to standard
    retrieve_context(). APEX must never crash due to a HyDE failure.
    """
    try:
        hyp_abstract = generate_hypothetical_abstract(query)
        hyp_embedding = embed_texts([hyp_abstract])[0]
    except Exception as exc:
        logger.warning("HyDE generation failed, falling back to standard retrieval: %s", exc)
        return retrieve_context(role, query, k)

    # Query role-specific knowledge
    role_hits = query_collection_by_embedding(role, hyp_embedding, k=k)

    # Query shared knowledge
    shared_hits = query_collection_by_embedding("shared", hyp_embedding, k=max(2, k // 2))

    # Merge, deduplicate, sort by relevance
    all_hits = role_hits + shared_hits
    seen: set[str] = set()
    unique_hits: list[dict] = []
    for hit in all_hits:
        text_key = hit["text"][:100]
        if text_key not in seen:
            seen.add(text_key)
            unique_hits.append(hit)

    unique_hits.sort(key=lambda h: h["distance"])
    top_hits = unique_hits[:k]

    if not top_hits:
        return ""

    lines = ["INSTITUTIONAL KNOWLEDGE BASE (HyDE-enhanced):"]
    for i, hit in enumerate(top_hits, 1):
        source = hit["metadata"].get("source", "knowledge base")
        lines.append(f"\n[{i}] Source: {source}")
        lines.append(hit["text"])

    return "\n".join(lines)

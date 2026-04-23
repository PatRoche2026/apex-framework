"""CRAG (Corrective RAG) — quality-graded retrieval with PubMed escalation.

Grades ChromaDB retrieval results by cosine distance. If too few hits pass
the quality threshold, auto-escalates to live PubMed search to fill the gap.
"""

from __future__ import annotations

import asyncio
import logging

from agents import PUBMED_RATE_LIMIT_DELAY
from .store import query_collection

logger = logging.getLogger(__name__)

# Cosine distance threshold: hits below this are RELEVANT, above are IRRELEVANT.
# ChromaDB cosine distance range: 0.0 (identical) to 2.0 (opposite).
# 0.7 ≈ similarity 0.3 — a reasonable cutoff for weak matches.
DEFAULT_THRESHOLD = 0.7

# Minimum number of relevant hits before triggering PubMed escalation.
MIN_RELEVANT_HITS = 2


async def crag_retrieve_context(
    role: str,
    query: str,
    k: int = 5,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Quality-graded RAG retrieval with PubMed fallback.

    Pipeline:
    1. Query ChromaDB (role-specific + shared collections)
    2. Grade each hit: distance < threshold = RELEVANT, else IRRELEVANT
    3. If fewer than MIN_RELEVANT_HITS pass: escalate to live PubMed search
    4. Return formatted context with source annotations

    Args:
        role: Agent role (cso, cto, cmo, cbo).
        query: Search query string.
        k: Max results to return.
        threshold: Cosine distance threshold (default 0.7).

    Returns:
        Formatted context string ready for prompt injection.
    """
    # Step 1: Query ChromaDB
    role_hits = query_collection(role, query, k=k)
    shared_hits = query_collection("shared", query, k=max(2, k // 2))

    # Merge and deduplicate
    all_hits = role_hits + shared_hits
    seen: set[str] = set()
    unique_hits: list[dict] = []
    for hit in all_hits:
        text_key = hit["text"][:100]
        if text_key not in seen:
            seen.add(text_key)
            unique_hits.append(hit)

    unique_hits.sort(key=lambda h: h["distance"])

    # Step 2: Grade by cosine distance
    relevant = [h for h in unique_hits if h["distance"] < threshold]
    irrelevant = [h for h in unique_hits if h["distance"] >= threshold]

    logger.info(
        "CRAG grading for '%s' (role=%s): %d relevant, %d irrelevant (threshold=%.2f)",
        query[:60], role, len(relevant), len(irrelevant), threshold,
    )

    # Step 3: Escalate to PubMed if insufficient relevant hits
    pubmed_supplement = ""
    if len(relevant) < MIN_RELEVANT_HITS:
        logger.info(
            "CRAG escalation: only %d relevant hits (need %d), querying PubMed...",
            len(relevant), MIN_RELEVANT_HITS,
        )
        pubmed_supplement = await _pubmed_escalation(query)

    # Step 4: Determine retrieval quality tier
    top_relevant = relevant[:k]
    has_chromadb = len(top_relevant) > 0
    has_pubmed = len(pubmed_supplement) > 0

    if has_chromadb and not has_pubmed:
        retrieval_quality = "HIGH (ChromaDB)"
    elif has_chromadb and has_pubmed:
        retrieval_quality = "MIXED (ChromaDB + PubMed)"
    elif has_pubmed:
        retrieval_quality = "LOW (PubMed only)"
    else:
        return ""

    # Step 5: Format output with quality header
    lines: list[str] = [
        f"RETRIEVAL_QUALITY: {retrieval_quality}",
        "Use this quality indicator when producing your Evidence Reflection tags:",
        "- HIGH = claims supported by this context can be tagged [SUPPORTED]",
        "- MIXED = cross-check ChromaDB and PubMed sources before tagging [SUPPORTED]",
        "- LOW = treat all claims based on this context as [UNCERTAIN] unless independently verified",
        "",
    ]

    if top_relevant:
        lines.append("INSTITUTIONAL KNOWLEDGE BASE (quality-graded):")
        for i, hit in enumerate(top_relevant, 1):
            source = hit["metadata"].get("source", "knowledge base")
            dist = hit["distance"]
            lines.append(f"\n[{i}] Source: {source} (relevance: {1 - dist:.2f})")
            lines.append(hit["text"])

    if has_pubmed:
        lines.append("")
        lines.append("LIVE PUBMED SUPPLEMENT (auto-retrieved — knowledge base had insufficient coverage):")
        lines.append(pubmed_supplement)

    return "\n".join(lines)


async def _pubmed_escalation(query: str, max_results: int = 3) -> str:
    """Search PubMed as fallback when ChromaDB retrieval quality is low."""
    from agents.scout import search_pubmed

    try:
        papers = await asyncio.to_thread(search_pubmed, query, max_results)
        if not papers:
            return ""

        lines = []
        for p in papers:
            lines.append(
                f"\nPMID {p['pmid']}: {p['title']}\n"
                f"Journal: {p['journal']} ({p['year']})\n"
                f"Abstract: {p['abstract'][:500]}"
            )
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("CRAG PubMed escalation failed: %s", exc)
        return ""

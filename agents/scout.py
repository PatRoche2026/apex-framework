"""Research Scout node — multi-query PubMed search + LLM relevance filter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Bio import Entrez, Medline
from langchain_anthropic import ChatAnthropic

from agents import (
    ANTHROPIC_API_KEY,
    ENTREZ_EMAIL,
    LLM_TIMEOUT_SECONDS,
    PUBMED_RATE_LIMIT_DELAY,
    ROLE_MODELS,
    estimate_cost,
    llm_semaphore,
)
from agents.state import APEXState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

Entrez.email = ENTREZ_EMAIL
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# PubMed search
# ---------------------------------------------------------------------------


def search_pubmed(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search PubMed and return structured article data."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    time.sleep(PUBMED_RATE_LIMIT_DELAY)

    id_list = record.get("IdList", [])
    if not id_list:
        return []

    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()
    time.sleep(PUBMED_RATE_LIMIT_DELAY)

    results = []
    for rec in records:
        authors_list = rec.get("AU", [])
        results.append({
            "pmid": rec.get("PMID", ""),
            "title": rec.get("TI", ""),
            "authors": ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else ""),
            "journal": rec.get("JT", rec.get("TA", "")),
            "year": rec.get("DP", "").split()[0] if rec.get("DP") else "",
            "abstract": rec.get("AB", "No abstract available."),
        })
    return results


# ---------------------------------------------------------------------------
# Multi-query strategy: 3 variants, deduplicate by PMID
# ---------------------------------------------------------------------------


def _parse_target_indication(query: str) -> tuple[str, str]:
    """Best-effort extraction of target and indication from the query string."""
    # Try patterns like "X for Y", "X as target for Y", "X in Y"
    for pattern in [
        r"(?i)(?:evaluate\s+)?(.+?)\s+(?:as\s+(?:a\s+)?(?:therapeutic\s+)?target\s+)?(?:for|in)\s+(.+)",
        r"(?i)(.+?)\s+(?:for|in)\s+(.+)",
    ]:
        m = re.match(pattern, query.strip())
        if m:
            return m.group(1).strip(), m.group(2).strip()
    # Fallback: use the whole query as both
    return query.strip(), ""


def search_pubmed_multi(query: str) -> list[dict[str, Any]]:
    """Run 3 PubMed query variants and deduplicate results by PMID."""
    target, indication = _parse_target_indication(query)

    queries = [query]  # Always include the raw user query
    if indication:
        queries.append(f"{target} {indication}")
        queries.append(f"{target} mechanism of action")
        queries.append(f"{target} clinical trial")
    else:
        queries.append(f"{target} therapeutic target")
        queries.append(f"{target} clinical trial")

    seen_pmids: set[str] = set()
    all_results: list[dict] = []

    for q in queries[:3]:  # Cap at 3 queries
        try:
            results = search_pubmed(q, max_results=10)
            for r in results:
                if r["pmid"] and r["pmid"] not in seen_pmids:
                    seen_pmids.add(r["pmid"])
                    all_results.append(r)
        except Exception:
            continue  # Don't let one query failure kill the whole scout

    return all_results


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(query: str) -> str:
    """Normalize query into a deterministic cache filename."""
    normalized = " ".join(query.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def _load_cache(query: str) -> dict | None:
    """Load cached scout data if available."""
    path = CACHE_DIR / f"{_cache_key(query)}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(query: str, scout_data: str, scout_sources: list[dict]) -> None:
    """Save scout data to cache."""
    path = CACHE_DIR / f"{_cache_key(query)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scout_data": scout_data, "scout_sources": scout_sources}, f, indent=2)


# ---------------------------------------------------------------------------
# LLM relevance filter — rank and select top papers
# ---------------------------------------------------------------------------

RELEVANCE_FILTER_PROMPT = """You are a biomedical research librarian. A team of biotech executives needs to evaluate a drug target.

QUERY: {query}

Below are {n_papers} PubMed abstracts retrieved for this query. Select the TOP 8 most relevant papers for evaluating this drug target's viability. Rank them by relevance.

For each selected paper, output EXACTLY this format (one block per paper):

PMID: <pmid>
RELEVANCE: <one sentence explaining why this paper matters for the evaluation>

---

PAPERS:
{papers_text}

Select the 8 most relevant. If fewer than 8 are relevant, select all relevant ones."""


async def _filter_relevance(
    papers: list[dict], query: str, provider: str = "anthropic"
) -> tuple[list[dict], float]:
    """Use LLM to rank and filter papers by relevance to the query.

    Args:
        papers: list of PubMed paper dicts.
        query: the original user query.
        provider: LLM backend (anthropic | openai | google). Defaults to
            anthropic for backward compatibility.

    Returns:
        (filtered_papers, cost_usd)
    """
    if len(papers) <= 8:
        return papers, 0.0  # No filtering needed

    papers_text = ""
    for i, p in enumerate(papers, 1):
        abstract_snippet = p["abstract"][:400] + ("..." if len(p["abstract"]) > 400 else "")
        papers_text += f"\n[{i}] PMID: {p['pmid']}\nTitle: {p['title']}\nJournal: {p['journal']} ({p['year']})\nAbstract: {abstract_snippet}\n"

    prompt = RELEVANCE_FILTER_PROMPT.format(
        query=query,
        n_papers=len(papers),
        papers_text=papers_text,
    )

    from agents.llm_router import get_llm as router_get_llm, estimate_cost_from_response
    llm = router_get_llm("scout", provider=provider, temperature=0.1, max_tokens=2000)
    call_cost = 0.0

    async with llm_semaphore:
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            call_cost = estimate_cost_from_response(provider, "scout", response)
        except (asyncio.TimeoutError, Exception):
            return papers[:8], 0.0  # Fallback: just take first 8

    # Parse selected PMIDs from response
    selected_pmids = re.findall(r"PMID:\s*(\d+)", response.content)
    pmid_set = set(selected_pmids[:8])

    # Return papers in the LLM's ranked order
    pmid_to_paper = {p["pmid"]: p for p in papers}
    filtered = [pmid_to_paper[pid] for pid in selected_pmids[:8] if pid in pmid_to_paper]

    # If LLM didn't return enough, pad with remaining papers
    if len(filtered) < 5:
        remaining = [p for p in papers if p["pmid"] not in pmid_set]
        filtered.extend(remaining[: 8 - len(filtered)])

    return filtered, call_cost


# ---------------------------------------------------------------------------
# Format papers for executive consumption
# ---------------------------------------------------------------------------


def _format_as_prose(papers: list[dict], query: str) -> str:
    """Format papers into a structured research brief for the executives."""
    lines = [f"# Research Scout Brief: {query}", f"**Papers analyzed:** {len(papers)}", ""]
    for i, p in enumerate(papers, 1):
        lines.append(f"## [{i}] {p['title']}")
        lines.append(f"**{p['journal']}** ({p['year']}) | PMID: {p['pmid']} | {p['authors']}")
        full_text = p.get("full_text", "")
        if full_text:
            lines.append(f"\n**[FULL TEXT from PMC]**\n{full_text}\n")
        else:
            lines.append(f"\n{p['abstract']}\n")
        lines.append("---")
    return "\n".join(lines)


def _format_as_sources(papers: list[dict]) -> list[dict]:
    """Format papers into structured source list for frontend."""
    return [
        {
            "pmid": p["pmid"],
            "title": p["title"],
            "journal": p["journal"],
            "year": p["year"],
            "abstract_snippet": p["abstract"][:200] + ("..." if len(p["abstract"]) > 200 else ""),
        }
        for p in papers
    ]


# ---------------------------------------------------------------------------
# Scout node for LangGraph
# ---------------------------------------------------------------------------


async def scout_node(state: APEXState) -> dict:
    """Research Scout: search PubMed, filter for relevance, return structured data."""
    query = state["query"]
    provider = state.get("provider") or "anthropic"
    ts = datetime.now(timezone.utc).isoformat()

    # Check cache first
    cached = _load_cache(query)
    if cached:
        return {
            "scout_data": cached["scout_data"],
            "scout_sources": cached["scout_sources"],
            "activity_log": [{"node": "scout", "status": "complete", "source": "cache", "timestamp": ts}],
        }

    # Run multi-query PubMed search
    raw_papers = search_pubmed_multi(query)

    if not raw_papers:
        return {
            "scout_data": f"No PubMed results found for: {query}",
            "scout_sources": [],
            "activity_log": [{"node": "scout", "status": "complete", "n_papers": 0, "timestamp": ts}],
        }

    # LLM relevance filter
    filtered_papers, filter_cost = await _filter_relevance(raw_papers, query, provider=provider)

    # Enrich top 3 papers with PMC full text (lazy import to avoid circular dependency)
    from agents.tools import _pmids_to_pmc_ids, _fetch_pmc_full_text

    top_pmids = [p["pmid"] for p in filtered_papers[:3] if p["pmid"]]
    pmc_map: dict[str, str] = {}
    if top_pmids:
        try:
            pmc_map = await asyncio.to_thread(_pmids_to_pmc_ids, top_pmids)
        except Exception:
            pass  # Full-text enrichment is best-effort

    full_text_count = 0
    for p in filtered_papers[:3]:
        pmc_id = pmc_map.get(p["pmid"])
        if pmc_id:
            try:
                full_text = await asyncio.to_thread(_fetch_pmc_full_text, pmc_id, 3000)
                if full_text:
                    p["full_text"] = full_text
                    full_text_count += 1
            except Exception:
                pass

    # Format outputs
    scout_data = _format_as_prose(filtered_papers, query)
    scout_sources = _format_as_sources(filtered_papers)

    # Cache for future use
    _save_cache(query, scout_data, scout_sources)

    return {
        "scout_data": scout_data,
        "scout_sources": scout_sources,
        "activity_log": [
            {
                "node": "scout",
                "status": "complete",
                "n_papers_raw": len(raw_papers),
                "n_papers_filtered": len(filtered_papers),
                "n_full_text": full_text_count,
                "cost_usd": filter_cost,
                "timestamp": ts,
            }
        ],
    }

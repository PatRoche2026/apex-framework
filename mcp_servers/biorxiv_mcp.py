"""bioRxiv/medRxiv MCP Server — exposes preprint search and detail retrieval
as FastMCP tools.

Run standalone:
    python -m mcp_servers.biorxiv_mcp
    # or
    fastmcp run mcp_servers/biorxiv_mcp.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx
from fastmcp import FastMCP

mcp = FastMCP("bioRxiv")

# bioRxiv content detail API (returns papers by date range + server)
BIORXIV_API = "https://api.biorxiv.org/details"
# bioRxiv search is not officially supported via API — use content detail
# with date ranges and filter client-side, or use the pubs API
BIORXIV_PUBS_API = "https://api.biorxiv.org/pubs"

_HEADERS = {"User-Agent": "APEX-BiotechAgent/1.0 (patroche@mit.edu)"}
_TIMEOUT = 20.0


# ---------------------------------------------------------------------------
# Tool 1: search_biorxiv
# ---------------------------------------------------------------------------

@mcp.tool()
def search_biorxiv(query: str, max_results: int = 5) -> str:
    """Search bioRxiv and medRxiv for recent preprints matching a query.

    Uses the bioRxiv content API to retrieve recent preprints and filters
    by keyword matching in title and abstract. Covers the last 90 days.

    Args:
        query: Biomedical search query (e.g. "OSMR ulcerative colitis").
        max_results: Maximum number of preprints to return (default 5, max 15).
    """
    max_results = min(max_results, 15)

    # bioRxiv API requires date ranges — search last 90 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    query_terms = query.lower().split()
    all_results: list[dict] = []

    for server in ("biorxiv", "medrxiv"):
        try:
            # The API paginates at 100 results per call
            # Fetch up to 100 recent papers and filter client-side
            url = f"{BIORXIV_API}/{server}/{start_date}/{end_date}/0/100"

            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

            papers = data.get("collection", [])

            for paper in papers:
                title = paper.get("title", "").lower()
                abstract = paper.get("abstract", "").lower()
                combined = title + " " + abstract

                # Match if any query term appears in title or abstract
                if any(term in combined for term in query_terms):
                    all_results.append({
                        "doi": paper.get("doi", "N/A"),
                        "title": paper.get("title", "No title"),
                        "authors": paper.get("authors", "N/A"),
                        "date": paper.get("date", "N/A"),
                        "category": paper.get("category", "N/A"),
                        "abstract": paper.get("abstract", "No abstract"),
                        "server": server,
                        "version": paper.get("version", "1"),
                    })

        except Exception:
            # Silently skip server if API fails — try the other one
            continue

    if not all_results:
        return (
            f"No bioRxiv/medRxiv preprints found for '{query}' in the last 90 days. "
            "Try broader search terms or check PubMed for peer-reviewed articles."
        )

    # Sort by date (newest first) and limit
    all_results.sort(key=lambda p: p["date"], reverse=True)
    top = all_results[:max_results]

    lines = [
        f"bioRxiv/medRxiv results for '{query}' "
        f"({len(top)} of {len(all_results)} matches, last 90 days):"
    ]
    for paper in top:
        # Truncate author list
        authors = paper["authors"]
        if len(authors) > 100:
            authors = authors[:100] + "..."

        lines.append(
            f"\nDOI: {paper['doi']}\n"
            f"Title: {paper['title']}\n"
            f"Authors: {authors}\n"
            f"Date: {paper['date']} | Server: {paper['server']} | "
            f"Category: {paper['category']}\n"
            f"Abstract: {paper['abstract'][:500]}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: get_preprint_details
# ---------------------------------------------------------------------------

@mcp.tool()
def get_preprint_details(doi: str) -> str:
    """Fetch detailed metadata for a specific bioRxiv or medRxiv preprint.

    Returns title, authors, abstract, dates, category, published DOI
    (if peer-reviewed version exists), and version history.

    Args:
        doi: bioRxiv/medRxiv DOI (e.g. "10.1101/2024.01.15.123456").
    """
    doi = doi.strip()

    # Try the pubs API first — it maps preprint DOIs to published versions
    published_doi = None
    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
            response = client.get(f"{BIORXIV_PUBS_API}/biorxiv/{doi}")
            response.raise_for_status()
            pubs_data = response.json()

        pubs_collection = pubs_data.get("collection", [])
        if pubs_collection:
            published_doi = pubs_collection[0].get("published_doi")
    except Exception:
        pass

    # Fetch the preprint details via content API
    # The DOI-based endpoint: /details/{server}/{doi}
    paper = None
    for server in ("biorxiv", "medrxiv"):
        try:
            url = f"{BIORXIV_API}/{server}/{doi}/na/na"
            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

            collection = data.get("collection", [])
            if collection:
                paper = collection[-1]  # Latest version
                paper["server"] = server
                break

        except Exception:
            continue

    if not paper:
        return (
            f"Preprint with DOI '{doi}' not found on bioRxiv or medRxiv. "
            "Verify the DOI is correct and includes the '10.1101/' prefix."
        )

    lines = [f"Preprint Details: {doi}"]
    lines.append(f"Title: {paper.get('title', 'N/A')}")
    lines.append(f"Authors: {paper.get('authors', 'N/A')}")
    lines.append(f"Date: {paper.get('date', 'N/A')}")
    lines.append(f"Server: {paper.get('server', 'N/A')}")
    lines.append(f"Category: {paper.get('category', 'N/A')}")
    lines.append(f"Version: {paper.get('version', 'N/A')}")
    lines.append(f"License: {paper.get('license', 'N/A')}")

    if published_doi:
        lines.append(f"Published (peer-reviewed) DOI: {published_doi}")
    else:
        lines.append("Published version: Not yet published in a peer-reviewed journal")

    abstract = paper.get("abstract", "No abstract available.")
    lines.append(f"\nAbstract:\n{abstract}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

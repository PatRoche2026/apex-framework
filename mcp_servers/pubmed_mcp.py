"""PubMed MCP Server — exposes PubMed E-utilities search and PMC full-text
retrieval as FastMCP tools.

Run standalone:
    python -m mcp_servers.pubmed_mcp
    # or
    fastmcp run mcp_servers/pubmed_mcp.py
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

from Bio import Entrez, Medline
from fastmcp import FastMCP

from agents import ENTREZ_EMAIL, PUBMED_RATE_LIMIT_DELAY

Entrez.email = ENTREZ_EMAIL

mcp = FastMCP("PubMed")


# ---------------------------------------------------------------------------
# Tool 1: search_pubmed
# ---------------------------------------------------------------------------

@mcp.tool()
def search_pubmed(query: str, max_results: int = 5) -> str:
    """Search PubMed E-utilities for papers matching a biomedical query.

    Returns PMIDs, titles, journals, years, and abstracts for the top results
    ranked by relevance.

    Args:
        query: Biomedical search query (e.g. "OSMR ulcerative colitis").
        max_results: Maximum number of papers to return (default 5, max 20).
    """
    max_results = min(max_results, 20)

    try:
        handle = Entrez.esearch(
            db="pubmed", term=query, retmax=max_results, sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        time.sleep(PUBMED_RATE_LIMIT_DELAY)

        id_list = record.get("IdList", [])
        if not id_list:
            return f"No PubMed results found for: {query}"

        handle = Entrez.efetch(
            db="pubmed", id=",".join(id_list), rettype="medline", retmode="text"
        )
        records = list(Medline.parse(handle))
        handle.close()
        time.sleep(PUBMED_RATE_LIMIT_DELAY)

        lines = [f"PubMed results for '{query}' ({len(records)} papers):"]
        for rec in records:
            pmid = rec.get("PMID", "N/A")
            title = rec.get("TI", "No title")
            journal = rec.get("JT", rec.get("TA", "Unknown"))
            year = rec.get("DP", "").split()[0] if rec.get("DP") else "N/A"
            abstract = rec.get("AB", "No abstract available.")

            lines.append(
                f"\nPMID: {pmid}\n"
                f"Title: {title}\n"
                f"Journal: {journal} ({year})\n"
                f"Abstract: {abstract}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"PubMed search failed: {str(e)[:300]}"


# ---------------------------------------------------------------------------
# Tool 2: fetch_full_text
# ---------------------------------------------------------------------------

def _pmid_to_pmc_id(pmid: str) -> str | None:
    """Use Entrez.elink to find the PMC ID for a given PubMed ID.

    Returns the numeric PMC ID (without 'PMC' prefix) or None.
    """
    try:
        handle = Entrez.elink(
            dbfrom="pubmed", db="pmc", id=[pmid], linkname="pubmed_pmc"
        )
        records = Entrez.read(handle)
        handle.close()
        time.sleep(PUBMED_RATE_LIMIT_DELAY)
    except Exception:
        return None

    for record in records:
        links = record.get("LinkSetDb", [])
        if links:
            pmc_links = links[0].get("Link", [])
            if pmc_links:
                return str(pmc_links[0]["Id"])
    return None


def _fetch_pmc_body(pmc_id: str, max_chars: int = 5000) -> str:
    """Fetch full text from PMC and extract body text from XML."""
    try:
        handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="xml", retmode="xml")
        xml_bytes = handle.read()
        handle.close()
        time.sleep(PUBMED_RATE_LIMIT_DELAY)
    except Exception:
        return ""

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    body = root.find(".//body")
    if body is None:
        return ""

    parts: list[str] = []
    for elem in body.iter():
        if elem.tag == "title" and elem.text:
            parts.append(f"\n### {elem.text.strip()}\n")
        elif elem.tag == "p":
            text = "".join(elem.itertext()).strip()
            if text:
                parts.append(text)

    full_text = "\n".join(parts)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n[...truncated]"

    return full_text


@mcp.tool()
def fetch_full_text(pmid: str) -> str:
    """Fetch the full text of a PubMed paper via PMC (PubMed Central).

    Looks up the PMC ID for the given PMID, then retrieves and extracts
    the body text from the PMC XML. Returns the full text if available,
    or a message explaining why it could not be retrieved.

    Args:
        pmid: PubMed ID (numeric string, e.g. "12345678").
    """
    pmid = pmid.strip().lstrip("PMID:").strip()

    try:
        # Step 1: Map PMID → PMC ID
        pmc_id = _pmid_to_pmc_id(pmid)
        if not pmc_id:
            return (
                f"PMID {pmid}: No PMC full text available. "
                "This paper may not be open access or may not be indexed in PMC."
            )

        # Step 2: Fetch full text from PMC
        full_text = _fetch_pmc_body(pmc_id)
        if not full_text:
            return (
                f"PMID {pmid} (PMC{pmc_id}): PMC entry exists but body text "
                "could not be extracted (may be a scanned PDF or non-standard format)."
            )

        return (
            f"PMID {pmid} | PMC{pmc_id} | FULL TEXT\n"
            f"{'=' * 60}\n"
            f"{full_text}"
        )

    except Exception as e:
        return f"Full text retrieval failed for PMID {pmid}: {str(e)[:300]}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

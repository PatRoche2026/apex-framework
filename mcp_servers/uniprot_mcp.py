"""UniProt MCP Server — exposes UniProt protein search and entry retrieval
as FastMCP tools.

Run standalone:
    python -m mcp_servers.uniprot_mcp
    # or
    fastmcp run mcp_servers/uniprot_mcp.py
"""

from __future__ import annotations

import httpx
from fastmcp import FastMCP

mcp = FastMCP("UniProt")

UNIPROT_SEARCH_API = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_API = "https://rest.uniprot.org/uniprotkb"
_HEADERS = {"User-Agent": "APEX-BiotechAgent/1.0 (patroche@mit.edu)"}
_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Tool 1: search_uniprot
# ---------------------------------------------------------------------------

@mcp.tool()
def search_uniprot(query: str, max_results: int = 3) -> str:
    """Search UniProt for protein entries matching a gene or protein name.

    Returns accession IDs, protein names, gene names, organism, function
    descriptions, and subcellular location.

    Args:
        query: Gene symbol or protein name (e.g. "OSMR", "oncostatin M receptor").
        max_results: Maximum number of entries to return (default 3, max 10).
    """
    max_results = min(max_results, 10)

    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
            response = client.get(
                UNIPROT_SEARCH_API,
                params={
                    "query": query,
                    "format": "json",
                    "size": max_results,
                    "fields": (
                        "accession,protein_name,gene_names,organism_name,"
                        "cc_function,cc_subcellular_location,length,cc_tissue_specificity"
                    ),
                },
            )
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        if not results:
            return f"No UniProt results found for: {query}"

        lines = [f"UniProt results for '{query}' ({len(results)} entries):"]
        for entry in results:
            acc = entry.get("primaryAccession", "N/A")

            # Protein name
            protein = entry.get("proteinDescription", {})
            rec_name = (
                protein.get("recommendedName", {})
                .get("fullName", {})
                .get("value", "Unknown")
            )

            # Gene names
            genes = [
                g.get("geneName", {}).get("value", "")
                for g in entry.get("genes", [])
            ]
            gene_str = ", ".join(g for g in genes if g) or "N/A"

            # Organism
            organism = entry.get("organism", {}).get("scientificName", "N/A")

            # Length
            length = entry.get("sequence", {}).get("length", "N/A")

            lines.append(
                f"\nAccession: {acc}\n"
                f"Protein: {rec_name}\n"
                f"Gene(s): {gene_str}\n"
                f"Organism: {organism}\n"
                f"Length: {length} aa"
            )

            # Function
            func_comments = [
                c for c in entry.get("comments", [])
                if c.get("commentType") == "FUNCTION"
            ]
            if func_comments:
                texts = func_comments[0].get("texts", [])
                if texts:
                    lines.append(f"Function: {texts[0].get('value', '')[:500]}")

            # Subcellular location
            loc_comments = [
                c for c in entry.get("comments", [])
                if c.get("commentType") == "SUBCELLULAR LOCATION"
            ]
            if loc_comments:
                locs = loc_comments[0].get("subcellularLocations", [])
                loc_names = [
                    sl.get("location", {}).get("value", "")
                    for sl in locs
                ]
                loc_str = ", ".join(l for l in loc_names if l)
                if loc_str:
                    lines.append(f"Location: {loc_str}")

            # Tissue specificity
            tissue_comments = [
                c for c in entry.get("comments", [])
                if c.get("commentType") == "TISSUE SPECIFICITY"
            ]
            if tissue_comments:
                texts = tissue_comments[0].get("texts", [])
                if texts:
                    lines.append(f"Tissue specificity: {texts[0].get('value', '')[:300]}")

        return "\n".join(lines)

    except Exception as e:
        return f"UniProt search failed: {str(e)[:300]}"


# ---------------------------------------------------------------------------
# Tool 2: get_protein_entry
# ---------------------------------------------------------------------------

@mcp.tool()
def get_protein_entry(accession: str) -> str:
    """Fetch a detailed UniProt protein entry by accession ID.

    Returns protein name, gene names, function, subcellular location,
    tissue specificity, involvement in disease, and key features.

    Args:
        accession: UniProt accession ID (e.g. "Q99650" for OSMR).
    """
    accession = accession.strip().upper()

    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
            response = client.get(
                f"{UNIPROT_ENTRY_API}/{accession}.json",
            )
            response.raise_for_status()
            entry = response.json()

        lines = [f"UniProt Entry: {accession}"]

        # Protein name
        protein = entry.get("proteinDescription", {})
        rec_name = (
            protein.get("recommendedName", {})
            .get("fullName", {})
            .get("value", "Unknown")
        )
        lines.append(f"Protein: {rec_name}")

        # Gene names
        genes = [
            g.get("geneName", {}).get("value", "")
            for g in entry.get("genes", [])
        ]
        lines.append(f"Gene(s): {', '.join(g for g in genes if g) or 'N/A'}")

        # Organism
        organism = entry.get("organism", {}).get("scientificName", "N/A")
        lines.append(f"Organism: {organism}")

        # Length
        length = entry.get("sequence", {}).get("length", "N/A")
        lines.append(f"Length: {length} aa")

        # All comments by type
        comments = entry.get("comments", [])

        for comment_type, label in [
            ("FUNCTION", "Function"),
            ("SUBCELLULAR LOCATION", "Subcellular Location"),
            ("TISSUE SPECIFICITY", "Tissue Specificity"),
            ("INVOLVEMENT IN DISEASE", "Disease Involvement"),
            ("PATHWAY", "Pathway"),
            ("SUBUNIT", "Subunit Structure"),
        ]:
            matching = [c for c in comments if c.get("commentType") == comment_type]
            if not matching:
                continue

            c = matching[0]
            if comment_type == "SUBCELLULAR LOCATION":
                locs = c.get("subcellularLocations", [])
                loc_names = [
                    sl.get("location", {}).get("value", "")
                    for sl in locs
                ]
                val = ", ".join(l for l in loc_names if l)
            elif comment_type == "INVOLVEMENT IN DISEASE":
                diseases = c.get("disease", {})
                if diseases:
                    val = (
                        f"{diseases.get('diseaseId', '')} — "
                        f"{diseases.get('description', '')[:500]}"
                    )
                else:
                    texts = c.get("texts", [])
                    val = texts[0].get("value", "")[:500] if texts else ""
            else:
                texts = c.get("texts", [])
                val = texts[0].get("value", "")[:500] if texts else ""

            if val:
                lines.append(f"\n{label}:\n{val}")

        # Key features (domains, binding sites, active sites)
        features = entry.get("features", [])
        interesting = [
            f for f in features
            if f.get("type") in (
                "Domain", "Binding site", "Active site", "Signal peptide",
                "Transmembrane", "Disulfide bond",
            )
        ]
        if interesting:
            lines.append("\nKey Features:")
            for f in interesting[:15]:
                loc = f.get("location", {})
                start = loc.get("start", {}).get("value", "?")
                end = loc.get("end", {}).get("value", "?")
                desc = f.get("description", "")
                lines.append(
                    f"  - {f.get('type', 'N/A')} [{start}-{end}]"
                    f"{': ' + desc if desc else ''}"
                )

        return "\n".join(lines)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"UniProt accession {accession} not found."
        return f"UniProt request failed: {str(e)[:300]}"
    except Exception as e:
        return f"UniProt entry retrieval failed for {accession}: {str(e)[:300]}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

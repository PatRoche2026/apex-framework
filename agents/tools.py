"""Autonomous tool use for executive agents — PubMed, ClinicalTrials.gov, Open Targets."""

from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from Bio import Entrez

from agents import ENTREZ_EMAIL, PUBMED_RATE_LIMIT_DELAY

Entrez.email = ENTREZ_EMAIL

# ---------------------------------------------------------------------------
# Rate limit for external API calls
# ---------------------------------------------------------------------------

_TOOL_RATE_LIMIT = PUBMED_RATE_LIMIT_DELAY  # 0.5s between calls
_tool_lock = asyncio.Lock()


_HEADERS = {"User-Agent": "APEX-BiotechAgent/1.0 (patroche@mit.edu)"}


async def _rate_limited_get(url: str, params: dict | None = None, timeout: float = 15.0) -> httpx.Response:
    """Make a rate-limited async GET request."""
    async with _tool_lock:
        async with httpx.AsyncClient(timeout=timeout, headers=_HEADERS) as client:
            response = await client.get(url, params=params)
            await asyncio.sleep(_TOOL_RATE_LIMIT)
            return response


async def _rate_limited_post(url: str, json: dict | None = None, timeout: float = 15.0) -> httpx.Response:
    """Make a rate-limited async POST request."""
    async with _tool_lock:
        async with httpx.AsyncClient(timeout=timeout, headers=_HEADERS) as client:
            response = await client.post(url, json=json)
            await asyncio.sleep(_TOOL_RATE_LIMIT)
            return response


# ---------------------------------------------------------------------------
# Tool 1: PubMed search (reused from scout.py)
# ---------------------------------------------------------------------------

async def search_pubmed_tool(query: str, max_results: int = 5) -> str:
    """Search PubMed for papers matching the query. Returns formatted results."""
    from agents.scout import search_pubmed

    try:
        papers = search_pubmed(query, max_results=max_results)
        if not papers:
            return f"No PubMed results found for: {query}"

        lines = [f"PubMed results for '{query}' ({len(papers)} papers):"]
        for p in papers:
            lines.append(
                f"- PMID {p['pmid']}: {p['title']} — {p['journal']} ({p['year']}). "
                f"{p.get('abstract_snippet', '')[:200]}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"PubMed search failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Tool 2: ClinicalTrials.gov API v2
# ---------------------------------------------------------------------------

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"


async def search_clinical_trials(query: str, max_results: int = 5) -> str:
    """Search ClinicalTrials.gov for studies matching the query."""
    try:
        response = await _rate_limited_get(
            CTGOV_API,
            params={
                "query.term": query,
                "pageSize": max_results,
                "format": "json",
            },
        )
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        if not studies:
            return f"No clinical trials found for: {query}"

        lines = [f"ClinicalTrials.gov results for '{query}' ({len(studies)} studies):"]
        for study in studies:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})

            nct_id = ident.get("nctId", "N/A")
            title = ident.get("briefTitle", "No title")
            status = status_mod.get("overallStatus", "Unknown")
            phase_list = design.get("phases", [])
            phase = ", ".join(phase_list) if phase_list else "N/A"

            lines.append(f"- {nct_id}: {title} | Status: {status} | Phase: {phase}")

        return "\n".join(lines)
    except Exception as e:
        return f"ClinicalTrials.gov search failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Tool 3: Open Targets Platform (GraphQL API)
# ---------------------------------------------------------------------------

OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"


async def search_open_targets(target_gene: str, disease: str) -> str:
    """Query Open Targets for target-disease association evidence."""
    # GraphQL query for target-disease associations
    query = """
    query targetDiseaseAssociation($ensemblId: String!, $diseaseQuery: String!) {
      search(queryString: $diseaseQuery, entityNames: ["disease"], page: {size: 3, index: 0}) {
        hits {
          id
          name
          entity
          score
        }
      }
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        approvedName
        biotype
        functionDescriptions
      }
    }
    """

    # First, search for the target gene to get its Ensembl ID
    target_search_query = """
    query searchTarget($queryString: String!) {
      search(queryString: $queryString, entityNames: ["target"], page: {size: 1, index: 0}) {
        hits {
          id
          name
          entity
          score
        }
      }
    }
    """

    try:
        # Step 1: Find target Ensembl ID
        resp = await _rate_limited_post(
            OPENTARGETS_API,
            json={
                "query": target_search_query,
                "variables": {"queryString": target_gene},
            },
        )
        resp.raise_for_status()
        search_data = resp.json()

        hits = search_data.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return f"No Open Targets results found for target: {target_gene}"

        ensembl_id = hits[0]["id"]
        target_name = hits[0]["name"]

        # Step 2: Get target info and disease associations
        target_info_query = """
        query targetInfo($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            biotype
            functionDescriptions
          }
        }
        """

        resp2 = await _rate_limited_post(
            OPENTARGETS_API,
            json={
                "query": target_info_query,
                "variables": {"ensemblId": ensembl_id},
            },
        )
        resp2.raise_for_status()
        target_data = resp2.json().get("data", {}).get("target", {})

        if not target_data:
            return f"Open Targets: Found ID {ensembl_id} for {target_gene} but no target details available."

        lines = [
            f"Open Targets data for {target_gene} ({disease}):",
            f"  Ensembl ID: {target_data.get('id', 'N/A')}",
            f"  Symbol: {target_data.get('approvedSymbol', 'N/A')}",
            f"  Name: {target_data.get('approvedName', 'N/A')}",
            f"  Biotype: {target_data.get('biotype', 'N/A')}",
        ]

        func_desc = target_data.get("functionDescriptions", [])
        if func_desc:
            lines.append(f"  Function: {func_desc[0][:300]}")

        return "\n".join(lines)
    except Exception as e:
        return f"Open Targets query failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Tool 4: UniProt (protein info for CSO)
# ---------------------------------------------------------------------------

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"


async def search_uniprot(query: str, max_results: int = 3) -> str:
    """Search UniProt for protein information."""
    try:
        response = await _rate_limited_get(
            UNIPROT_API,
            params={
                "query": query,
                "format": "json",
                "size": max_results,
                "fields": "accession,protein_name,gene_names,organism_name,cc_function,cc_subcellular_location",
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
            protein = entry.get("proteinDescription", {})
            rec_name = protein.get("recommendedName", {}).get("fullName", {}).get("value", "Unknown")
            genes = [g.get("geneName", {}).get("value", "") for g in entry.get("genes", [])]
            gene_str = ", ".join(genes) if genes else "N/A"
            organism = entry.get("organism", {}).get("scientificName", "N/A")

            func_comments = [c for c in entry.get("comments", []) if c.get("commentType") == "FUNCTION"]
            func_text = ""
            if func_comments:
                texts = func_comments[0].get("texts", [])
                if texts:
                    func_text = texts[0].get("value", "")[:300]

            lines.append(f"- {acc}: {rec_name} | Gene: {gene_str} | Organism: {organism}")
            if func_text:
                lines.append(f"  Function: {func_text}")

        return "\n".join(lines)
    except Exception as e:
        return f"UniProt search failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Tool 5: STRING-DB (protein-protein interactions for CSO)
# ---------------------------------------------------------------------------

STRING_API = "https://string-db.org/api/json/network"


async def search_string_db(protein: str, species: int = 9606) -> str:
    """Query STRING-DB for protein-protein interaction network."""
    try:
        response = await _rate_limited_get(
            STRING_API,
            params={
                "identifiers": protein,
                "species": species,
                "limit": 10,
                "caller_identity": "APEX-BiotechAgent",
            },
            timeout=20.0,
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            return f"No STRING-DB interactions found for: {protein}"

        lines = [f"STRING-DB protein interactions for '{protein}' ({len(data)} interactions):"]
        seen = set()
        for interaction in data[:10]:
            p1 = interaction.get("preferredName_A", interaction.get("stringId_A", "?"))
            p2 = interaction.get("preferredName_B", interaction.get("stringId_B", "?"))
            score = interaction.get("score", 0)
            pair = tuple(sorted([p1, p2]))
            if pair not in seen:
                seen.add(pair)
                lines.append(f"- {p1} <-> {p2} | Combined score: {score:.3f}")

        return "\n".join(lines)
    except Exception as e:
        return f"STRING-DB query failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Tool 6: ChEMBL (drug/compound info for CTO)
# ---------------------------------------------------------------------------

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


async def search_chembl(target_name: str, max_results: int = 5) -> str:
    """Search ChEMBL for drug compounds targeting a specific protein/gene."""
    try:
        response = await _rate_limited_get(
            f"{CHEMBL_API}/target/search.json",
            params={"q": target_name, "limit": 3},
        )
        response.raise_for_status()
        data = response.json()

        targets = data.get("targets", [])
        if not targets:
            return f"No ChEMBL targets found for: {target_name}"

        lines = [f"ChEMBL data for '{target_name}':"]
        for t in targets[:3]:
            chembl_id = t.get("target_chembl_id", "N/A")
            name = t.get("pref_name", "Unknown")
            target_type = t.get("target_type", "N/A")
            organism = t.get("organism", "N/A")
            lines.append(f"- {chembl_id}: {name} | Type: {target_type} | Organism: {organism}")

        # Try to get approved drugs for the first target
        if targets:
            first_id = targets[0].get("target_chembl_id")
            if first_id:
                try:
                    resp2 = await _rate_limited_get(
                        f"{CHEMBL_API}/mechanism.json",
                        params={"target_chembl_id": first_id, "limit": 5},
                    )
                    resp2.raise_for_status()
                    mechs = resp2.json().get("mechanisms", [])
                    if mechs:
                        lines.append(f"\n  Known drug mechanisms for {first_id}:")
                        for m in mechs[:5]:
                            drug = m.get("molecule_chembl_id", "N/A")
                            action = m.get("action_type", "N/A")
                            mech_name = m.get("mechanism_of_action", "N/A")
                            lines.append(f"  - {drug}: {mech_name} ({action})")
                except Exception:
                    pass

        return "\n".join(lines)
    except Exception as e:
        return f"ChEMBL search failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Tool 7: Open Targets Tractability (for CTO)
# ---------------------------------------------------------------------------


async def search_open_targets_tractability(target_gene: str) -> str:
    """Query Open Targets for target tractability assessment."""
    target_search_query = """
    query searchTarget($queryString: String!) {
      search(queryString: $queryString, entityNames: ["target"], page: {size: 1, index: 0}) {
        hits { id name }
      }
    }
    """
    tractability_query = """
    query targetTractability($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        tractability {
          label
          modality
          value
        }
      }
    }
    """
    try:
        resp = await _rate_limited_post(
            OPENTARGETS_API,
            json={"query": target_search_query, "variables": {"queryString": target_gene}},
        )
        resp.raise_for_status()
        hits = resp.json().get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return f"No Open Targets tractability data for: {target_gene}"

        ensembl_id = hits[0]["id"]

        resp2 = await _rate_limited_post(
            OPENTARGETS_API,
            json={"query": tractability_query, "variables": {"ensemblId": ensembl_id}},
        )
        resp2.raise_for_status()
        target_data = resp2.json().get("data", {}).get("target", {})
        tractability = target_data.get("tractability", [])

        if not tractability:
            return f"Open Targets: No tractability assessments for {target_gene} ({ensembl_id})"

        lines = [f"Open Targets Tractability for {target_gene} ({target_data.get('approvedSymbol', '')}):"]
        for t in tractability:
            lines.append(f"- {t.get('modality', 'N/A')}: {t.get('label', 'N/A')} = {t.get('value', 'N/A')}")

        return "\n".join(lines)
    except Exception as e:
        return f"Open Targets tractability query failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Tool 8: Open Targets Safety (for CMO)
# ---------------------------------------------------------------------------


async def search_open_targets_safety(target_gene: str) -> str:
    """Query Open Targets for target safety information."""
    target_search_query = """
    query searchTarget($queryString: String!) {
      search(queryString: $queryString, entityNames: ["target"], page: {size: 1, index: 0}) {
        hits { id name }
      }
    }
    """
    safety_query = """
    query targetSafety($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        safetyLiabilities {
          event
          effects {
            direction
            dosing
          }
          biosample {
            tissueLabel
          }
        }
      }
    }
    """
    try:
        resp = await _rate_limited_post(
            OPENTARGETS_API,
            json={"query": target_search_query, "variables": {"queryString": target_gene}},
        )
        resp.raise_for_status()
        hits = resp.json().get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return f"No Open Targets safety data for: {target_gene}"

        ensembl_id = hits[0]["id"]

        resp2 = await _rate_limited_post(
            OPENTARGETS_API,
            json={"query": safety_query, "variables": {"ensemblId": ensembl_id}},
        )
        resp2.raise_for_status()
        target_data = resp2.json().get("data", {}).get("target", {})
        safety = target_data.get("safetyLiabilities", [])

        if not safety:
            return f"Open Targets: No known safety liabilities for {target_gene} ({ensembl_id})"

        lines = [f"Open Targets Safety for {target_gene} ({target_data.get('approvedSymbol', '')}):"]
        for s in safety[:10]:
            event = s.get("event", "N/A")
            tissue = s.get("biosample", {}).get("tissueLabel", "N/A") if s.get("biosample") else "N/A"
            effects = s.get("effects", [])
            effect_str = ", ".join(
                f"{e.get('direction', '?')} ({e.get('dosing', '?')})" for e in effects
            ) if effects else "N/A"
            lines.append(f"- {event} | Tissue: {tissue} | Effects: {effect_str}")

        return "\n".join(lines)
    except Exception as e:
        return f"Open Targets safety query failed: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# Tool 9: Full-text paper retrieval via PubMed + PMC
# ---------------------------------------------------------------------------


def _pmids_to_pmc_ids(pmids: list[str]) -> dict[str, str]:
    """Use Entrez.elink to map PubMed IDs to PMC IDs.

    Returns:
        {pmid: pmc_id} for papers that have free full text in PMC.
    """
    if not pmids:
        return {}

    try:
        handle = Entrez.elink(
            dbfrom="pubmed", db="pmc", id=pmids, linkname="pubmed_pmc"
        )
        records = Entrez.read(handle)
        handle.close()
        time.sleep(PUBMED_RATE_LIMIT_DELAY)
    except Exception:
        return {}

    pmid_to_pmc: dict[str, str] = {}
    for record in records:
        pmid = str(record.get("IdList", [""])[0])
        links = record.get("LinkSetDb", [])
        if links:
            pmc_links = links[0].get("Link", [])
            if pmc_links:
                pmid_to_pmc[pmid] = str(pmc_links[0]["Id"])

    return pmid_to_pmc


def _fetch_pmc_full_text(pmc_id: str, max_chars: int = 3000) -> str:
    """Fetch full text from PMC and extract body text from XML.

    Args:
        pmc_id: Numeric PMC ID (without "PMC" prefix).
        max_chars: Maximum characters to return.

    Returns:
        Extracted body text, truncated to max_chars.
    """
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

    # Extract all text from <body> element
    body = root.find(".//body")
    if body is None:
        return ""

    # Walk all text nodes under <body>, preserving section headers
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


async def search_and_read_papers(query: str, max_papers: int = 3) -> str:
    """Search PubMed for papers, then attempt to download full text from PMC.

    Flow:
    1. Search PubMed for the query
    2. For each result, check if it has a PMC ID (free full text available)
    3. If PMC ID exists, download the full text via Entrez.efetch(db="pmc")
    4. If no PMC ID, fall back to the abstract
    5. Return the combined text (full papers + abstracts)
    """
    from agents.scout import search_pubmed

    try:
        papers = search_pubmed(query, max_results=max_papers + 2)  # fetch extras in case some lack PMC
    except Exception as e:
        return f"PubMed search failed: {str(e)[:200]}"

    if not papers:
        return f"No PubMed results found for: {query}"

    # Look up PMC IDs for all found papers
    pmids = [p["pmid"] for p in papers if p["pmid"]]
    pmid_to_pmc = await asyncio.to_thread(_pmids_to_pmc_ids, pmids)

    lines = [f"Full-text paper retrieval for '{query}':"]
    papers_with_full_text = 0

    for p in papers:
        if papers_with_full_text >= max_papers:
            break

        pmid = p["pmid"]
        pmc_id = pmid_to_pmc.get(pmid)

        if pmc_id:
            # Fetch full text from PMC
            full_text = await asyncio.to_thread(_fetch_pmc_full_text, pmc_id, 3000)
            if full_text:
                lines.append(
                    f"\n{'='*60}\n"
                    f"PMID {pmid} | PMC{pmc_id} | FULL TEXT\n"
                    f"Title: {p['title']}\n"
                    f"Journal: {p['journal']} ({p['year']})\n"
                    f"{'='*60}\n"
                    f"{full_text}"
                )
                papers_with_full_text += 1
                continue

        # Fallback to abstract
        lines.append(
            f"\n{'='*60}\n"
            f"PMID {pmid} | ABSTRACT ONLY{' (no PMC full text)' if not pmc_id else ' (PMC fetch failed)'}\n"
            f"Title: {p['title']}\n"
            f"Journal: {p['journal']} ({p['year']})\n"
            f"{'='*60}\n"
            f"{p['abstract']}"
        )
        papers_with_full_text += 1

    lines.append(f"\n[{papers_with_full_text} papers retrieved, "
                 f"{len(pmid_to_pmc)} had PMC full text available]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool dispatcher — parse TOOL_REQUESTS from LLM output and execute
# ---------------------------------------------------------------------------

# Pattern: TOOL_REQUEST: tool_name(arg1, arg2)
_ALL_TOOL_NAMES = (
    "search_pubmed|search_clinical_trials|search_open_targets|"
    "search_uniprot|search_string_db|search_chembl|"
    "search_open_targets_tractability|search_open_targets_safety|"
    "search_and_read_papers"
)
_TOOL_REQUEST_PATTERN = re.compile(
    rf"TOOL_REQUEST:\s*({_ALL_TOOL_NAMES})\(([^)]*)\)",
    re.IGNORECASE,
)

TOOL_REGISTRY: dict[str, Any] = {
    "search_pubmed": search_pubmed_tool,
    "search_clinical_trials": search_clinical_trials,
    "search_open_targets": search_open_targets,
    "search_uniprot": search_uniprot,
    "search_string_db": search_string_db,
    "search_chembl": search_chembl,
    "search_open_targets_tractability": search_open_targets_tractability,
    "search_open_targets_safety": search_open_targets_safety,
    "search_and_read_papers": search_and_read_papers,
}

# ---------------------------------------------------------------------------
# Role-specific tool assignments — Feature 5
# ---------------------------------------------------------------------------

ROLE_TOOL_REGISTRY: dict[str, list[str]] = {
    "cso": ["search_pubmed", "search_uniprot", "search_string_db", "search_and_read_papers"],
    "cto": ["search_pubmed", "search_chembl", "search_open_targets_tractability", "search_and_read_papers"],
    "cmo": ["search_clinical_trials", "search_pubmed", "search_open_targets_safety", "search_and_read_papers"],
    "cbo": ["search_pubmed", "search_open_targets", "search_clinical_trials", "search_and_read_papers"],
}


def parse_tool_requests(text: str) -> list[tuple[str, list[str]]]:
    """Parse TOOL_REQUEST lines from LLM output.

    Returns list of (tool_name, [args]).
    """
    requests = []
    for match in _TOOL_REQUEST_PATTERN.finditer(text):
        tool_name = match.group(1).lower()
        raw_args = match.group(2)
        # Parse comma-separated args, strip quotes and whitespace
        args = [a.strip().strip("'\"") for a in raw_args.split(",") if a.strip()]
        requests.append((tool_name, args))
    return requests


async def execute_tool_requests(
    requests: list[tuple[str, list[str]]], role: str | None = None
) -> str:
    """Execute parsed tool requests and return formatted results.

    If role is provided, only tools in ROLE_TOOL_REGISTRY[role] are allowed.
    """
    if not requests:
        return ""

    allowed_tools = None
    if role and role in ROLE_TOOL_REGISTRY:
        allowed_tools = set(ROLE_TOOL_REGISTRY[role])

    results = []
    for tool_name, args in requests[:3]:  # Max 3 tool calls per agent
        # Enforce role-specific tool access
        if allowed_tools and tool_name not in allowed_tools:
            results.append(f"Tool '{tool_name}' not available for {role.upper()} role")
            continue

        tool_fn = TOOL_REGISTRY.get(tool_name)
        if not tool_fn:
            results.append(f"Unknown tool: {tool_name}")
            continue

        try:
            if tool_name in ("search_pubmed", "search_clinical_trials"):
                query = args[0] if args else ""
                max_results = int(args[1]) if len(args) > 1 else 5
                result = await tool_fn(query, max_results)
            elif tool_name == "search_open_targets":
                target = args[0] if args else ""
                disease = args[1] if len(args) > 1 else ""
                result = await tool_fn(target, disease)
            elif tool_name in ("search_uniprot", "search_chembl"):
                query = args[0] if args else ""
                max_results = int(args[1]) if len(args) > 1 else 3
                result = await tool_fn(query, max_results)
            elif tool_name == "search_string_db":
                protein = args[0] if args else ""
                result = await tool_fn(protein)
            elif tool_name in ("search_open_targets_tractability", "search_open_targets_safety"):
                gene = args[0] if args else ""
                result = await tool_fn(gene)
            elif tool_name == "search_and_read_papers":
                query = args[0] if args else ""
                max_papers = int(args[1]) if len(args) > 1 else 3
                result = await tool_fn(query, max_papers)
            else:
                result = f"Tool {tool_name} not implemented"

            results.append(result)
        except Exception as e:
            results.append(f"Tool {tool_name} error: {str(e)[:200]}")

    if not results:
        return ""

    return "\n\n---\n\n".join(["AUTONOMOUS TOOL RESULTS:"] + results)

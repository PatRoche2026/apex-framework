"""IP Attorney agent — searches Lens.org patents and assesses freedom-to-operate risk."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents import (
    ANTHROPIC_API_KEY,
    LENS_API_TOKEN,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    ROLE_MODELS,
    estimate_cost,
    llm_semaphore,
)
from agents.prompts import IP_ATTORNEY_SYSTEM_PROMPT
from agents.state import APEXState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM instance — IP Attorney uses Haiku (fast model)
# ---------------------------------------------------------------------------

_IP_MODEL = ROLE_MODELS["ip_attorney"]
_llm = ChatAnthropic(
    model=_IP_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=ANTHROPIC_API_KEY,
    max_tokens=2500,
)

# ---------------------------------------------------------------------------
# Gene name lookup (NCBI)
# ---------------------------------------------------------------------------


async def _lookup_gene_full_name(symbol: str) -> str:
    """Look up gene full name from NCBI. Falls back to symbol if unavailable."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "gene",
                    "term": f"{symbol}[Gene Name] AND Homo sapiens[Organism]",
                    "retmode": "json", "retmax": 1,
                    "tool": "apex-bioresearch", "email": "patroche@mit.edu",
                },
            )
            gene_ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not gene_ids:
                return symbol

            await asyncio.sleep(0.4)

            resp2 = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={
                    "db": "gene", "id": gene_ids[0], "retmode": "json",
                    "tool": "apex-bioresearch", "email": "patroche@mit.edu",
                },
            )
            full_name = resp2.json().get("result", {}).get(gene_ids[0], {}).get("description", "")
            return full_name if full_name else symbol
    except Exception:
        return symbol


# ---------------------------------------------------------------------------
# Lens.org Patent Search API
# ---------------------------------------------------------------------------


async def search_lens_patents(target: str, full_name: str = "", max_results: int = 10) -> list[dict]:
    """Search Lens.org Patent API for patents mentioning the target gene.

    Uses a simple query string search across titles, abstracts, and claims.
    Falls back to empty list if API unavailable or token not set.
    """
    # Read token at call time (not module load) to handle Railway env injection
    import os
    token = (os.getenv("LENS_API_TOKEN", "") or LENS_API_TOKEN).strip()
    logger.info(f"Lens.org token: {'SET' if token else 'EMPTY'} (len={len(token)})")
    if not token:
        logger.warning("LENS_API_TOKEN not set — proceeding without patent data")
        return []

    # Build query — search by gene symbol and optionally full gene name.
    # Sanitize full_name: NCBI gene lookup sometimes returns messy strings
    # with parentheses, slashes, or mouse-ortholog notation (e.g. "Col2a1(+/d...")
    # which break the Lens.org query parser. Strip to clean alphanumeric+space only.
    query_parts = [target]
    if full_name and full_name.lower() != target.lower():
        clean_name = re.sub(r"[^\w\s-]", " ", full_name).strip()
        clean_name = re.sub(r"\s+", " ", clean_name)
        # Reject if too short, too long, or contains the target already
        if (
            clean_name
            and len(clean_name) >= 4
            and len(clean_name) <= 80
            and target.lower() not in clean_name.lower().split()[:1]
        ):
            query_parts.append(f'"{clean_name}"')
    query_str = " OR ".join(query_parts)
    logger.info(f"Lens.org query for {target}: {query_str}")

    payload = {
        "query": query_str,
        "include": [
            "lens_id", "doc_number", "jurisdiction",
            "biblio.invention_title", "date_published", "abstract",
            "biblio.parties.applicants", "biblio.parties.inventors",
            "biblio.classifications_cpc", "legal_status",
        ],
        "size": max_results,
        "sort": [{"date_published": "desc"}],
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.lens.org/patent/search",
                json=payload,
                headers=headers,
            )

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", [])
                total = data.get("total", 0)
                logger.info(f"Lens.org: {len(results)} patents returned for '{target}' (total: {total})")
            elif resp.status_code == 403:
                logger.warning(f"Lens.org 403 Forbidden — token may lack patent search scope")
                return []
            else:
                logger.warning(f"Lens.org returned {resp.status_code}: {resp.text[:300]}")
                return []
    except httpx.TimeoutException:
        logger.warning("Lens.org API timeout")
        return []
    except Exception as e:
        logger.warning(f"Lens.org API error: {e}")
        return []

    patents = []
    for p in results:
        # Title
        title_obj = p.get("biblio", {}).get("invention_title", [])
        title = ""
        if isinstance(title_obj, list) and title_obj:
            first = title_obj[0]
            title = first.get("text", "") if isinstance(first, dict) else str(first)
        elif isinstance(title_obj, str):
            title = title_obj

        # Assignee / applicant
        applicants = p.get("biblio", {}).get("parties", {}).get("applicants", [])
        assignee = "Unknown"
        if applicants and isinstance(applicants[0], dict):
            assignee = (
                applicants[0].get("extracted_name", {}).get("value", "")
                or applicants[0].get("name", "")
                or "Unknown"
            )

        # Inventors
        inventors_raw = p.get("biblio", {}).get("parties", {}).get("inventors", [])
        inventors = []
        for inv in (inventors_raw or [])[:3]:
            if isinstance(inv, dict):
                name = inv.get("extracted_name", {}).get("value", "") or inv.get("name", "")
                if name:
                    inventors.append(name)

        # CPC codes — Lens.org nests them under classifications_cpc.classifications
        cpcs_raw = p.get("biblio", {}).get("classifications_cpc", {})
        if isinstance(cpcs_raw, dict):
            cpcs_list = cpcs_raw.get("classifications", [])
        elif isinstance(cpcs_raw, list):
            cpcs_list = cpcs_raw
        else:
            cpcs_list = []
        cpc_codes = []
        for c in cpcs_list[:5]:
            if isinstance(c, dict) and c.get("symbol"):
                cpc_codes.append(c["symbol"])

        # Legal status
        legal = p.get("legal_status", {})
        patent_status = ""
        if isinstance(legal, dict):
            patent_status = legal.get("patent_status", "")
        elif isinstance(legal, str):
            patent_status = legal

        # Citation count (field not requested; default 0)
        citation_count = 0

        # Abstract
        abstract_obj = p.get("abstract", [])
        abstract_text = ""
        if isinstance(abstract_obj, list) and abstract_obj:
            first_abs = abstract_obj[0]
            abstract_text = first_abs.get("text", "") if isinstance(first_abs, dict) else str(first_abs)
        elif isinstance(abstract_obj, str):
            abstract_text = abstract_obj

        patents.append({
            "lens_id": p.get("lens_id", ""),
            "doc_number": p.get("doc_number", ""),
            "jurisdiction": p.get("jurisdiction", ""),
            "title": (title or "")[:200],
            "date_published": p.get("date_published", ""),
            "abstract": (abstract_text or "")[:500],
            "assignee": assignee,
            "inventors": inventors,
            "cpc_codes": cpc_codes,
            "legal_status": patent_status,
            "citation_count": citation_count,
        })

    return patents


# ---------------------------------------------------------------------------
# Format patents for LLM context
# ---------------------------------------------------------------------------


def _format_patents_for_llm(patents: list[dict], target: str) -> str:
    """Format Lens.org patent search results into structured text for the LLM."""
    if not patents:
        return (
            f"PATENT SEARCH RESULTS FOR: {target}\n\n"
            f"No patents found mentioning '{target}' in patent titles, abstracts, or claims.\n"
            f"Note: This could indicate an open IP landscape (positive signal) or that "
            f"patents use different terminology for this target.\n"
            f"Assessment should be based on general domain knowledge."
        )

    lines = [
        f"PATENT SEARCH RESULTS FOR: {target}",
        f"Source: Lens.org Patent Database (comprehensive global patent data)",
        f"Total patents found: {len(patents)}",
        "",
    ]

    for i, p in enumerate(patents, 1):
        lines.append(f"--- Patent {i} ---")
        doc_num = p.get("doc_number", "")
        jurisdiction = p.get("jurisdiction", "")
        if doc_num:
            lines.append(f"Document Number: {jurisdiction}{doc_num}")
        lines.append(f"Title: {p['title']}")
        lines.append(f"Date Published: {p.get('date_published', 'Unknown')}")
        lines.append(f"Assignee/Applicant: {p['assignee']}")
        if p.get("inventors"):
            lines.append(f"Inventors: {', '.join(p['inventors'])}")
        if p.get("legal_status"):
            lines.append(f"Legal Status: {p['legal_status']}")
        if p.get("cpc_codes"):
            lines.append(f"CPC Classifications: {', '.join(p['cpc_codes'])}")
        if p.get("citation_count"):
            lines.append(f"Cited by: {p['citation_count']} other patents")
        if p.get("abstract"):
            lines.append(f"Abstract: {p['abstract']}")
        lines.append(f"Lens ID: {p.get('lens_id', '')}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IP Attorney node
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embedding for better retrieval
# ---------------------------------------------------------------------------

CRAG_QUALITY_THRESHOLD = 0.7


async def _generate_hyde_document(target: str, query: str) -> str:
    """Generate a hypothetical IP assessment to improve retrieval quality.

    The hypothetical doc's embedding matches chunks that semantically look like
    good IP assessments, not just keyword matches on the target name.
    """
    hyde_prompt = (
        f"Write a concise patent landscape analysis for the gene target {target} "
        f"as a potential therapeutic target for {query}. Cover: existing patent "
        f"holders, key patent families, freedom-to-operate considerations, "
        f"relevant patent eligibility issues under 35 USC 101 (Mayo/Myriad/Alice), "
        f"and patent cliff timing. Be specific about patent types "
        f"(composition-of-matter vs method-of-treatment vs diagnostic)."
    )
    try:
        response = await asyncio.wait_for(
            _llm.ainvoke([HumanMessage(content=hyde_prompt)]),
            timeout=30,
        )
        return response.content
    except Exception:
        # Fallback: use the raw query if HyDE fails
        return f"Patent landscape for {target} in {query}"


async def _retrieve_ip_context(hyde_doc: str, k: int = 6) -> tuple[str, float]:
    """Retrieve IP law context from the ip_attorney ChromaDB collection.

    Uses the HyDE document for semantic matching against the IP law corpus.
    Returns (formatted_context, avg_distance). Lower distance = more relevant.
    """
    try:
        from rag.store import query_collection
        results = query_collection("ip_attorney", hyde_doc, k=k)
        if not results:
            return "", 1.0
        distances = [r.get("distance", 1.0) for r in results]
        avg_distance = sum(distances) / len(distances) if distances else 1.0
        parts = []
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "Unknown")
            dist = r.get("distance", 1.0)
            parts.append(f"[Source {i}: {src}, relevance: {1-dist:.2f}]\n{r['text']}")
        return "\n\n".join(parts), avg_distance
    except Exception:
        return "", 1.0


async def _crag_quality_check(
    target: str, ip_context: str, avg_distance: float
) -> str:
    """CRAG: if retrieved IP context is low quality, escalate to PubMed."""
    if avg_distance > CRAG_QUALITY_THRESHOLD or not ip_context.strip():
        try:
            from agents.tools import search_pubmed_tool
            pubmed_results = await search_pubmed_tool(
                f"{target} patent landscape intellectual property", max_results=3
            )
            if pubmed_results and "No results" not in pubmed_results:
                escalation_note = (
                    f"\n\n[CRAG ESCALATION: Knowledge base relevance low "
                    f"(avg distance: {avg_distance:.2f}). Supplementing with "
                    f"PubMed patent landscape publications:]\n\n{pubmed_results}"
                )
                return ip_context + escalation_note
        except Exception:
            pass
    return ip_context


def _verify_cited_patents(assessment_text: str, patents: list[dict]) -> dict:
    """Verify that patent numbers cited in the assessment match real search results.

    Returns dict with total_cited, verified, fabricated counts.
    """
    cited = set(re.findall(
        r"(?:US|EP|WO|CN|JP)\s?\d{6,12}[A-Z]?\d?",
        assessment_text,
    ))
    real_numbers: set[str] = set()
    for p in patents:
        doc_num = (p.get("doc_number", "") or "").strip()
        jurisdiction = (p.get("jurisdiction", "") or "").strip()
        if doc_num:
            real_numbers.add(f"{jurisdiction}{doc_num}")
            real_numbers.add(doc_num)
    verified = cited & real_numbers if real_numbers else set()
    fabricated = cited - real_numbers if real_numbers else set()
    return {
        "total_cited": len(cited),
        "verified": len(verified),
        "fabricated": len(fabricated),
        "fabricated_numbers": sorted(fabricated)[:5],
    }


# ---------------------------------------------------------------------------
# IP Attorney node — full RAG stack
# ---------------------------------------------------------------------------


async def ip_attorney_node(state: APEXState) -> dict[str, Any]:
    """IP Attorney: Lens.org patents + HyDE + CRAG + Self-RAG + citation verification."""
    ts = datetime.now(timezone.utc).isoformat()

    query = state["query"]
    target = query.strip().split()[0].upper() if query.strip() else "UNKNOWN"

    # 1. Search Lens.org for real patent data (symbol-only; thousands of hits)
    patents = []
    try:
        patents = await search_lens_patents(target, full_name="")
    except Exception:
        pass

    patent_context = _format_patents_for_llm(patents, target)

    # 2. HyDE — generate hypothetical IP assessment to improve retrieval
    hyde_doc = await _generate_hyde_document(target, query)

    # 3. Retrieve from ip_attorney ChromaDB collection using HyDE embedding
    ip_law_context, avg_distance = await _retrieve_ip_context(hyde_doc, k=6)

    # 4. CRAG quality gate — escalate to PubMed if retrieval was weak
    ip_law_context = await _crag_quality_check(target, ip_law_context, avg_distance)

    # 5. Also retrieve from shared collection (cross-role context)
    shared_context = ""
    try:
        from rag.retriever import retrieve_context
        shared_context = retrieve_context(
            "shared", f"{target} patent IP freedom to operate", k=2
        )
    except Exception:
        pass

    # 6. Assemble user prompt with all context
    user_prompt = f"TARGET QUERY: {query}\n\n{patent_context}\n\n"
    if ip_law_context:
        user_prompt += f"RELEVANT IP LAW AND PRECEDENT FROM KNOWLEDGE BASE:\n{ip_law_context}\n\n"
    if shared_context:
        user_prompt += f"ADDITIONAL SHARED CONTEXT:\n{shared_context}\n\n"
    user_prompt += (
        "Based on ALL the above — the patent search results, the IP law context, "
        "and any additional information — produce your complete IP assessment. "
        "Follow the output format exactly. Tag every factual claim with a "
        "reflection token [SUPPORTED], [UNSUPPORTED], or [UNCERTAIN]. "
        "NEVER fabricate patent numbers — only cite patents from the search results provided."
    )

    messages = [
        SystemMessage(content=IP_ATTORNEY_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    call_cost = 0.0
    async with llm_semaphore:
        try:
            response = await asyncio.wait_for(
                _llm.ainvoke(messages),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            assessment_text = response.content
            usage = getattr(response, "usage_metadata", None) or {}
            call_cost = estimate_cost(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                model=_IP_MODEL,
            )
        except asyncio.TimeoutError:
            assessment_text = (
                "[IP Attorney timed out — defaulting to MEDIUM FTO risk]\n\n"
                "IP_SCORE: 5\nFTO_RISK: MEDIUM"
            )
        except Exception as e:
            err_str = str(e).lower()
            if "credit balance" in err_str or "billing" in err_str:
                assessment_text = "[ANTHROPIC_CREDIT_ERROR] API credits need renewal"
            elif "rate limit" in err_str:
                assessment_text = "[ANTHROPIC_RATE_LIMIT] Too many requests"
            else:
                assessment_text = (
                    f"[IP Attorney error: {str(e)[:200]}]\n\n"
                    "IP_SCORE: 5\nFTO_RISK: MEDIUM"
                )

    # 7. Parse scores and verify citations
    ip_score_match = re.search(r"IP_SCORE:\s*(\d+)", assessment_text)
    ip_score = min(int(ip_score_match.group(1)), 10) if ip_score_match else 5

    fto_match = re.search(r"FTO_RISK:\s*(HIGH|MEDIUM|LOW)", assessment_text, re.IGNORECASE)
    fto_risk = fto_match.group(1).upper() if fto_match else "MEDIUM"

    # 8. Patent citation verification (no hallucinated patent numbers)
    citation_check = _verify_cited_patents(assessment_text, patents)

    return {
        "ip_assessment": assessment_text,
        "executive_scores": {"ip_attorney": {"ip_landscape": ip_score}},
        "activity_log": [
            {
                "node": "ip_assess",
                "role": "IP Strategy Advisor",
                "status": "complete",
                "ip_score": ip_score,
                "fto_risk": fto_risk,
                "patents_found": len(patents),
                "kb_relevance": round(1 - avg_distance, 2),
                "citations_cited": citation_check["total_cited"],
                "citations_verified": citation_check["verified"],
                "citations_fabricated": citation_check["fabricated"],
                "cost_usd": call_cost,
                "timestamp": ts,
            }
        ],
    }

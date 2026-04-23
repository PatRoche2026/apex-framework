"""APEX Backend — FastAPI + WebSocket streaming + REST endpoints."""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from agents.graph import build_graph, compiled_planning_graph, make_initial_state
from agents.prompts import AGENT_PERSONAS, EXECUTIVE_PROMPTS, EXECUTIVE_ROLES, DIRECTOR_SYSTEM_PROMPT
from agents.state import APEXState
from generate_ddp import generate_ddp_pdf

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="APEX — Agentic Pipeline for Executive Decisions",
    description=(
        "Multi-agent biotech executive debate system. "
        "5 AI agents evaluate drug targets through parallel assessment, "
        "structured debate, and Portfolio Director synthesis."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session storage
# ---------------------------------------------------------------------------

sessions: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="Drug target evaluation query (e.g. 'OSMR ulcerative colitis')",
    )


class FeedbackRequest(BaseModel):
    feedback: str = Field(
        ...,
        min_length=3,
        description="CEO feedback to inject into the next evaluation round",
    )
    action: Optional[str] = Field(
        default=None,
        description="Optional action: 'approve' triggers DDP planning, 'reject' marks session rejected",
    )


class SessionSummary(BaseModel):
    session_id: str
    query: str
    status: str
    confidence_score: int
    verdict: str
    debate_rounds: int
    timestamp: str


# ---------------------------------------------------------------------------
# Node name mapping for WebSocket events
# ---------------------------------------------------------------------------

_NODE_AGENT_MAP = {
    "scout": "scout",
    "cso_assess": "cso",
    "cto_assess": "cto",
    "cmo_assess": "cmo",
    "cbo_assess": "cbo",
    "ip_assess": "ip_attorney",
    "debate_router": "system",
    "cso_rebuttal": "cso",
    "cto_rebuttal": "cto",
    "cmo_rebuttal": "cmo",
    "cbo_rebuttal": "cbo",
    "portfolio_director": "portfolio_director",
}

_NODE_TYPE_MAP = {
    "scout": "scout",
    "cso_assess": "assessment",
    "cto_assess": "assessment",
    "cmo_assess": "assessment",
    "cbo_assess": "assessment",
    "ip_assess": "assessment",
    "debate_router": "sync",
    "cso_rebuttal": "rebuttal",
    "cto_rebuttal": "rebuttal",
    "cmo_rebuttal": "rebuttal",
    "cbo_rebuttal": "rebuttal",
    "portfolio_director": "verdict",
}


def _parse_verdict_short(text: str) -> str:
    """Extract short verdict from portfolio verdict text.

    Delegates to the canonical parse_verdict for consistency.
    """
    from agents.executives import parse_verdict
    return parse_verdict(text)


def _sum_costs(result: dict) -> float:
    """Sum all cost_usd values from activity_log entries."""
    total = 0.0
    for entry in result.get("activity_log", []):
        total += entry.get("cost_usd", 0.0)
    return round(total, 4)


# ---------------------------------------------------------------------------
# PMID auto-verification (NCBI E-utilities esummary)
# ---------------------------------------------------------------------------

_PMID_PATTERN = re.compile(r"PMID[\s:]?\s*(\d{6,9})")


def _extract_all_pmids(result: dict) -> list[str]:
    """Scan all agent outputs for PMID references and return a deduped, sorted list."""
    text_fields = [
        "scout_data",
        "cso_assessment", "cto_assessment", "cmo_assessment", "cbo_assessment",
        "cso_rebuttal", "cto_rebuttal", "cmo_rebuttal", "cbo_rebuttal",
        "portfolio_verdict",
    ]
    found = set()
    for f in text_fields:
        text = result.get(f) or ""
        if isinstance(text, str):
            for m in _PMID_PATTERN.finditer(text):
                found.add(m.group(1))
    return sorted(found, key=int)


async def verify_pmids(result: dict, max_pmids: int = 20) -> dict:
    """Verify PMIDs against NCBI E-utilities esummary.

    Returns dict mapping each PMID to a verification record:
        {
          "valid": bool,
          "title": str (truncated to 200 chars),
          "year": str,
          "journal": str,
          "authors": str,
        }

    Caps at `max_pmids` to avoid NCBI rate limit (3 req/s without API key).
    Uses 100ms delay between requests for safety. Total worst case ~2 seconds.
    """
    pmids = _extract_all_pmids(result)
    if not pmids:
        return {}

    pmids = pmids[:max_pmids]
    verification: dict[str, dict] = {}

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    async with httpx.AsyncClient(timeout=8.0) as client:
        for pmid in pmids:
            try:
                resp = await client.get(
                    base_url,
                    params={
                        "db": "pubmed",
                        "id": pmid,
                        "retmode": "json",
                        "tool": "apex-bioresearch",
                        "email": "patroche@mit.edu",
                    },
                )
                if resp.status_code != 200:
                    verification[pmid] = {"valid": False, "error": f"HTTP {resp.status_code}"}
                    continue
                data = resp.json()
                result_obj = data.get("result", {})
                # esummary returns the article keyed by uid string
                article_keys = [k for k in result_obj.keys() if k != "uids"]
                if not article_keys:
                    verification[pmid] = {"valid": False, "error": "no result"}
                    continue
                article = result_obj[article_keys[0]]
                # esummary returns 'error' field for invalid PMIDs
                if article.get("error"):
                    verification[pmid] = {"valid": False, "error": article["error"]}
                    continue
                authors_list = article.get("authors", []) or []
                first_authors = ", ".join(
                    a.get("name", "") for a in authors_list[:3] if isinstance(a, dict)
                )
                if len(authors_list) > 3:
                    first_authors += " et al."
                verification[pmid] = {
                    "valid": True,
                    "title": (article.get("title", "") or "")[:200],
                    "year": (article.get("pubdate", "") or "").split()[0]
                            if article.get("pubdate") else "",
                    "journal": article.get("source", article.get("fulljournalname", ""))[:120],
                    "authors": first_authors,
                }
            except Exception as e:
                verification[pmid] = {"valid": False, "error": str(e)[:100]}

            # Rate limit safety: NCBI allows 3 req/s without API key
            await asyncio.sleep(0.4)

    return verification


def _extract_event_data(node_name: str, node_data: dict) -> dict:
    """Extract the relevant content field from a node's output for the frontend."""
    if node_data is None:
        return {}

    # Map node to its primary content field
    field_map = {
        "scout": ("scout_data", "scout_sources"),
        "cso_assess": ("cso_assessment",),
        "cto_assess": ("cto_assessment",),
        "cmo_assess": ("cmo_assessment",),
        "cbo_assess": ("cbo_assessment",),
        "ip_assess": ("ip_assessment",),
        "cso_rebuttal": ("cso_rebuttal",),
        "cto_rebuttal": ("cto_rebuttal",),
        "cmo_rebuttal": ("cmo_rebuttal",),
        "cbo_rebuttal": ("cbo_rebuttal",),
        "portfolio_director": ("portfolio_verdict", "confidence_score", "executive_scores", "debate_round"),
    }

    fields = field_map.get(node_name, ())
    data = {}
    for f in fields:
        if f in node_data:
            data[f] = node_data[f]

    # Include parsed scores if present
    if "executive_scores" in node_data and node_name != "portfolio_director":
        data["scores"] = node_data["executive_scores"]

    return data


# ---------------------------------------------------------------------------
# Planning pipeline helpers
# ---------------------------------------------------------------------------

_PLAN_NODE_AGENT_MAP = {
    "cso_plan": "cso",
    "cto_plan": "cto",
    "cmo_plan": "cmo",
    "cbo_plan": "cbo",
    "ip_attorney_plan": "ip_attorney",
    "director_synthesis": "portfolio_director",
}


def _make_planning_state(session: dict) -> APEXState:
    """Build an APEXState for the planning pipeline from a completed evaluation session."""
    result = session["result"]
    state = make_initial_state(session["query"])
    for field in [
        "scout_data", "scout_sources",
        "cso_assessment", "cto_assessment", "cmo_assessment", "cbo_assessment",
        "cso_rebuttal", "cto_rebuttal", "cmo_rebuttal", "cbo_rebuttal",
        "portfolio_verdict", "confidence_score", "executive_scores",
        "ceo_feedback", "ceo_feedback_history", "evaluation_round", "debate_round",
        "gene", "indication", "activity_log",
    ]:
        if field in result:
            state[field] = result[field]
    state["planning_triggered"] = True
    state["ddp_status"] = "in_progress"
    return state


def _extract_gene_indication(session: dict) -> tuple[str, str]:
    """Extract gene and indication from session state, falling back to query split."""
    result = session["result"]
    gene = result.get("gene", "").strip()
    indication = result.get("indication", "").strip()
    if not gene or not indication:
        tokens = session["query"].strip().split()
        gene = gene or (tokens[0].upper() if tokens else session["query"])
        indication = indication or (" ".join(tokens[1:]) if len(tokens) > 1 else session["query"])
    return gene, indication


def _extract_evidence_needed(verdict_text: str) -> str:
    """Extract the conditions/evidence needed to change the verdict from Director text."""
    if not verdict_text:
        return "Additional clinical and preclinical evidence required."
    # Scan for common conditional language
    patterns = [
        r"(?:would require|requires?|need[s]? to see|conditional on|pending|subject to)[^\.\n]{10,200}",
        r"(?:before (?:a )?go|prior to investment|key requirements?)[^\.\n]{10,200}",
        r"(?:evidence needed|data gaps?|open questions?)[^\.\n]{10,200}",
    ]
    found = []
    for pat in patterns:
        matches = re.findall(pat, verdict_text, flags=re.IGNORECASE)
        found.extend(m.strip() for m in matches[:2])
    if found:
        return " ".join(found[:3])
    # Fallback: return last paragraph (often contains conditions)
    paragraphs = [p.strip() for p in verdict_text.split("\n\n") if p.strip()]
    return paragraphs[-1][:400] if paragraphs else "See full Director verdict for details."


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    """App info and available endpoints."""
    return {
        "app": "APEX",
        "tagline": "Agentic Pipeline for Executive Decisions",
        "description": (
            "Multi-agent biotech executive debate system. "
            "5 AI agents (CSO, CTO, CMO, CBO, Portfolio Director) evaluate "
            "drug targets through parallel assessment, structured debate, "
            "and investment committee synthesis."
        ),
        "version": "1.0.0",
        "endpoints": {
            "GET /": "App info",
            "GET /personas": "Agent persona metadata",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "POST /evaluate": "Run full evaluation (synchronous)",
            "POST /feedback/{session_id}": "Submit CEO feedback for re-evaluation",
            "WS /ws/evaluate": "WebSocket streaming evaluation",
            "WS /ws/feedback/{session_id}": "WebSocket streaming re-evaluation with CEO feedback",
            "WS /ws/plan/{session_id}": "WebSocket streaming DDP planning pipeline",
            "POST /reject/{session_id}": "CEO rejects verdict — marks session rejected",
            "GET /download/ddp/{session_id}": "Download generated DDP PDF",
            "GET /results/{session_id}": "Fetch completed evaluation results",
            "GET /sessions": "List all completed sessions",
            "GET /export/{session_id}": "Download Markdown report",
        },
        "example_queries": [
            "OSMR ulcerative colitis",
            "GLP-1 receptor agonists Alzheimer's disease",
            "CD47 cancer immunotherapy",
        ],
        "architecture": {
            "agents": ["Research Scout", "CSO", "CTO", "CMO", "CBO", "Portfolio Director"],
            "flow": "Scout -> 4 parallel assessments -> Debate rebuttals -> Portfolio Director verdict",
            "conditional_loop": "If confidence < 60%, agents debate again (max 2 rounds)",
            "ceo_feedback": "Human-in-the-loop: submit CEO feedback to trigger re-evaluation with directives",
            "domain_tools": {
                "CSO": ["PubMed", "UniProt", "STRING-DB"],
                "CTO": ["PubMed", "ChEMBL", "Open Targets Tractability"],
                "CMO": ["ClinicalTrials.gov", "PubMed", "Open Targets Safety"],
                "CBO": ["PubMed", "Open Targets", "ClinicalTrials.gov"],
            },
        },
        "author": "Patrick Roche, MIT Media Lab MAS.664",
    }


@app.get("/personas")
def personas():
    """Return agent persona metadata for frontend rendering."""
    return {"personas": AGENT_PERSONAS}


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sessions_completed": len(sessions),
    }


@app.get('/targets')
async def get_targets():
    """Return the demo target catalog from config.DEMO_TARGETS.

    Replace the DEMO_TARGETS list in config.py to expose your own targets.
    """
    from config import DEMO_TARGETS
    return {
        'targets': DEMO_TARGETS,
        'count': len(DEMO_TARGETS),
    }

@app.get('/metrics')
async def get_metrics():
    """Return framework-level metrics (agent count, RAG collections, debate config).

    This endpoint deliberately excludes any model-performance metrics — those
    are domain-specific and should be surfaced via an evaluation results API.
    """
    from config import (
        SCORING_DIMENSIONS,
        MAX_DEBATE_ROUNDS,
        CONFIDENCE_THRESHOLD,
        MAX_CONCURRENT_LLM_CALLS,
        EMBEDDING_MODEL,
    )
    try:
        from rag.store import get_collection, AGENT_ROLES
        rag_counts = {role: get_collection(role).count() for role in AGENT_ROLES}
        rag_counts['total'] = sum(rag_counts.values())
    except Exception:
        rag_counts = {'total': 0, 'note': 'ChromaDB not yet initialized; run python -m rag.ingest'}

    return {
        'framework': 'APEX',
        'version': '1.0.0',
        'agent_count': 7,  # scout + 4 advisors + IP advisor + director
        'scoring_dimensions': {dim: meta['weight'] for dim, meta in SCORING_DIMENSIONS.items()},
        'debate': {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'max_rounds': MAX_DEBATE_ROUNDS,
            'max_concurrent_calls': MAX_CONCURRENT_LLM_CALLS,
        },
        'rag': {
            'embedding_model': EMBEDDING_MODEL,
            'collection_counts': rag_counts,
        },
        'supported_tools': ['pubmed', 'clinicaltrials.gov', 'open_targets', 'lens.org'],
    }


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    """Run a full synchronous evaluation. Returns the complete result."""
    session_id = str(uuid.uuid4())
    graph = build_graph()
    state = make_initial_state(req.query)

    result = await graph.ainvoke(state)

    # Store session
    sessions[session_id] = {
        "session_id": session_id,
        "query": req.query,
        "status": "complete",
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "session_id": session_id,
        "query": req.query,
        "confidence_score": result.get("confidence_score", 0),
        "debate_rounds": result.get("debate_round", 0),
        "verdict": result.get("portfolio_verdict", ""),
        "executive_scores": result.get("executive_scores", {}),
        "scout_sources": result.get("scout_sources", []),
        "assessments": {
            "cso": result.get("cso_assessment", ""),
            "cto": result.get("cto_assessment", ""),
            "cmo": result.get("cmo_assessment", ""),
            "cbo": result.get("cbo_assessment", ""),
        },
        "rebuttals": {
            "cso": result.get("cso_rebuttal", ""),
            "cto": result.get("cto_rebuttal", ""),
            "cmo": result.get("cmo_rebuttal", ""),
            "cbo": result.get("cbo_rebuttal", ""),
        },
        "activity_log": result.get("activity_log", []),
        "estimated_cost_usd": _sum_costs(result),
    }


@app.get("/results/{session_id}")
def get_results(session_id: str):
    """Fetch completed evaluation results by session ID."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session = sessions[session_id]
    result = session["result"]

    return {
        "session_id": session_id,
        "query": session["query"],
        "status": session["status"],
        "confidence_score": result.get("confidence_score", 0),
        "debate_rounds": result.get("debate_round", 0),
        "verdict": result.get("portfolio_verdict", ""),
        "executive_scores": result.get("executive_scores", {}),
        "scout_sources": result.get("scout_sources", []),
        "assessments": {
            "cso": result.get("cso_assessment", ""),
            "cto": result.get("cto_assessment", ""),
            "cmo": result.get("cmo_assessment", ""),
            "cbo": result.get("cbo_assessment", ""),
        },
        "rebuttals": {
            "cso": result.get("cso_rebuttal", ""),
            "cto": result.get("cto_rebuttal", ""),
            "cmo": result.get("cmo_rebuttal", ""),
            "cbo": result.get("cbo_rebuttal", ""),
        },
        "activity_log": result.get("activity_log", []),
        "estimated_cost_usd": _sum_costs(result),
        "pmid_verification": session.get("pmid_verification", {}),
        "pmid_summary": session.get("pmid_summary", {"verified": 0, "total": 0}),
        "timestamp": session["timestamp"],
    }


@app.get("/sessions")
def list_sessions():
    """List all completed evaluation sessions."""
    summaries = []
    for sid, session in sessions.items():
        result = session["result"]
        verdict_short = _parse_verdict_short(result.get("portfolio_verdict", ""))

        summaries.append({
            "session_id": sid,
            "query": session["query"],
            "status": session["status"],
            "confidence_score": result.get("confidence_score", 0),
            "verdict": verdict_short,
            "debate_rounds": result.get("debate_round", 0),
            "timestamp": session["timestamp"],
        })
    return {"sessions": summaries, "count": len(summaries)}


@app.get("/export/{session_id}")
def export_session(session_id: str):
    """Export a completed evaluation as a downloadable Markdown report."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session = sessions[session_id]
    result = session["result"]

    report = _generate_markdown_report(session["query"], result, session_id, session)
    return Response(
        content=report,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=apex-{session_id[:8]}.md"},
    )


def _generate_markdown_report(query: str, result: dict, session_id: str, session: dict | None = None) -> str:
    """Generate a formatted Markdown report of the full debate."""
    lines = [
        f"# APEX Evaluation Report",
        f"**Query:** {query}",
        f"**Session:** {session_id[:8]}",
        f"**Confidence:** {result.get('confidence_score', 'N/A')}%",
        f"**Debate Rounds:** {result.get('debate_round', 'N/A')}",
        "",
        "---",
        "",
        "## Research Scout Summary",
        "",
        f"**Sources:** {len(result.get('scout_sources', []))} papers analyzed",
        "",
    ]

    for src in result.get("scout_sources", []):
        lines.append(f"- **PMID {src['pmid']}**: {src['title']} — {src['journal']} ({src['year']})")
    lines.append("")

    # Assessments
    for role, label in [("cso", "CSO"), ("cto", "CTO"), ("cmo", "CMO"), ("cbo", "CBO")]:
        lines.append(f"---\n\n## {label} Assessment\n")
        lines.append(result.get(f"{role}_assessment", "*No assessment available*"))
        lines.append("")

    # IP Strategy Advisor Assessment
    ip_text = result.get("ip_assessment", "")
    if ip_text:
        lines.append("---\n\n## IP Strategy Advisor Assessment\n")
        lines.append(ip_text)
        lines.append("")

    # Rebuttals
    lines.append("---\n\n# Debate Rebuttals\n")
    for role, label in [("cso", "CSO"), ("cto", "CTO"), ("cmo", "CMO"), ("cbo", "CBO")]:
        lines.append(f"## {label} Rebuttal\n")
        lines.append(result.get(f"{role}_rebuttal", "*No rebuttal available*"))
        lines.append("")

    # Verdict
    lines.append("---\n\n# Portfolio Director Verdict\n")
    lines.append(result.get("portfolio_verdict", "*No verdict available*"))
    lines.append("")

    # Scores
    scores = result.get("executive_scores", {})
    if "per_dimension" in scores:
        lines.append("\n## Weighted Scores\n")
        for dim, val in scores["per_dimension"].items():
            lines.append(f"- **{dim.replace('_', ' ').title()}:** {val}/10")
        lines.append(f"- **Weighted Total:** {scores.get('weighted_total', 'N/A')}/10")

    # PMID Citation Verification (NCBI E-utilities)
    pmid_verification = (session or {}).get("pmid_verification", {}) if session else {}
    if pmid_verification:
        n_total = sum(1 for k in pmid_verification.keys() if not k.startswith("_"))
        n_valid = sum(1 for k, v in pmid_verification.items()
                      if not k.startswith("_") and v.get("valid"))
        lines.append("\n---\n")
        lines.append(f"\n## Citation Verification (NCBI PubMed)\n")
        lines.append(f"**{n_valid}/{n_total} PMIDs verified against PubMed E-utilities**\n")
        for pmid in sorted(pmid_verification.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            if pmid.startswith("_"):
                continue
            v = pmid_verification[pmid]
            if v.get("valid"):
                title = v.get("title", "")
                year = v.get("year", "")
                journal = v.get("journal", "")
                lines.append(f"- ✅ **VERIFIED** PMID {pmid}: {title} — *{journal}* ({year})")
            else:
                err = v.get("error", "not found")
                lines.append(f"- ⚠️ **UNVERIFIED** PMID {pmid}: {err}")
        lines.append("")

    lines.append(f"\n---\n\n*Generated by APEX — Agentic Pipeline for Executive Decisions*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DDP Download Endpoint
# ---------------------------------------------------------------------------


@app.get("/download/ddp/{session_id}")
def download_ddp(session_id: str):
    """Download the generated DDP PDF for a session.

    Returns 404 if the planning pipeline has not been run yet.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session = sessions[session_id]
    pdf_path = session.get("ddp_pdf_path")

    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(
            status_code=404,
            detail="DDP PDF not yet generated. Run the planning pipeline first.",
        )

    gene, indication = _extract_gene_indication(session)
    filename = f"APEX_DDP_{gene}_{indication.replace(' ', '_')}.pdf"
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )


# ---------------------------------------------------------------------------
# Reject Endpoint
# ---------------------------------------------------------------------------


class RejectRequest(BaseModel):
    reason: Optional[str] = Field(default="", description="Optional CEO rejection note")


@app.post("/reject/{session_id}")
def reject_session(session_id: str, req: Optional[RejectRequest] = None):
    """Mark a session as rejected by the CEO.

    Updates ddp_status to 'rejected' and returns a summary including
    what evidence would be needed to revisit the decision.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session = sessions[session_id]
    if session["status"] != "complete":
        raise HTTPException(status_code=400, detail="Session is not complete yet")

    result = session["result"]
    gene, indication = _extract_gene_indication(session)

    # Update session state
    session["result"]["ddp_status"] = "rejected"
    session["ddp_status"] = "rejected"
    session["rejection_reason"] = (req.reason if req else "") or ""
    session["rejected_at"] = datetime.now(timezone.utc).isoformat()

    verdict_short = _parse_verdict_short(result.get("portfolio_verdict", ""))
    evidence_needed = _extract_evidence_needed(result.get("portfolio_verdict", ""))

    return {
        "status": "rejected",
        "session_id": session_id,
        "gene": gene,
        "indication": indication,
        "verdict": verdict_short,
        "confidence_score": result.get("confidence_score", 0),
        "reason": req.reason or "No reason provided.",
        "evidence_needed": evidence_needed,
        "rejected_at": session["rejected_at"],
    }


# ---------------------------------------------------------------------------
# CEO Feedback Endpoint — Human-in-the-Loop (Feature 4)
# ---------------------------------------------------------------------------


@app.post("/feedback/{session_id}")
async def submit_feedback(session_id: str, req: FeedbackRequest):
    """Submit CEO feedback on a completed evaluation, triggering a re-evaluation.

    The feedback is injected into agent prompts for the next round.
    Returns a new session_id for the re-evaluation.

    Optional action field:
      "approve" — triggers the DDP planning pipeline synchronously
      "reject"  — delegates to the reject endpoint logic
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session = sessions[session_id]
    if session["status"] != "complete":
        raise HTTPException(status_code=400, detail="Session is not complete yet")

    # Handle action shortcuts
    if req.action == "approve":
        # Trigger planning pipeline synchronously
        state = _make_planning_state(session)
        plan_result = await compiled_planning_graph.ainvoke(state)

        gene, indication = _extract_gene_indication(session)
        pdf_path = generate_ddp_pdf(
            gene=gene,
            indication=indication,
            verdict=_parse_verdict_short(session["result"].get("portfolio_verdict", "")),
            confidence=float(session["result"].get("confidence_score", 0)),
            cso_plan=plan_result.get("cso_plan", ""),
            cto_plan=plan_result.get("cto_plan", ""),
            cmo_plan=plan_result.get("cmo_plan", ""),
            cbo_plan=plan_result.get("cbo_plan", ""),
            ip_attorney_plan=plan_result.get("ip_attorney_plan", ""),
            director_synthesis=plan_result.get("director_synthesis", ""),
            session_id=session_id,
            executive_scores=session["result"].get("executive_scores", {}),
        )

        for field in ["cso_plan", "cto_plan", "cmo_plan", "cbo_plan", "ip_attorney_plan", "director_synthesis", "ddp_status"]:
            session["result"][field] = plan_result.get(field, "")
        session["result"].setdefault("activity_log", []).extend(plan_result.get("activity_log", []))
        session["ddp_pdf_path"] = pdf_path
        session["ddp_status"] = "complete"

        return {
            "action": "approve",
            "session_id": session_id,
            "ddp_status": "complete",
            "pdf_url": f"/download/ddp/{session_id}",
            "estimated_cost_usd": _sum_costs(plan_result),
        }

    if req.action == "reject":
        gene, indication = _extract_gene_indication(session)
        result = session["result"]
        session["result"]["ddp_status"] = "rejected"
        session["ddp_status"] = "rejected"
        session["rejection_reason"] = req.feedback
        session["rejected_at"] = datetime.now(timezone.utc).isoformat()
        return {
            "action": "reject",
            "status": "rejected",
            "session_id": session_id,
            "gene": gene,
            "indication": indication,
            "verdict": _parse_verdict_short(result.get("portfolio_verdict", "")),
            "reason": req.feedback,
            "evidence_needed": _extract_evidence_needed(result.get("portfolio_verdict", "")),
            "rejected_at": session["rejected_at"],
        }

    original_result = session["result"]
    original_query = session["query"]
    eval_round = original_result.get("evaluation_round", 0) + 1

    # Create new session for re-evaluation
    new_session_id = str(uuid.uuid4())
    graph = build_graph()

    # Build state with CEO feedback + previous scout data (skip re-scouting)
    state = make_initial_state(original_query)
    state["ceo_feedback"] = req.feedback
    state["ceo_feedback_history"] = original_result.get("ceo_feedback_history", []) + [
        {
            "feedback": req.feedback,
            "round": eval_round,
            "from_session": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ]
    state["evaluation_round"] = eval_round
    # Carry forward scout data to avoid redundant PubMed calls
    state["scout_data"] = original_result.get("scout_data", "")
    state["scout_sources"] = original_result.get("scout_sources", [])

    result = await graph.ainvoke(state)

    sessions[new_session_id] = {
        "session_id": new_session_id,
        "query": original_query,
        "status": "complete",
        "result": result,
        "parent_session": session_id,
        "ceo_feedback": req.feedback,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "session_id": new_session_id,
        "parent_session": session_id,
        "evaluation_round": eval_round,
        "ceo_feedback": req.feedback,
        "confidence_score": result.get("confidence_score", 0),
        "verdict": result.get("portfolio_verdict", "")[:200],
        "debate_rounds": result.get("debate_round", 0),
        "estimated_cost_usd": _sum_costs(result),
    }


# ---------------------------------------------------------------------------
# Directed CEO feedback — single-agent LLM call (no full graph)
# ---------------------------------------------------------------------------

_VALID_DIRECTED_ROLES = {"cso", "cto", "cmo", "cbo", "director"}

_DIRECTED_FEEDBACK_PROMPT = """\
The CEO asks you directly: "{feedback}"

Context: You just evaluated {query} and your assessment is below.

YOUR PREVIOUS ASSESSMENT:
{previous_assessment}

Answer the CEO's question directly and concisely — 3-5 sentences maximum. \
Speak naturally as if you're in a boardroom meeting, not writing a report. \
If they ask about data, cite 1-2 key papers. If they ask your opinion, give it clearly. \
Don't repeat your full assessment. Don't use headers or score formats. \
Just answer the question like a human executive would."""


async def _directed_agent_call(
    role: str,
    feedback: str,
    session: dict,
) -> tuple[str, float]:
    """Make a single LLM call to one agent with CEO feedback.

    Returns (response_text, cost_usd).
    """
    from agents import (
        ANTHROPIC_API_KEY, LLM_MODEL_STRONG, LLM_TEMPERATURE,
        LLM_TIMEOUT_SECONDS, ROLE_MODELS, estimate_cost, llm_semaphore,
    )
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    result = session["result"]
    query = session["query"]

    # Pick the right system prompt and previous text
    if role == "director":
        system_prompt = DIRECTOR_SYSTEM_PROMPT
        role_label = "Portfolio Director"
        previous = result.get("portfolio_verdict", "No previous verdict.")
    else:
        system_prompt = EXECUTIVE_PROMPTS[role]
        role_label = EXECUTIVE_ROLES[role]
        # Use rebuttal if available, else assessment
        previous = result.get(f"{role}_rebuttal") or result.get(f"{role}_assessment", "")

    user_prompt = _DIRECTED_FEEDBACK_PROMPT.format(
        query=query,
        previous_assessment=previous[:1500],
        feedback=feedback,
    )

    # Use role-specific model (director → Sonnet, others → per ROLE_MODELS)
    model_key = "portfolio_director" if role == "director" else role
    model = ROLE_MODELS.get(model_key, LLM_MODEL_STRONG)
    llm = ChatAnthropic(
        model=model,
        temperature=LLM_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=500,
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    cost = 0.0

    async with llm_semaphore:
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            usage = getattr(response, "usage_metadata", None) or {}
            cost = estimate_cost(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                model=model,
            )
            return response.content, cost
        except asyncio.TimeoutError:
            return f"[{role_label} timed out — please try again]", 0.0
        except Exception as e:
            return f"[Error: {str(e)[:200]}]", 0.0


# ---------------------------------------------------------------------------
# WebSocket Feedback — directed single-agent OR full re-evaluation
# ---------------------------------------------------------------------------


@app.websocket("/ws/feedback/{session_id}")
async def ws_feedback(websocket: WebSocket, session_id: str):
    """WebSocket CEO feedback handler.

    Client sends:
      Directed:  {"message": "@elena what data?", "directed_to": "cso"}
      Full:      {"message": "re-evaluate with focus on safety"}
    """
    await websocket.accept()
    keepalive_task = asyncio.create_task(_ws_keepalive(websocket))

    try:
        # Validate session
        if session_id not in sessions:
            await websocket.send_json({
                "type": "error",
                "node": "system",
                "data": {"message": f"Session {session_id} not found"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await websocket.close()
            return

        session = sessions[session_id]
        if session["status"] != "complete":
            await websocket.send_json({
                "type": "error",
                "node": "system",
                "data": {"message": "Session is not complete yet"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await websocket.close()
            return

        # Parse incoming message
        data = await websocket.receive_json()
        feedback = (data.get("feedback") or data.get("message") or "").strip()
        directed_to = (data.get("directed_to") or "").strip().lower()

        if not feedback or len(feedback) < 3:
            await websocket.send_json({
                "type": "error",
                "node": "system",
                "data": {"message": "Feedback must be at least 3 characters"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await websocket.close()
            return

        original_result = session["result"]
        original_query = session["query"]
        ts = datetime.now(timezone.utc).isoformat()

        # -----------------------------------------------------------
        # DIRECTED: single-agent response (~10s instead of 90s)
        # -----------------------------------------------------------
        if directed_to and directed_to in _VALID_DIRECTED_ROLES:
            agent_role = directed_to if directed_to != "director" else "portfolio_director"
            persona = AGENT_PERSONAS.get(agent_role, AGENT_PERSONAS.get(directed_to, {}))
            node_name = f"{directed_to}_feedback"

            # Send session start
            await websocket.send_json({
                "type": "session_start",
                "node": "system",
                "data": {
                    "session_id": session_id,
                    "query": original_query,
                    "directed_to": directed_to,
                    "ceo_feedback": feedback,
                },
                "timestamp": ts,
            })

            # Send node_start
            await websocket.send_json({
                "type": "node_start",
                "node": node_name,
                "agent": agent_role,
                "persona": persona,
                "timestamp": ts,
            })

            # Single LLM call
            response_text, cost = await _directed_agent_call(
                directed_to, feedback, session,
            )

            # Send node_complete with the response
            await websocket.send_json({
                "type": "node_complete",
                "node": node_name,
                "agent": agent_role,
                "persona": persona,
                "data": {
                    "content": response_text,
                    "directed_to": directed_to,
                    "ceo_feedback": feedback,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Send complete
            await websocket.send_json({
                "type": "complete",
                "node": "system",
                "data": {
                    "session_id": session_id,
                    "directed_to": directed_to,
                    "ceo_feedback": feedback,
                    "confidence_score": original_result.get("confidence_score", 0),
                    "verdict": _parse_verdict_short(original_result.get("portfolio_verdict", "")),
                    "debate_rounds": original_result.get("debate_round", 0),
                    "estimated_cost_usd": cost,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return

        # -----------------------------------------------------------
        # FULL re-evaluation: all agents + debate + director
        # -----------------------------------------------------------
        eval_round = original_result.get("evaluation_round", 0) + 1
        new_session_id = str(uuid.uuid4())

        await websocket.send_json({
            "type": "session_start",
            "node": "system",
            "data": {
                "session_id": new_session_id,
                "query": original_query,
                "parent_session": session_id,
                "evaluation_round": eval_round,
                "ceo_feedback": feedback,
            },
            "timestamp": ts,
        })

        graph = build_graph()
        state = make_initial_state(original_query)
        state["ceo_feedback"] = feedback
        state["ceo_feedback_history"] = original_result.get("ceo_feedback_history", []) + [
            {
                "feedback": feedback,
                "round": eval_round,
                "from_session": session_id,
                "timestamp": ts,
            }
        ]
        state["evaluation_round"] = eval_round
        state["scout_data"] = original_result.get("scout_data", "")
        state["scout_sources"] = original_result.get("scout_sources", [])

        current_round = 0
        final_result = dict(state)

        async for event in graph.astream(state, stream_mode="updates"):
            for node_name, node_data in event.items():
                evt_ts = datetime.now(timezone.utc).isoformat()

                if node_data is None:
                    continue

                if node_name == "portfolio_director":
                    current_round += 1

                agent_role = _NODE_AGENT_MAP.get(node_name, "system")
                persona = AGENT_PERSONAS.get(agent_role, {})
                await websocket.send_json({
                    "type": "node_complete",
                    "node": node_name,
                    "agent": agent_role,
                    "persona": persona,
                    "data": _extract_event_data(node_name, node_data),
                    "timestamp": evt_ts,
                })

                for k, v in node_data.items():
                    if k == "activity_log":
                        final_result.setdefault("activity_log", []).extend(v)
                    elif k == "executive_scores" and isinstance(v, dict):
                        final_result.setdefault("executive_scores", {}).update(v)
                    else:
                        final_result[k] = v

        sessions[new_session_id] = {
            "session_id": new_session_id,
            "query": original_query,
            "status": "complete",
            "result": final_result,
            "parent_session": session_id,
            "ceo_feedback": feedback,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        verdict_short = _parse_verdict_short(final_result.get("portfolio_verdict", ""))
        await websocket.send_json({
            "type": "complete",
            "node": "system",
            "data": {
                "session_id": new_session_id,
                "parent_session": session_id,
                "evaluation_round": eval_round,
                "ceo_feedback": feedback,
                "confidence_score": final_result.get("confidence_score", 0),
                "verdict": verdict_short,
                "debate_rounds": final_result.get("debate_round", 0),
                "estimated_cost_usd": _sum_costs(final_result),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "node": "system",
                "data": {"message": f"Feedback handler error: {str(e)[:500]}"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        keepalive_task.cancel()


# ---------------------------------------------------------------------------
# Batch Evaluation — sequential multi-target evaluation for HW8 scale testing
# ---------------------------------------------------------------------------

batch_sessions: dict[str, dict[str, Any]] = {}


class BatchEvaluateRequest(BaseModel):
    targets: list[str] = Field(..., min_length=1, max_length=10, description="List of gene targets")
    indication: str = Field(default="knee osteoarthritis", description="Disease indication")


async def _run_batch(batch_id: str, targets: list[str], indication: str):
    """Run sequential evaluations for a batch of targets."""
    import time as _time

    batch = batch_sessions[batch_id]
    batch["status"] = "running"

    for i, target in enumerate(targets):
        query = f"{target} {indication}"
        target_result = {
            "target": target,
            "query": query,
            "status": "running",
            "session_id": None,
            "verdict": None,
            "confidence": None,
            "composite_score": None,
            "ip_score": None,
            "fto_risk": None,
            "citations": 0,
            "wall_clock_seconds": 0,
            "agent_invocations": 0,
            "cost_usd": 0.0,
            "error": None,
        }
        batch["results"].append(target_result)

        t_start = _time.time()
        try:
            graph = build_graph()
            state = make_initial_state(query)
            result = await graph.ainvoke(state)

            wall_clock = round(_time.time() - t_start, 1)
            session_id = str(uuid.uuid4())
            cost = _sum_costs(result)

            # PMID verification
            pmid_verification = {}
            try:
                pmid_verification = await verify_pmids(result, max_pmids=20)
            except Exception:
                pass

            n_pmid_total = sum(1 for k in pmid_verification if not k.startswith("_"))
            n_pmid_valid = sum(1 for k, v in pmid_verification.items()
                               if not k.startswith("_") and v.get("valid"))

            # Store session (same as regular evaluate)
            sessions[session_id] = {
                "session_id": session_id,
                "query": query,
                "status": "complete",
                "result": result,
                "pmid_verification": pmid_verification,
                "pmid_summary": {"verified": n_pmid_valid, "total": n_pmid_total},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Extract IP score from activity log
            ip_score = None
            fto_risk = None
            for entry in result.get("activity_log", []):
                if entry.get("node") == "ip_assess":
                    ip_score = entry.get("ip_score")
                    fto_risk = entry.get("fto_risk")

            # Count agent invocations from activity log
            n_invocations = len(result.get("activity_log", []))

            # Extract verdict
            verdict_short = _parse_verdict_short(result.get("portfolio_verdict", ""))
            composite = result.get("executive_scores", {}).get("weighted_total", 0)

            target_result.update({
                "status": "complete",
                "session_id": session_id,
                "verdict": verdict_short,
                "confidence": result.get("confidence_score", 0),
                "composite_score": composite,
                "ip_score": ip_score,
                "fto_risk": fto_risk,
                "citations": n_pmid_total,
                "citations_verified": n_pmid_valid,
                "wall_clock_seconds": wall_clock,
                "agent_invocations": n_invocations,
                "cost_usd": cost,
            })

        except Exception as e:
            wall_clock = round(_time.time() - t_start, 1)
            target_result.update({
                "status": "failed",
                "wall_clock_seconds": wall_clock,
                "error": str(e)[:300],
            })

        batch["completed"] = i + 1

        # Update batch totals
        completed_results = [r for r in batch["results"] if r["status"] in ("complete", "failed")]
        batch["totals"] = {
            "total_agent_invocations": sum(r.get("agent_invocations", 0) for r in completed_results),
            "total_wall_clock_seconds": round(sum(r.get("wall_clock_seconds", 0) for r in completed_results), 1),
            "total_citations": sum(r.get("citations", 0) for r in completed_results),
            "total_citations_verified": sum(r.get("citations_verified", 0) for r in completed_results),
            "estimated_cost_usd": round(sum(r.get("cost_usd", 0) for r in completed_results), 4),
        }

        # Cooldown between evaluations (avoid rate limits)
        if i < len(targets) - 1:
            await asyncio.sleep(5)

    batch["status"] = "complete"
    batch["completed_at"] = datetime.now(timezone.utc).isoformat()



@app.post("/batch/evaluate")
async def batch_evaluate(req: BatchEvaluateRequest):
    """Start a batch evaluation of multiple targets sequentially.

    Returns immediately with a batch_id. Poll GET /batch/status/{batch_id}
    to track progress. Each target runs a full 11-agent evaluation.
    """
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    n_targets = len(req.targets)
    agents_per_eval = 11  # scout + 4 assess + ip + 4 rebuttals + director

    batch_sessions[batch_id] = {
        "batch_id": batch_id,
        "status": "starting",
        "targets": req.targets,
        "indication": req.indication,
        "completed": 0,
        "total": n_targets,
        "results": [],
        "totals": {
            "total_agent_invocations": 0,
            "total_wall_clock_seconds": 0,
            "total_citations": 0,
            "estimated_cost_usd": 0.0,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    # Launch in background
    asyncio.create_task(_run_batch(batch_id, req.targets, req.indication))

    return {
        "batch_id": batch_id,
        "targets": req.targets,
        "total_agent_invocations": n_targets * agents_per_eval,
        "status": "running",
        "message": f"Batch evaluation started. {n_targets} targets x {agents_per_eval} agents = {n_targets * agents_per_eval} agent invocations.",
    }


@app.get("/batch/status/{batch_id}")
def batch_status(batch_id: str):
    """Get status of a batch evaluation."""
    if batch_id not in batch_sessions:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    return batch_sessions[batch_id]


# ---------------------------------------------------------------------------
# WebSocket keepalive — prevent Railway/proxy timeout on long evaluations
# ---------------------------------------------------------------------------


async def _ws_keepalive(websocket: WebSocket, interval: int = 8) -> None:
    """Send ping frames every `interval` seconds to keep the connection alive.

    Lowered from 15s to 8s to better tolerate Railway's edge proxy idle timeout
    during the long-running Portfolio Director node (which can take 30-60s).
    """
    try:
        while True:
            await asyncio.sleep(interval)
            await websocket.send_json({
                "type": "ping",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    except Exception:
        pass  # Connection closed or send failed — just stop


async def _ws_aggressive_ping(websocket: WebSocket, duration_s: int = 90, interval_s: int = 5):
    """Send pings every `interval_s` seconds for `duration_s` total.

    Used during the Portfolio Director node, which is the bottleneck.
    Sonnet first-token can take 30+ seconds for the synthesis prompt; this
    aggressive ping ensures Railway's proxy never sees the connection as idle.
    """
    try:
        elapsed = 0
        while elapsed < duration_s:
            await asyncio.sleep(interval_s)
            elapsed += interval_s
            await websocket.send_json({
                "type": "ping",
                "phase": "director_thinking",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    except Exception:
        pass


# ---------------------------------------------------------------------------
# WebSocket Planning Endpoint — streams DDP planning as each section completes
# ---------------------------------------------------------------------------


@app.websocket("/ws/plan/{session_id}")
async def ws_plan(websocket: WebSocket, session_id: str):
    """WebSocket streaming DDP planning pipeline.

    Connect after a GO verdict is accepted. Streams:
      {"type": "plan_started"}
      {"type": "plan_progress", "agent": "cso", "status": "complete", "plan": "..."}
      {"type": "plan_progress", "agent": "cto", "status": "complete", "plan": "..."}
      {"type": "plan_progress", "agent": "cmo", "status": "complete", "plan": "..."}
      {"type": "plan_progress", "agent": "cbo", "status": "complete", "plan": "..."}
      {"type": "synthesis_complete", "synthesis": "..."}
      {"type": "ddp_complete", "pdf_url": "/download/ddp/{session_id}", "estimated_cost_usd": 0.XX}
    """
    await websocket.accept()
    keepalive_task = asyncio.create_task(_ws_keepalive(websocket))

    try:
        if session_id not in sessions:
            await websocket.send_json({
                "type": "error",
                "data": {"message": f"Session {session_id} not found"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await websocket.close()
            return

        session = sessions[session_id]
        if session["status"] != "complete":
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Session is not complete yet"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await websocket.close()
            return

        await websocket.send_json({
            "type": "plan_started",
            "data": {"session_id": session_id, "query": session["query"]},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        state = _make_planning_state(session)
        plan_result: dict[str, Any] = {}

        async for event in compiled_planning_graph.astream(state, stream_mode="updates"):
            for node_name, node_data in event.items():
                if node_data is None:
                    continue

                evt_ts = datetime.now(timezone.utc).isoformat()
                agent_role = _PLAN_NODE_AGENT_MAP.get(node_name, "system")

                if node_name == "director_synthesis":
                    await websocket.send_json({
                        "type": "synthesis_complete",
                        "agent": agent_role,
                        "data": {"synthesis": node_data.get("director_synthesis", "")},
                        "timestamp": evt_ts,
                    })
                elif node_name in _PLAN_NODE_AGENT_MAP:
                    # Node name IS the state field for all planner nodes except director
                    # (cso_plan, cto_plan, cmo_plan, cbo_plan, ip_attorney_plan all work directly)
                    field = node_name
                    await websocket.send_json({
                        "type": "plan_progress",
                        "agent": agent_role,
                        "status": "complete",
                        "data": {"plan": node_data.get(field, "")},
                        "timestamp": evt_ts,
                    })

                # Accumulate results
                for k, v in node_data.items():
                    if k == "activity_log":
                        plan_result.setdefault("activity_log", []).extend(v)
                    else:
                        plan_result[k] = v

        # Generate PDF
        gene, indication = _extract_gene_indication(session)
        pdf_path = generate_ddp_pdf(
            gene=gene,
            indication=indication,
            verdict=_parse_verdict_short(session["result"].get("portfolio_verdict", "")),
            confidence=float(session["result"].get("confidence_score", 0)),
            cso_plan=plan_result.get("cso_plan", ""),
            cto_plan=plan_result.get("cto_plan", ""),
            cmo_plan=plan_result.get("cmo_plan", ""),
            cbo_plan=plan_result.get("cbo_plan", ""),
            ip_attorney_plan=plan_result.get("ip_attorney_plan", ""),
            director_synthesis=plan_result.get("director_synthesis", ""),
            session_id=session_id,
            executive_scores=session["result"].get("executive_scores", {}),
        )

        # Persist results on session
        for field in ["cso_plan", "cto_plan", "cmo_plan", "cbo_plan", "ip_attorney_plan", "director_synthesis", "ddp_status"]:
            session["result"][field] = plan_result.get(field, "")
        session["result"].setdefault("activity_log", []).extend(plan_result.get("activity_log", []))
        session["ddp_pdf_path"] = pdf_path
        session["ddp_status"] = "complete"

        await websocket.send_json({
            "type": "ddp_complete",
            "data": {
                "session_id": session_id,
                "pdf_url": f"/download/ddp/{session_id}",
                "estimated_cost_usd": _sum_costs(plan_result),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": f"Planning pipeline error: {str(e)[:500]}"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        keepalive_task.cancel()


# ---------------------------------------------------------------------------
# WebSocket Endpoint — THE CORE DIFFERENTIATOR
# ---------------------------------------------------------------------------


@app.websocket("/ws/evaluate")
async def ws_evaluate(websocket: WebSocket):
    """WebSocket streaming evaluation.

    Client sends: {"query": "OSMR ulcerative colitis"}
    Server streams events as each node completes.
    """
    await websocket.accept()
    keepalive_task = asyncio.create_task(_ws_keepalive(websocket))

    try:
        data = await websocket.receive_json()
        query = data.get("query", "").strip()

        if not query or len(query) < 3:
            await websocket.send_json({
                "type": "error",
                "node": "system",
                "data": {"message": "Query must be at least 3 characters"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await websocket.close()
            return

        session_id = str(uuid.uuid4())

        # Send session start event
        await websocket.send_json({
            "type": "session_start",
            "node": "system",
            "data": {"session_id": session_id, "query": query},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        graph = build_graph()
        state = make_initial_state(query)
        current_round = 0
        final_result = dict(state)
        rebuttals_seen = 0
        director_ping_task = None

        async for event in graph.astream(state, stream_mode="updates"):
            for node_name, node_data in event.items():
                ts = datetime.now(timezone.utc).isoformat()

                # Skip None data (pass-through nodes)
                if node_data is None:
                    continue

                # Cancel aggressive Director ping task once Director completes
                if node_name == "portfolio_director" and director_ping_task is not None:
                    director_ping_task.cancel()
                    director_ping_task = None

                # Track debate rounds
                if node_name == "portfolio_director":
                    current_round += 1
                    new_round = node_data.get("debate_round", current_round)
                    if new_round > current_round:
                        # Entering a new debate round
                        await websocket.send_json({
                            "type": "debate_round",
                            "node": "system",
                            "data": {"round": new_round},
                            "timestamp": ts,
                        })

                # Send node_complete event with persona metadata
                agent_role = _NODE_AGENT_MAP.get(node_name, "system")
                persona = AGENT_PERSONAS.get(agent_role, {})
                event_data = _extract_event_data(node_name, node_data)

                # Detect Anthropic sentinel errors and translate to friendly message
                event_data_str = str(event_data)
                if "[ANTHROPIC_CREDIT_ERROR]" in event_data_str:
                    if director_ping_task is not None:
                        director_ping_task.cancel()
                        director_ping_task = None
                    await websocket.send_json({
                        "type": "error",
                        "node": "system",
                        "data": {
                            "message": "Evaluation paused — API credits need renewal. Please contact administrator.",
                            "code": "credit_balance_too_low",
                        },
                        "timestamp": ts,
                    })
                    await websocket.close()
                    return
                if "[ANTHROPIC_AUTH_ERROR]" in event_data_str:
                    if director_ping_task is not None:
                        director_ping_task.cancel()
                        director_ping_task = None
                    await websocket.send_json({
                        "type": "error",
                        "node": "system",
                        "data": {
                            "message": "Evaluation paused — system requires attention.",
                            "code": "auth_error",
                        },
                        "timestamp": ts,
                    })
                    await websocket.close()
                    return
                if "[ANTHROPIC_RATE_LIMIT]" in event_data_str:
                    if director_ping_task is not None:
                        director_ping_task.cancel()
                        director_ping_task = None
                    await websocket.send_json({
                        "type": "error",
                        "node": "system",
                        "data": {
                            "message": "Evaluation paused — please retry in a moment.",
                            "code": "rate_limit",
                        },
                        "timestamp": ts,
                    })
                    await websocket.close()
                    return

                await websocket.send_json({
                    "type": "node_complete",
                    "node": node_name,
                    "agent": agent_role,
                    "persona": persona,
                    "data": event_data,
                    "timestamp": ts,
                })

                # After the 4th rebuttal completes, the Portfolio Director is next.
                # Director uses Sonnet + 3000 max_tokens and can take 30-60s for first
                # token. Start aggressive ping (every 5s) to prevent Railway proxy
                # idle timeout. Task auto-cancels when director node completes.
                if node_name.endswith("_rebuttal"):
                    rebuttals_seen += 1
                    if rebuttals_seen % 4 == 0 and director_ping_task is None:
                        # Send an immediate ping right now too
                        await websocket.send_json({
                            "type": "ping",
                            "phase": "director_starting",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        director_ping_task = asyncio.create_task(
                            _ws_aggressive_ping(websocket, duration_s=120, interval_s=5)
                        )

                # Update final result
                for k, v in node_data.items():
                    if k == "activity_log":
                        final_result.setdefault("activity_log", []).extend(v)
                    elif k == "executive_scores" and isinstance(v, dict):
                        final_result.setdefault("executive_scores", {}).update(v)
                    else:
                        final_result[k] = v

        # PMID auto-verification — runs against NCBI E-utilities
        # Catches hallucinated citations before they reach the user
        pmid_verification = {}
        try:
            await websocket.send_json({
                "type": "verifying_citations",
                "node": "system",
                "data": {"message": "Verifying PMID citations against PubMed..."},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            pmid_verification = await verify_pmids(final_result, max_pmids=20)
        except Exception as e:
            # Never let citation verification break the evaluation
            pmid_verification = {"_error": str(e)[:200]}

        n_total = sum(1 for k in pmid_verification.keys() if not k.startswith("_"))
        n_valid = sum(1 for k, v in pmid_verification.items()
                      if not k.startswith("_") and v.get("valid"))

        # Store session (with PMID verification)
        sessions[session_id] = {
            "session_id": session_id,
            "query": query,
            "status": "complete",
            "result": final_result,
            "pmid_verification": pmid_verification,
            "pmid_summary": {"verified": n_valid, "total": n_total},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Send citation verification event
        await websocket.send_json({
            "type": "citations_verified",
            "node": "system",
            "data": {
                "verified": n_valid,
                "total": n_total,
                "verification": pmid_verification,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Send completion event with parsed verdict
        verdict_short = _parse_verdict_short(final_result.get("portfolio_verdict", ""))
        await websocket.send_json({
            "type": "complete",
            "node": "system",
            "data": {
                "session_id": session_id,
                "confidence_score": final_result.get("confidence_score", 0),
                "verdict": verdict_short,
                "debate_rounds": final_result.get("debate_round", 0),
                "weighted_total": final_result.get("executive_scores", {}).get("weighted_total", 0),
                "estimated_cost_usd": _sum_costs(final_result),
                "pmid_verified": n_valid,
                "pmid_total": n_total,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        sessions[session_id]["result"]["_verdict_short"] = verdict_short

    except WebSocketDisconnect:
        pass  # Client disconnected mid-evaluation
    except Exception as e:
        try:
            err_str = str(e).lower()
            if "credit balance" in err_str or "billing" in err_str or "insufficient" in err_str:
                friendly_msg = "Evaluation paused — API credits need renewal. Please contact administrator."
                code = "credit_balance_too_low"
            elif "rate limit" in err_str or "rate_limit" in err_str:
                friendly_msg = "Evaluation paused — please retry in a moment."
                code = "rate_limit"
            elif "authentication" in err_str or "invalid api key" in err_str:
                friendly_msg = "Evaluation paused — system requires attention."
                code = "auth_error"
            else:
                friendly_msg = str(e)[:500]
                code = "unknown"
            await websocket.send_json({
                "type": "error",
                "node": "system",
                "data": {"message": friendly_msg, "code": code},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        try:
            if 'director_ping_task' in locals() and director_ping_task is not None:
                director_ping_task.cancel()
        except Exception:
            pass
        keepalive_task.cancel()

"""Single-agent Q&A subgraph for @mention routing.

Builds a minimal LangGraph pipeline that routes a directed question to ONE
agent (no debate, no rebuttals, no director synthesis). Target latency <10s.

Graph:
    START -> rag_retrieve -> single_agent_llm -> claim_verify -> END

Each node is a pure-async function that takes AskState and returns updates.
Designed for both the REST POST /ask and WebSocket /ws/ask endpoints.
"""

from __future__ import annotations

import asyncio
import operator
import re
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents import LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS, llm_semaphore
from agents.llm_router import estimate_cost_from_response, get_llm
from agents.prompts import AGENT_PERSONAS, EXECUTIVE_PROMPTS, DIRECTOR_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AskState(TypedDict, total=False):
    # Inputs
    query: str                              # question with @mention already stripped
    role: str                               # canonical role key (cso/cto/cmo/cbo/ip_attorney/portfolio_director)
    provider: str                           # LLM provider: anthropic | openai | google
    session_id: Optional[str]               # optional prior session for follow-up context
    prior_assessment: Optional[str]         # prior agent output from that session, if any

    # Node outputs
    rag_context: str                        # retrieved knowledge-base context
    answer: str                             # agent's response
    claims: list[dict]                      # extracted [SUPPORTED]/[UNSUPPORTED]/[UNCERTAIN] tags
    cost_usd: float                         # LLM cost for this single call

    # Instrumentation
    activity_log: Annotated[list[dict], operator.add]


# ---------------------------------------------------------------------------
# Role -> system prompt
# ---------------------------------------------------------------------------

def _system_prompt_for(role: str) -> str:
    """Return the persona system prompt for a role, including the director."""
    if role == "portfolio_director":
        return DIRECTOR_SYSTEM_PROMPT
    return EXECUTIVE_PROMPTS[role]


# ---------------------------------------------------------------------------
# Q&A-specific user prompt wrapper
# ---------------------------------------------------------------------------

_ASK_USER_TEMPLATE = """\
You are answering a DIRECTED QUESTION from the reviewer during a review session.
This is not a full target assessment — it's a focused question to you specifically.

{prior_section}\
{context_section}\
QUESTION FROM REVIEWER: {query}

Answer briefly — hard cap 300 words, aim for 150-250. Use your role and
domain expertise, but DO NOT output the full structured assessment format
(no "### Key Findings", no "### Scores", no "### Verdict"). Write a direct
conversational answer grounded in the provided context. Be decisive and cite
sources. Time budget: your entire answer should fit in one phone screen.

REQUIRED: tag every factual claim with exactly one of:
  [SUPPORTED: the claim] — directly backed by the retrieved context above
  [UNSUPPORTED: the claim] — asserted without evidence in the context
  [UNCERTAIN: the claim] — context is weak, partial, or conflicting

Cite PMIDs or source file names from the context where possible.
If the context does not answer the question, say so plainly and tag the
overall answer [UNCERTAIN].
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def rag_retrieve_node(state: AskState) -> dict[str, Any]:
    """Retrieve top-k context from the role's collection + shared collection."""
    role = state["role"]
    query = state["query"]
    t0 = time.time()

    # Map portfolio_director to shared-only (no dedicated collection).
    retrieve_role = role if role != "portfolio_director" else "shared"

    try:
        from rag.retriever import retrieve_context
        context = retrieve_context(retrieve_role, query, k=5)
    except Exception as e:
        # Graceful degradation: if RAG is offline, continue with no context.
        print(f"[/ask rag_retrieve] {type(e).__name__}: {str(e)[:200]}")
        context = ""

    elapsed = time.time() - t0
    return {
        "rag_context": context,
        "activity_log": [{
            "node": "rag_retrieve",
            "role": role,
            "elapsed_s": round(elapsed, 3),
            "context_chars": len(context),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }


async def single_agent_llm_node(state: AskState) -> dict[str, Any]:
    """Call the single agent's LLM with persona + context + question."""
    role = state["role"]
    query = state["query"]
    provider = state.get("provider") or "anthropic"
    rag_context = state.get("rag_context", "")
    prior = state.get("prior_assessment")

    llm = get_llm(role, provider=provider, temperature=LLM_TEMPERATURE, max_tokens=900)

    # Build context and prior sections for the user prompt
    context_section = ""
    if rag_context:
        context_section = f"RETRIEVED CONTEXT:\n{rag_context}\n\n"

    prior_section = ""
    if prior:
        prior_section = (
            "YOUR EARLIER ASSESSMENT (from the same review session):\n"
            f"{prior[:3000]}\n\n"
            "Build on that assessment where relevant; do not repeat it verbatim.\n\n"
        )

    system_prompt = _system_prompt_for(role)
    user_prompt = _ASK_USER_TEMPLATE.format(
        prior_section=prior_section,
        context_section=context_section,
        query=query,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    t0 = time.time()
    answer = ""
    cost = 0.0
    async with llm_semaphore:
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            answer = response.content
            cost = estimate_cost_from_response(provider, role, response)
        except asyncio.TimeoutError:
            answer = f"[Agent timed out after {LLM_TIMEOUT_SECONDS}s]"
        except Exception as e:
            err_str = str(e).lower()
            if "credit balance" in err_str or "billing" in err_str or "insufficient" in err_str:
                answer = "[PROVIDER_CREDIT_ERROR] API credits need renewal"
            elif "rate limit" in err_str or "rate_limit" in err_str or "429" in err_str:
                answer = "[PROVIDER_RATE_LIMIT] Too many requests — please retry"
            elif (
                "authentication_error" in err_str
                or "invalid api key" in err_str
                or "invalid x-api-key" in err_str
                or "401 unauthorized" in err_str
                or "status_code=401" in err_str
            ):
                answer = "[PROVIDER_AUTH_ERROR] API key invalid"
            else:
                print(f"[/ask single_agent_llm role={role}] non-fatal: {type(e).__name__}: {str(e)[:300]}")
                answer = f"[Agent transient error: {type(e).__name__}]"

    elapsed = time.time() - t0
    return {
        "answer": answer,
        "cost_usd": cost,
        "activity_log": [{
            "node": "single_agent_llm",
            "role": role,
            "provider": provider,
            "elapsed_s": round(elapsed, 3),
            "answer_chars": len(answer),
            "cost_usd": cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }


# Regex for [SUPPORTED: ...] / [UNSUPPORTED: ...] / [UNCERTAIN: ...] tokens.
# Matches across newlines; caps each tag content at ~400 chars to avoid
# runaway captures when the LLM forgets the closing bracket.
_CLAIM_TAG_RE = re.compile(
    r"\[(SUPPORTED|UNSUPPORTED|UNCERTAIN)\s*:\s*(.*?)\]",
    re.IGNORECASE | re.DOTALL,
)


async def claim_verify_node(state: AskState) -> dict[str, Any]:
    """Extract Self-RAG reflection tokens and attach them as structured claims.

    This is a FAST extraction (regex only) — deliberately does NOT call
    PubMed. For @mention routing we want <10s total latency; full claim
    verification via rag/claim_verifier.verify_claims() would add 15-40s.
    The frontend can surface the tags inline with the answer.
    """
    answer = state.get("answer", "")
    claims: list[dict] = []
    for match in _CLAIM_TAG_RE.finditer(answer):
        tag = match.group(1).upper()
        claim_text = match.group(2).strip()[:400]
        claims.append({"tag": tag, "text": claim_text})

    return {
        "claims": claims,
        "activity_log": [{
            "node": "claim_verify",
            "n_supported":   sum(1 for c in claims if c["tag"] == "SUPPORTED"),
            "n_unsupported": sum(1 for c in claims if c["tag"] == "UNSUPPORTED"),
            "n_uncertain":   sum(1 for c in claims if c["tag"] == "UNCERTAIN"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_compiled_graph = None


def build_ask_graph():
    """Build and compile the single-agent @ask subgraph."""
    g = StateGraph(AskState)
    g.add_node("rag_retrieve", rag_retrieve_node)
    g.add_node("single_agent_llm", single_agent_llm_node)
    g.add_node("claim_verify", claim_verify_node)

    g.add_edge(START, "rag_retrieve")
    g.add_edge("rag_retrieve", "single_agent_llm")
    g.add_edge("single_agent_llm", "claim_verify")
    g.add_edge("claim_verify", END)

    return g.compile()


def get_compiled_ask_graph():
    """Lazy-compile (avoid import-time overhead)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_ask_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_prior_assessment(session_record: dict, role: str) -> Optional[str]:
    """Pull the role's prior assessment out of a session record (if any).

    Looks in ``session["result"]`` for completed sessions, then
    ``session["partial_result"]`` for in-progress sessions.
    Returns None if no prior output is available for that role.
    """
    if not session_record:
        return None

    role_to_field = {
        "cso": "cso_assessment",
        "cto": "cto_assessment",
        "cmo": "cmo_assessment",
        "cbo": "cbo_assessment",
        "ip_attorney": "ip_assessment",
        "portfolio_director": "portfolio_verdict",
    }
    field = role_to_field.get(role)
    if not field:
        return None

    for source_key in ("result", "partial_result"):
        source = session_record.get(source_key) or {}
        val = source.get(field)
        if val and isinstance(val, str) and val.strip():
            return val
    return None


async def run_ask(
    role: str,
    query: str,
    provider: str = "anthropic",
    session_id: Optional[str] = None,
    prior_assessment: Optional[str] = None,
) -> dict[str, Any]:
    """Run the single-agent ask graph end-to-end and return a flat result dict.

    Args:
        role: canonical role key (cso/cto/cmo/cbo/ip_attorney/portfolio_director)
        query: the question with @mention already stripped
        provider: LLM backend (anthropic | openai | google)
        session_id: optional prior session for follow-up context
        prior_assessment: optional pre-fetched prior assessment (saves a lookup)

    Returns:
        dict with keys: role, query, answer, claims, rag_context, cost_usd,
        agent_persona, activity_log, elapsed_s
    """
    t0 = time.time()
    graph = get_compiled_ask_graph()

    initial_state: AskState = {
        "role": role,
        "query": query,
        "provider": provider,
        "session_id": session_id,
        "prior_assessment": prior_assessment,
        "rag_context": "",
        "answer": "",
        "claims": [],
        "cost_usd": 0.0,
        "activity_log": [],
    }

    final_state = await graph.ainvoke(initial_state)
    elapsed = time.time() - t0

    return {
        "role": role,
        "agent_persona": AGENT_PERSONAS.get(role, {}),
        "query": query,
        "session_id": session_id,
        "answer": final_state.get("answer", ""),
        "claims": final_state.get("claims", []),
        "rag_context_chars": len(final_state.get("rag_context", "")),
        "cost_usd": final_state.get("cost_usd", 0.0),
        "activity_log": final_state.get("activity_log", []),
        "elapsed_s": round(elapsed, 3),
    }

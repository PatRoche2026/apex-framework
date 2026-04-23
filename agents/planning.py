"""DDP planning nodes — each executive drafts their section after a GO verdict.

Pattern mirrors executives.py exactly:
  system_prompt = EXECUTIVE_PROMPTS[role]       (existing persona system prompts)
  user_prompt   = DDP_PLANNING_PROMPTS[role]    (DDP planning user prompts)

Includes the same two-pass tool use pattern: first pass generates the DDP section,
tool requests are parsed and executed, second pass incorporates tool results.
RAG context is injected the same way as assessment nodes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents import (
    ANTHROPIC_API_KEY,
    LLM_MODEL_STRONG,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    ROLE_MODELS,
    estimate_cost,
    llm_semaphore,
)
from agents.prompts import (
    DDP_PLANNING_PROMPTS,
    DIRECTOR_SYSTEM_PROMPT,
    EXECUTIVE_PROMPTS,
    EXECUTIVE_ROLES,
    TOOL_FOLLOWUP_TEMPLATE,
)
from agents.state import APEXState
from agents.tools import execute_tool_requests, parse_tool_requests

# RAG is optional — gracefully degrade if not installed
try:
    from rag.retriever import retrieve_context as _retrieve_context
except ImportError:
    _retrieve_context = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# LLM instance cache — same pattern as executives.py
# ---------------------------------------------------------------------------

_llm_cache: dict[str, ChatAnthropic] = {}


def _get_llm(model: str) -> ChatAnthropic:
    if model not in _llm_cache:
        _llm_cache[model] = ChatAnthropic(
            model=model,
            temperature=LLM_TEMPERATURE,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=3000,  # DDP sections are longer than assessments
        )
    return _llm_cache[model]


async def _call_llm(system_prompt: str, user_prompt: str, role: str) -> tuple[str, float]:
    """Rate-limited LLM call using role-specific model. Returns (text, cost_usd)."""
    model = ROLE_MODELS.get(role, LLM_MODEL_STRONG)
    llm = _get_llm(model)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
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
            return "[Agent timed out — DDP section incomplete]", 0.0
        except Exception as e:
            return f"[Agent error: {str(e)[:200]}]", 0.0


# ---------------------------------------------------------------------------
# Context builders — summarise evaluation state for the planning prompts
# ---------------------------------------------------------------------------


def _gene_and_indication(state: APEXState) -> tuple[str, str]:
    """Extract gene and indication from state, falling back to query split."""
    gene = state.get("gene", "").strip()
    indication = state.get("indication", "").strip()
    if not gene or not indication:
        tokens = state["query"].strip().split()
        gene = gene or (tokens[0].upper() if tokens else state["query"])
        indication = indication or (" ".join(tokens[1:]) if len(tokens) > 1 else state["query"])
    return gene, indication


def _assessment_summary(state: APEXState) -> str:
    parts = []
    for role, label in [
        ("cso", "Scientific Advisor"),
        ("cto", "Technical Advisor"),
        ("cmo", "Clinical Advisor"),
        ("cbo", "Commercial Advisor"),
    ]:
        text = state.get(f"{role}_assessment", "").strip()
        if text:
            parts.append(f"--- {label} ---\n{text}")
    return "\n\n".join(parts) if parts else "(no assessments available)"


def _rebuttal_summary(state: APEXState) -> str:
    parts = []
    for role, label in [
        ("cso", "Scientific Advisor"),
        ("cto", "Technical Advisor"),
        ("cmo", "Clinical Advisor"),
        ("cbo", "Commercial Advisor"),
    ]:
        text = state.get(f"{role}_rebuttal", "").strip()
        if text:
            parts.append(f"--- {label} ---\n{text}")
    return "\n\n".join(parts) if parts else "(no rebuttals available)"


def _ddp_plans_summary(state: APEXState) -> str:
    """Concatenate the 5 completed plan sections for the Director's synthesis."""
    parts = []
    for role, label in [
        ("cso", "TARGET VALIDATION STRATEGY — Scientific Advisor"),
        ("cto", "MODALITY & MANUFACTURING STRATEGY — Technical Advisor"),
        ("cmo", "CLINICAL DEVELOPMENT STRATEGY — Clinical Advisor"),
        ("cbo", "COMMERCIAL & STRATEGIC ASSESSMENT — Commercial Advisor"),
        ("ip_attorney", "IP STRATEGY & PROSECUTION PLAN — IP Strategy Advisor"),
    ]:
        text = state.get(f"{role}_plan", "").strip()
        if text:
            parts.append(f"=== {label} ===\n{text}")
    return "\n\n".join(parts) if parts else "(no DDP sections available)"


# ---------------------------------------------------------------------------
# Planning node factory — mirrors _make_assess_node from executives.py
# ---------------------------------------------------------------------------


def _make_plan_node(role: str):
    """Create a DDP planning node for the given executive role.

    system_prompt = EXECUTIVE_PROMPTS[role]      — existing persona system prompt
    user_prompt   = DDP_PLANNING_PROMPTS[role]   — DDP planning user prompt

    Two-pass tool use: same pattern as assessment nodes.
    RAG context injected the same way.
    """
    field_name = f"{role}_plan"
    system_prompt = EXECUTIVE_PROMPTS[role]
    role_label = EXECUTIVE_ROLES[role]

    async def plan_node(state: APEXState) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        gene, indication = _gene_and_indication(state)

        # Retrieve RAG context (graceful fallback if unavailable)
        rag_context = ""
        if _retrieve_context is not None:
            try:
                rag_context = _retrieve_context(role, state["query"], k=5)
            except Exception:
                pass

        # Build user prompt from DDP planning template
        user_prompt = DDP_PLANNING_PROMPTS[role].format(
            gene=gene,
            indication=indication,
            verdict_summary=state.get("portfolio_verdict", "")[:800],
            assessment_summary=_assessment_summary(state),
            rebuttal_summary=_rebuttal_summary(state),
            ceo_feedback=state.get("ceo_feedback", "") or "None provided.",
        )

        # Inject RAG context before the planning prompt (same pattern as assess_node)
        if rag_context:
            user_prompt = f"{rag_context}\n\n---\n\n{user_prompt}"

        # First pass — generate initial DDP section
        section_text, call_cost = await _call_llm(system_prompt, user_prompt, role=role)
        node_cost = call_cost

        # Two-pass: check for TOOL_REQUESTS and execute if found (role-filtered)
        tool_requests = parse_tool_requests(section_text)
        tools_used = []
        if tool_requests:
            try:
                tool_results = await execute_tool_requests(tool_requests, role=role)
                if tool_results:
                    tools_used = [t[0] for t in tool_requests]
                    followup_prompt = TOOL_FOLLOWUP_TEMPLATE.format(
                        query=state["query"],
                        tool_results=tool_results,
                    )
                    section_text, followup_cost = await _call_llm(
                        system_prompt, followup_prompt, role=role
                    )
                    node_cost += followup_cost
            except Exception:
                pass  # Tool failure should not block planning

        log_entry: dict[str, Any] = {
            "node": f"{role}_plan",
            "role": role_label,
            "status": "complete",
            "cost_usd": node_cost,
            "timestamp": ts,
        }
        if tools_used:
            log_entry["tools_used"] = tools_used

        return {
            field_name: section_text,
            "activity_log": [log_entry],
        }

    plan_node.__name__ = f"{role}_plan_node"
    plan_node.__qualname__ = f"{role}_plan_node"
    return plan_node


# ---------------------------------------------------------------------------
# Create all 4 executive planning nodes
# ---------------------------------------------------------------------------

cso_plan_node = _make_plan_node("cso")
cto_plan_node = _make_plan_node("cto")
cmo_plan_node = _make_plan_node("cmo")
cbo_plan_node = _make_plan_node("cbo")
ip_attorney_plan_node = _make_plan_node("ip_attorney")

# ---------------------------------------------------------------------------
# Director synthesis node — reads all 4 plan sections, writes integrated DDP
#
# system_prompt = DIRECTOR_SYSTEM_PROMPT        (existing director persona)
# user_prompt   = DIRECTOR_SYNTHESIS_PROMPT     (DDP synthesis user prompt)
# ---------------------------------------------------------------------------

_DIRECTOR_MODEL = ROLE_MODELS["portfolio_director"]
_director_llm = ChatAnthropic(
    model=_DIRECTOR_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=ANTHROPIC_API_KEY,
    max_tokens=4000,  # Integrated timeline + budget + risks is the longest output
)


async def director_synthesis_node(state: APEXState) -> dict[str, Any]:
    """Portfolio Director synthesises all 4 DDP plan sections into an integrated plan."""
    ts = datetime.now(timezone.utc).isoformat()
    gene, indication = _gene_and_indication(state)

    # The 4 completed plan sections become the "assessment_summary" for the Director
    user_prompt = DDP_PLANNING_PROMPTS["portfolio_director"].format(
        gene=gene,
        indication=indication,
        verdict_summary=state.get("portfolio_verdict", "")[:800],
        assessment_summary=_ddp_plans_summary(state),
        rebuttal_summary="(Not applicable — this is the DDP synthesis stage.)",
        ceo_feedback=state.get("ceo_feedback", "") or "None provided.",
    )

    messages = [
        SystemMessage(content=DIRECTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    call_cost = 0.0
    async with llm_semaphore:
        try:
            response = await asyncio.wait_for(
                _director_llm.ainvoke(messages),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            synthesis_text = response.content
            usage = getattr(response, "usage_metadata", None) or {}
            call_cost = estimate_cost(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                model=_DIRECTOR_MODEL,
            )
        except asyncio.TimeoutError:
            synthesis_text = "[DDP Director timed out — synthesis incomplete]"
        except Exception as e:
            synthesis_text = f"[DDP Director error: {str(e)[:200]}]"

    return {
        "director_synthesis": synthesis_text,
        "ddp_status": "complete",
        "activity_log": [
            {
                "node": "director_synthesis",
                "status": "complete",
                "cost_usd": call_cost,
                "timestamp": ts,
            }
        ],
    }

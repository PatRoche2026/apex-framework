"""Executive agent nodes — assessment and rebuttal for CSO, CTO, CMO, CBO."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents import (
    ANTHROPIC_API_KEY,
    LLM_MODEL,
    LLM_MODEL_STRONG,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    ROLE_MODELS,
    estimate_cost,
    llm_semaphore,
)
from agents.prompts import (
    ASSESSMENT_USER_TEMPLATE,
    CEO_FEEDBACK_SECTION,
    EXECUTIVE_PROMPTS,
    EXECUTIVE_ROLES,
    REBUTTAL_SYSTEM_TEMPLATE,
    REBUTTAL_USER_TEMPLATE,
    ROLE_TOOL_DESCRIPTIONS,
    SHARPEN_INSTRUCTION,
    TOOL_FOLLOWUP_TEMPLATE,
)
from agents.state import APEXState
from agents.tools import parse_tool_requests, execute_tool_requests

# RAG is optional — gracefully degrade if chromadb/sentence-transformers not installed
# Uses CRAG (Corrective RAG) with quality grading + PubMed escalation
try:
    from rag.crag import crag_retrieve_context as _crag_retrieve_context
except ImportError:
    _crag_retrieve_context = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# LLM instance cache — one per model to avoid re-creating
# ---------------------------------------------------------------------------

_llm_cache: dict[str, ChatAnthropic] = {}


def _get_llm(model: str) -> ChatAnthropic:
    """Legacy Anthropic-only helper. Retained for backward-compat callers."""
    if model not in _llm_cache:
        _llm_cache[model] = ChatAnthropic(
            model=model,
            temperature=LLM_TEMPERATURE,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=2048,
        )
    return _llm_cache[model]

# ---------------------------------------------------------------------------
# Score parsing — regex extraction from structured LLM output
# ---------------------------------------------------------------------------

_SCORE_DIMENSIONS = [
    "SCIENTIFIC_VALIDITY",
    "TECHNICAL_FEASIBILITY",
    "CLINICAL_PATH",
    "COMMERCIAL_POTENTIAL",
]


def parse_scores(text: str) -> dict[str, int]:
    """Extract X/10 scores from executive output text."""
    scores: dict[str, int] = {}
    for dim in _SCORE_DIMENSIONS:
        match = re.search(rf"{dim}:\s*(\d+)\s*/\s*10", text)
        if match:
            scores[dim.lower()] = min(int(match.group(1)), 10)
    return scores


def parse_confidence(text: str) -> int:
    """Extract confidence percentage from executive output text."""
    match = re.search(r"CONFIDENCE:\s*(\d+)", text)
    return min(int(match.group(1)), 100) if match else 50


def parse_verdict(text: str) -> str:
    """Extract GO / CONDITIONAL GO / NO-GO from text.

    Scans from the BOTTOM of the text because the final recommendation
    is always at the end — earlier text discusses other agents' positions.
    """
    upper = text.upper()

    # 1. Scan lines from bottom — first verdict keyword found is the final one
    lines = upper.split("\n")
    for line in reversed(lines):
        line = line.strip()
        if "NO-GO" in line or "NO GO" in line or "NOGO" in line:
            if "CONDITIONAL" not in line:
                return "NO-GO"
        if "CONDITIONAL GO" in line or "CONDITIONAL-GO" in line:
            return "CONDITIONAL GO"

    # 2. Fallback: scan the last 500 chars for the final verdict
    tail = upper[-500:]
    if "NO-GO" in tail or "NO GO" in tail:
        return "NO-GO"
    if "CONDITIONAL GO" in tail:
        return "CONDITIONAL GO"
    if "VERDICT: GO" in tail or "VERDICT:\nGO" in tail:
        return "GO"

    # 3. Last resort: count occurrences across entire text
    no_go_count = upper.count("NO-GO") + upper.count("NO GO") + upper.count("NOGO")
    go_count = upper.count(" GO ") + upper.count(" GO,") + upper.count(" GO.")
    if no_go_count > 0:
        return "NO-GO"
    if "CONDITIONAL" in upper and go_count > 0:
        return "CONDITIONAL GO"
    if go_count > 0:
        return "GO"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Safe LLM call with semaphore + timeout
# ---------------------------------------------------------------------------


async def _call_llm(
    system_prompt: str,
    user_prompt: str,
    role: str = "cso",
    provider: str = "anthropic",
) -> tuple[str, float]:
    """Call LLM with rate limit semaphore and timeout protection.

    Provider-agnostic: routes through agents.llm_router so the Truth Tribunal
    can run the same node code against anthropic, openai, or google backends.
    Tier (synthesis vs specialist) is determined per-role inside the router.

    Returns:
        (response_text, estimated_cost_usd)
    """
    from agents.llm_router import get_llm as router_get_llm, estimate_cost_from_response
    llm = router_get_llm(role, provider=provider, temperature=LLM_TEMPERATURE, max_tokens=2048)
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
            cost = estimate_cost_from_response(provider, role, response)
            return response.content, cost
        except asyncio.TimeoutError:
            return "[Agent timed out — proceeding with available assessments]", 0.0
        except Exception as e:
            err_str = str(e).lower()
            if "credit balance" in err_str or "billing" in err_str or "insufficient" in err_str:
                return "[PROVIDER_CREDIT_ERROR] API credits need renewal", 0.0
            elif "rate limit" in err_str or "rate_limit" in err_str or "429" in err_str:
                return "[PROVIDER_RATE_LIMIT] Too many requests", 0.0
            elif (
                "authentication_error" in err_str
                or "invalid api key" in err_str
                or "invalid x-api-key" in err_str
                or "401 unauthorized" in err_str
                or "status_code=401" in err_str
            ):
                return "[PROVIDER_AUTH_ERROR] API key invalid", 0.0
            print(f"[Agent {role}] non-fatal error: {type(e).__name__}: {str(e)[:300]}")
            return f"[Agent {role} transient error: {type(e).__name__}]", 0.0


# ---------------------------------------------------------------------------
# Assessment node factory — creates a node function for each executive
# ---------------------------------------------------------------------------


def _make_assess_node(role: str):
    """Create an assessment node function for the given executive role.

    Args:
        role: One of 'cso', 'cto', 'cmo', 'cbo'
    """
    field_name = f"{role}_assessment"
    system_prompt = EXECUTIVE_PROMPTS[role]
    role_label = EXECUTIVE_ROLES[role]

    async def assess_node(state: APEXState) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        provider = state.get("provider") or "anthropic"

        # Retrieve RAG context via CRAG (quality-graded + PubMed escalation)
        rag_context = ""
        if _crag_retrieve_context is not None:
            try:
                rag_context = await _crag_retrieve_context(role, state["query"], k=5)
            except Exception:
                pass  # RAG failure should not block assessment

        # CEO feedback section (Feature 4)
        ceo_feedback_section = ""
        ceo_feedback = state.get("ceo_feedback", "")
        if ceo_feedback:
            ceo_feedback_section = CEO_FEEDBACK_SECTION.format(ceo_feedback=ceo_feedback)

        # Role-specific tool descriptions (Feature 5)
        role_tools_section = ROLE_TOOL_DESCRIPTIONS.get(role, "")

        # Build user prompt with optional RAG context injected
        user_prompt = ASSESSMENT_USER_TEMPLATE.format(
            query=state["query"],
            scout_data=state["scout_data"],
            ceo_feedback_section=ceo_feedback_section,
            role_tools_section=role_tools_section,
        )
        if rag_context:
            user_prompt = f"{rag_context}\n\n---\n\n{user_prompt}"

        assessment, call_cost = await _call_llm(system_prompt, user_prompt, role=role, provider=provider)
        node_cost = call_cost

        # Two-pass: check for TOOL_REQUESTS and execute if found (role-filtered)
        tool_requests = parse_tool_requests(assessment)
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
                    assessment, followup_cost = await _call_llm(system_prompt, followup_prompt, role=role, provider=provider)
                    node_cost += followup_cost
            except Exception:
                pass  # Tool failure should not block assessment

        scores = parse_scores(assessment)

        log_entry = {
            "node": f"{role}_assess",
            "role": role_label,
            "status": "complete",
            "scores": scores,
            "verdict": parse_verdict(assessment),
            "confidence": parse_confidence(assessment),
            "cost_usd": node_cost,
            "timestamp": ts,
        }
        if tools_used:
            log_entry["tools_used"] = tools_used

        return {
            field_name: assessment,
            "executive_scores": {role: scores},
            "activity_log": [log_entry],
        }

    assess_node.__name__ = f"{role}_assess_node"
    assess_node.__qualname__ = f"{role}_assess_node"
    return assess_node


# ---------------------------------------------------------------------------
# Create all 4 assessment nodes
# ---------------------------------------------------------------------------

cso_assess_node = _make_assess_node("cso")
cto_assess_node = _make_assess_node("cto")
cmo_assess_node = _make_assess_node("cmo")
cbo_assess_node = _make_assess_node("cbo")

# ---------------------------------------------------------------------------
# Rebuttal node factory — creates a rebuttal node for each executive
# ---------------------------------------------------------------------------


def _make_rebuttal_node(role: str):
    """Create a rebuttal node function for the given executive role.

    Args:
        role: One of 'cso', 'cto', 'cmo', 'cbo'
    """
    field_name = f"{role}_rebuttal"
    role_label = EXECUTIVE_ROLES[role]

    async def rebuttal_node(state: APEXState) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        round_num = state.get("debate_round", 0) + 1
        provider = state.get("provider") or "anthropic"

        # Build system prompt with optional sharpen instruction for round 2+
        sharpen = ""
        if round_num >= 2:
            sharpen = SHARPEN_INSTRUCTION.format(round_num=round_num)

        system_prompt = REBUTTAL_SYSTEM_TEMPLATE.format(
            role=role_label,
            round_num=round_num,
            sharpen_instruction=sharpen,
        )

        # CEO feedback section (Feature 4)
        ceo_feedback_section = ""
        ceo_feedback = state.get("ceo_feedback", "")
        if ceo_feedback:
            ceo_feedback_section = CEO_FEEDBACK_SECTION.format(ceo_feedback=ceo_feedback)

        user_prompt = REBUTTAL_USER_TEMPLATE.format(
            query=state["query"],
            ceo_feedback_section=ceo_feedback_section,
            cso_assessment=state["cso_assessment"],
            cto_assessment=state["cto_assessment"],
            cmo_assessment=state["cmo_assessment"],
            cbo_assessment=state["cbo_assessment"],
            ip_assessment=state.get("ip_assessment", "(IP assessment not available)"),
        )

        rebuttal, call_cost = await _call_llm(system_prompt, user_prompt, role=role, provider=provider)
        scores = parse_scores(rebuttal)

        result: dict = {
            field_name: rebuttal,
            "activity_log": [
                {
                    "node": f"{role}_rebuttal",
                    "role": role_label,
                    "status": "complete",
                    "round": round_num,
                    "scores": scores,
                    "verdict": parse_verdict(rebuttal),
                    "confidence": parse_confidence(rebuttal),
                    "cost_usd": call_cost,
                    "timestamp": ts,
                }
            ],
        }

        # Only update executive_scores if rebuttal contained parseable scores;
        # otherwise the empty dict would overwrite assessment scores via _merge_dicts
        if scores:
            result["executive_scores"] = {role: scores}

        return result

    rebuttal_node.__name__ = f"{role}_rebuttal_node"
    rebuttal_node.__qualname__ = f"{role}_rebuttal_node"
    return rebuttal_node


# ---------------------------------------------------------------------------
# Create all 4 rebuttal nodes
# ---------------------------------------------------------------------------

cso_rebuttal_node = _make_rebuttal_node("cso")
cto_rebuttal_node = _make_rebuttal_node("cto")
cmo_rebuttal_node = _make_rebuttal_node("cmo")
cbo_rebuttal_node = _make_rebuttal_node("cbo")

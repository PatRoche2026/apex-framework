"""Portfolio Director node — synthesizes debate and issues final verdict."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents import (
    ANTHROPIC_API_KEY,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_LONG,
    ROLE_MODELS,
    estimate_cost,
    llm_semaphore,
)
from agents.executives import parse_confidence, parse_scores, parse_verdict
from agents.prompts import CEO_FEEDBACK_SECTION, DIRECTOR_SYSTEM_PROMPT, DIRECTOR_USER_TEMPLATE
from agents.state import APEXState
from config import SCORING_DIMENSIONS

# ---------------------------------------------------------------------------
# LLM instance — Portfolio Director uses the synthesis-tier model for whichever
# provider is selected on the state. The router picks the heavy model (Sonnet /
# GPT-4o / Gemini Pro) automatically based on the "portfolio_director" role.
# ---------------------------------------------------------------------------

_DIRECTOR_MODEL = ROLE_MODELS["portfolio_director"]  # Retained for legacy cost calc

# ---------------------------------------------------------------------------
# Weighted composite score calculation
# ---------------------------------------------------------------------------

# Pulled from config.py — customize SCORING_DIMENSIONS there to re-weight.
SCORE_WEIGHTS = {dim: meta["weight"] for dim, meta in SCORING_DIMENSIONS.items()}


def compute_weighted_score(executive_scores: dict) -> dict:
    """Compute weighted composite from all advisors' final scores.

    Args:
        executive_scores: {role: {dimension: score, ...}, ...}

    Returns:
        {"per_dimension": {dim: avg}, "weighted_total": float,
         "per_executive": {role: {dim: score}}}
    """
    dims = list(SCORE_WEIGHTS.keys())

    # Average each dimension across all advisors who provided it
    dim_averages: dict[str, float] = {}
    for dim in dims:
        values = [
            scores[dim]
            for scores in executive_scores.values()
            if isinstance(scores, dict) and dim in scores
        ]
        dim_averages[dim] = round(sum(values) / len(values), 1) if values else 5.0

    # Weighted total
    weighted_total = sum(dim_averages[dim] * SCORE_WEIGHTS[dim] for dim in dims)

    return {
        "per_dimension": dim_averages,
        "weighted_total": round(weighted_total, 2),
        "per_executive": executive_scores,
    }


# ---------------------------------------------------------------------------
# Portfolio Director node
# ---------------------------------------------------------------------------


async def portfolio_director_node(state: APEXState) -> dict[str, Any]:
    """Portfolio Director: synthesize all assessments + rebuttals, issue verdict."""
    ts = datetime.now(timezone.utc).isoformat()
    provider = state.get("provider") or "anthropic"

    # Reviewer feedback section (human-in-the-loop)
    ceo_feedback_section = ""
    ceo_feedback = state.get("ceo_feedback", "")
    if ceo_feedback:
        ceo_feedback_section = CEO_FEEDBACK_SECTION.format(ceo_feedback=ceo_feedback)

    user_prompt = DIRECTOR_USER_TEMPLATE.format(
        query=state["query"],
        ceo_feedback_section=ceo_feedback_section,
        cso_assessment=state["cso_assessment"],
        cto_assessment=state["cto_assessment"],
        cmo_assessment=state["cmo_assessment"],
        cbo_assessment=state["cbo_assessment"],
        ip_assessment=state.get("ip_assessment", "(IP assessment not available)"),
        cso_rebuttal=state["cso_rebuttal"],
        cto_rebuttal=state["cto_rebuttal"],
        cmo_rebuttal=state["cmo_rebuttal"],
        cbo_rebuttal=state["cbo_rebuttal"],
    )

    messages = [
        SystemMessage(content=DIRECTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    from agents.llm_router import get_llm as router_get_llm, estimate_cost_from_response
    llm = router_get_llm(
        "portfolio_director",
        provider=provider,
        temperature=LLM_TEMPERATURE,
        max_tokens=3000,
    )

    call_cost = 0.0
    async with llm_semaphore:
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT_LONG,
            )
            verdict_text = response.content
            call_cost = estimate_cost_from_response(provider, "portfolio_director", response)
        except asyncio.TimeoutError:
            verdict_text = "[Portfolio Director timed out — defaulting to CONDITIONAL GO]"
        except Exception as e:
            err_str = str(e).lower()
            if "credit balance" in err_str or "billing" in err_str or "insufficient" in err_str:
                verdict_text = "[PROVIDER_CREDIT_ERROR] API credits need renewal"
            elif "rate limit" in err_str or "rate_limit" in err_str or "429" in err_str:
                verdict_text = "[PROVIDER_RATE_LIMIT] Too many requests — please retry"
            elif (
                "authentication_error" in err_str
                or "invalid api key" in err_str
                or "invalid x-api-key" in err_str
                or "401 unauthorized" in err_str
                or "status_code=401" in err_str
            ):
                verdict_text = "[PROVIDER_AUTH_ERROR] API key invalid"
            else:
                print(f"[Portfolio Director] non-fatal error: {type(e).__name__}: {str(e)[:300]}")
                verdict_text = (
                    f"[Portfolio Director transient error: {type(e).__name__}]\n\n"
                    "The director could not synthesize. Defaulting to CONDITIONAL GO "
                    "so the session can complete; please re-run for a full verdict."
                )

    # Parse director's own scores
    director_scores = parse_scores(verdict_text)
    confidence = parse_confidence(verdict_text)
    verdict = parse_verdict(verdict_text)

    # Compute weighted composite from ALL advisors' latest scores
    exec_scores = dict(state.get("executive_scores", {}))
    exec_scores["director"] = director_scores
    composite = compute_weighted_score(exec_scores)

    # Parse weighted total from director output if present (override computed)
    wt_match = re.search(r"WEIGHTED_TOTAL:\s*([\d.]+)\s*/\s*10", verdict_text)
    if wt_match:
        composite["weighted_total"] = float(wt_match.group(1))

    return {
        "portfolio_verdict": verdict_text,
        "confidence_score": confidence,
        "executive_scores": composite,
        "debate_round": state.get("debate_round", 0) + 1,
        "activity_log": [
            {
                "node": "portfolio_director",
                "status": "complete",
                "verdict": verdict,
                "confidence": confidence,
                "weighted_total": composite["weighted_total"],
                "round": state.get("debate_round", 0) + 1,
                "cost_usd": call_cost,
                "timestamp": ts,
            }
        ],
    }

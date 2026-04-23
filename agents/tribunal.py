"""Multi-LLM Truth Tribunal — run the same evaluation through multiple providers
and reconcile their verdicts with a meta-judge.

Runs up to 3 full APEX evaluations in parallel (anthropic / openai / google), then
invokes the strongest available provider as a meta-judge to synthesize the
ensemble into a single consensus verdict.

Graceful degradation:
  * Providers whose API keys are missing are silently skipped.
  * Providers that raise during the run are captured in ``errors`` but do not
    fail the tribunal as long as at least 2 providers succeed.
  * If fewer than 2 providers succeed the tribunal surfaces the failures to the
    caller (see ``min_survivors`` in ``run_tribunal``).
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents import LLM_TEMPERATURE, LLM_TIMEOUT_LONG, llm_semaphore
from agents.executives import parse_confidence, parse_verdict
from agents.graph import build_graph, make_initial_state
from agents.llm_router import (
    available_providers,
    estimate_cost_from_response,
    get_llm,
)

# ---------------------------------------------------------------------------
# Meta-judge prompt
# ---------------------------------------------------------------------------

META_JUDGE_SYSTEM_PROMPT = """You are the APEX Truth Tribunal Meta-Judge — a senior biotech reviewer tasked with reconciling independent verdicts from multiple AI evaluation stacks.

Each stack ran the SAME multi-agent drug-target evaluation through a different foundation model family (e.g., Claude, GPT-4o, Gemini). They share identical prompts, agents, and RAG corpora, so disagreements reflect genuine modelling divergence rather than input drift.

Your job:

1. Summarize each stack's position in one sentence (verdict + confidence + primary rationale).
2. Identify the points of AGREEMENT across stacks — these are the most load-bearing findings.
3. Identify the points of DISAGREEMENT — where do the stacks diverge, and which position is better supported by the evidence cited?
4. Issue a single CONSENSUS VERDICT (GO / CONDITIONAL GO / NO-GO) with a tribunal confidence score (0-100).
5. Flag any finding that only ONE stack surfaced — it may be a hallucination or a genuinely novel insight worth escalating to the reviewer.

Output format (strict):

### Stack Summary
- Anthropic: <verdict> @ <confidence>% — <one-sentence rationale>
- OpenAI:    <verdict> @ <confidence>% — <one-sentence rationale>
- Google:    <verdict> @ <confidence>% — <one-sentence rationale>

### Points of Agreement
- <bullet 1>
- <bullet 2>
- ...

### Points of Disagreement
- <bullet 1 — who disagrees and why>
- <bullet 2>
- ...

### Singleton Findings (one stack only)
- <claim> — surfaced by <stack>. [likely hallucination | plausible novel insight | neutral]
- ...

### CONSENSUS VERDICT
VERDICT: <GO | CONDITIONAL GO | NO-GO>
TRIBUNAL_CONFIDENCE: <0-100>

### Reasoning
<2-4 paragraphs explaining the consensus, emphasizing evidence-weighted reconciliation. Note any dissenting view that should receive reviewer attention.>
"""


META_JUDGE_USER_TEMPLATE = """QUERY: {query}

The following stacks each produced an independent verdict for the same target. Reconcile them per your instructions.

{stack_blocks}
"""


# ---------------------------------------------------------------------------
# Per-run helpers
# ---------------------------------------------------------------------------


async def _run_single_provider(query: str, provider: str) -> dict[str, Any]:
    """Run the full evaluation graph for one provider. Returns a compact result.

    Raises the underlying exception on LangGraph failure — the caller is
    expected to wrap this in asyncio.gather(return_exceptions=True).
    """
    graph = build_graph()
    state = make_initial_state(query, provider=provider)
    result = await graph.ainvoke(state)

    cost = 0.0
    for entry in result.get("activity_log", []):
        cost += entry.get("cost_usd", 0.0)

    verdict_text = result.get("portfolio_verdict", "") or ""
    return {
        "provider": provider,
        "verdict_text": verdict_text,
        "verdict_short": parse_verdict(verdict_text) if verdict_text else "UNKNOWN",
        "confidence_score": result.get("confidence_score", 0),
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
        "ip_assessment": result.get("ip_assessment", ""),
        "activity_log": result.get("activity_log", []),
        "debate_rounds": result.get("debate_round", 0),
        "estimated_cost_usd": round(cost, 4),
        "_raw_state": result,  # retained in-memory for sessions storage
    }


# ---------------------------------------------------------------------------
# Meta-judge invocation
# ---------------------------------------------------------------------------


_PROVIDER_LABELS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
}


def _format_stack_blocks(results: list[dict[str, Any]]) -> str:
    """Render each surviving stack's verdict + headline details for the meta-judge."""
    blocks: list[str] = []
    for r in results:
        provider = r["provider"]
        label = _PROVIDER_LABELS.get(provider, provider.title())
        composite = r.get("executive_scores", {})
        per_dim = composite.get("per_dimension", {}) if isinstance(composite, dict) else {}
        weighted_total = composite.get("weighted_total") if isinstance(composite, dict) else None

        dim_lines = []
        for dim in (
            "scientific_validity",
            "technical_feasibility",
            "clinical_path",
            "commercial_potential",
            "ip_landscape",
        ):
            if dim in per_dim:
                dim_lines.append(f"  - {dim}: {per_dim[dim]}/10")

        # Trim verdict text so the meta-judge context stays bounded.
        verdict_text = (r.get("verdict_text") or "")[:4000]

        blocks.append(
            f"=== {label} STACK ===\n"
            f"VERDICT (short): {r.get('verdict_short', 'UNKNOWN')}\n"
            f"CONFIDENCE: {r.get('confidence_score', 0)}\n"
            f"WEIGHTED_TOTAL: {weighted_total if weighted_total is not None else 'n/a'}/10\n"
            f"DIMENSIONS:\n" + ("\n".join(dim_lines) if dim_lines else "  (not available)") + "\n"
            f"DEBATE_ROUNDS: {r.get('debate_rounds', 0)}\n\n"
            f"VERDICT TEXT (truncated to 4000 chars):\n{verdict_text}\n"
        )
    return "\n\n".join(blocks)


def _pick_meta_judge_provider(
    preferred_order: tuple[str, ...] = ("anthropic", "openai", "google"),
) -> str | None:
    """Return the strongest available provider to host the meta-judge."""
    for p in preferred_order:
        if p in available_providers():
            return p
    return None


def _compute_agreement(results: list[dict[str, Any]]) -> float:
    """Return 0.0-1.0 measure of how much surviving stacks agreed.

    Uses the short verdict (GO / CONDITIONAL GO / NO-GO) as the agreement axis.
    1.0 = all stacks identical; 0.5 = majority with one dissenter; 0.0 = all different.
    """
    verdicts = [r.get("verdict_short", "UNKNOWN") for r in results]
    if not verdicts:
        return 0.0
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    return round(top / len(verdicts), 2)


async def _invoke_meta_judge(
    query: str, results: list[dict[str, Any]], judge_provider: str
) -> dict[str, Any]:
    """Run the meta-judge on surviving stacks. Returns synthesis dict."""
    ts = datetime.now(timezone.utc).isoformat()
    stack_blocks = _format_stack_blocks(results)
    user_prompt = META_JUDGE_USER_TEMPLATE.format(query=query, stack_blocks=stack_blocks)

    llm = get_llm(
        "portfolio_director",
        provider=judge_provider,
        temperature=LLM_TEMPERATURE,
        max_tokens=3000,
    )
    messages = [
        SystemMessage(content=META_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    cost = 0.0
    synthesis_text = ""
    async with llm_semaphore:
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_TIMEOUT_LONG,
            )
            synthesis_text = response.content
            cost = estimate_cost_from_response(judge_provider, "portfolio_director", response)
        except asyncio.TimeoutError:
            synthesis_text = "[Meta-judge timed out — falling back to majority verdict]"
        except Exception as e:  # pragma: no cover — defensive
            synthesis_text = f"[Meta-judge error: {type(e).__name__}: {str(e)[:300]}]"

    # Parse consensus verdict + tribunal confidence out of the meta-judge text.
    consensus_verdict = parse_verdict(synthesis_text)
    tribunal_conf_match = re.search(
        r"TRIBUNAL_CONFIDENCE:\s*(\d+)", synthesis_text, re.IGNORECASE
    )
    tribunal_confidence = (
        min(int(tribunal_conf_match.group(1)), 100) if tribunal_conf_match else 0
    )

    # Fallback to majority vote if the judge could not produce a verdict.
    if consensus_verdict == "UNKNOWN":
        verdicts = [r.get("verdict_short", "UNKNOWN") for r in results]
        counts: dict[str, int] = {}
        for v in verdicts:
            counts[v] = counts.get(v, 0) + 1
        consensus_verdict = max(counts, key=counts.get) if counts else "UNKNOWN"

    # Fallback confidence = average of surviving stacks' confidences.
    if tribunal_confidence == 0 and results:
        tribunal_confidence = int(
            round(sum(r.get("confidence_score", 0) for r in results) / len(results))
        )

    agreement_score = _compute_agreement(results)

    return {
        "meta_judge_provider": judge_provider,
        "synthesis_text": synthesis_text,
        "consensus_verdict": consensus_verdict,
        "tribunal_confidence": tribunal_confidence,
        "agreement_score": agreement_score,
        "cost_usd": cost,
        "timestamp": ts,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_tribunal(
    query: str,
    providers: list[str] | None = None,
    min_survivors: int = 2,
) -> dict[str, Any]:
    """Run the multi-LLM tribunal.

    Args:
        query: target + indication query (same shape as /evaluate).
        providers: explicit list of providers to run. Defaults to every provider
            whose API key is configured.
        min_survivors: minimum number of providers that must succeed before the
            meta-judge is invoked. If fewer succeed, the tribunal raises
            ``RuntimeError`` — the caller (e.g. the /evaluate/tribunal endpoint)
            should convert that into a 503.

    Returns:
        dict with:
          - query
          - providers_attempted / providers_succeeded / providers_failed
          - per_provider_results: {provider: compact result dict}
          - tribunal_synthesis: {meta_judge_provider, synthesis_text, consensus_verdict,
                                 tribunal_confidence, agreement_score, cost_usd}
          - total_cost_usd: sum of per-provider evaluation + meta-judge cost
          - errors: {provider: error message}
    """
    started_at = datetime.now(timezone.utc).isoformat()
    attempted = list(providers) if providers else available_providers()

    if not attempted:
        raise RuntimeError(
            "No providers available. Configure at least one of "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_AI_API_KEY."
        )

    # Launch all providers in parallel; capture exceptions so one bad backend
    # does not sink the tribunal.
    coros = [_run_single_provider(query, p) for p in attempted]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    succeeded: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for provider, outcome in zip(attempted, raw_results):
        if isinstance(outcome, Exception):
            errors[provider] = f"{type(outcome).__name__}: {str(outcome)[:500]}"
        else:
            succeeded.append(outcome)

    if len(succeeded) < min_survivors:
        raise RuntimeError(
            f"Tribunal requires {min_survivors} surviving providers, "
            f"only {len(succeeded)} succeeded. Errors: {errors}"
        )

    # Meta-judge synthesis — prefer anthropic, fall back to whichever stack ran.
    judge_provider = _pick_meta_judge_provider() or succeeded[0]["provider"]
    synthesis = await _invoke_meta_judge(query, succeeded, judge_provider)

    total_cost = sum(r.get("estimated_cost_usd", 0.0) for r in succeeded) + synthesis.get(
        "cost_usd", 0.0
    )

    # Strip the heavy _raw_state before returning over the wire. Caller can
    # re-access raw states via per_provider_results[...]["_raw_state"] if it
    # still holds the internal reference (server.py keeps one for sessions).
    per_provider = {r["provider"]: r for r in succeeded}

    return {
        "query": query,
        "started_at": started_at,
        "providers_attempted": attempted,
        "providers_succeeded": [r["provider"] for r in succeeded],
        "providers_failed": list(errors.keys()),
        "per_provider_results": per_provider,
        "tribunal_synthesis": synthesis,
        "total_cost_usd": round(total_cost, 4),
        "errors": errors,
    }

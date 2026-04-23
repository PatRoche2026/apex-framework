"""APEX agents package — shared configuration and .env loading."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

_local_env = Path(__file__).resolve().parent.parent / ".env"
if _local_env.exists():
    load_dotenv(_local_env)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LENS_API_TOKEN = os.getenv("LENS_API_TOKEN", "")

# ---------------------------------------------------------------------------
# Tiered LLM configuration — Sonnet for critical reasoning, Haiku for bulk
# ---------------------------------------------------------------------------

LLM_MODEL = "claude-sonnet-4-20250514"  # backward compat
LLM_MODEL_STRONG = "claude-sonnet-4-20250514"   # $3/$15 per M tokens
LLM_MODEL_FAST = "claude-haiku-4-5-20251001"    # $1/$5 per M tokens
LLM_TEMPERATURE = 0.3

# Role-based model assignment: CSO + Director get Sonnet, rest get Haiku
ROLE_MODELS: dict[str, str] = {
    "scout": LLM_MODEL_FAST,
    "cso": LLM_MODEL_STRONG,
    "cto": LLM_MODEL_FAST,
    "cmo": LLM_MODEL_FAST,
    "cbo": LLM_MODEL_FAST,
    "ip_attorney": LLM_MODEL_FAST,
    "portfolio_director": LLM_MODEL_STRONG,
}

# ---------------------------------------------------------------------------
# Rate limit protection — asyncio.Semaphore(3) for parallel LLM calls
# ---------------------------------------------------------------------------

llm_semaphore = asyncio.Semaphore(3)
# Base timeout for specialist-tier (Haiku / GPT-4o-mini / Gemini Flash) calls.
# Synthesis-tier calls (CSO + Portfolio Director on Sonnet / GPT-4o / Gemini Pro)
# use the longer timeout — large prompt contexts + 3k-token structured output
# can take 60-120s.
LLM_TIMEOUT_SECONDS = 90
LLM_TIMEOUT_LONG = 180

# ---------------------------------------------------------------------------
# PubMed configuration
# ---------------------------------------------------------------------------

ENTREZ_EMAIL = "patroche@mit.edu"
PUBMED_RATE_LIMIT_DELAY = 0.5  # seconds between NCBI API calls

# ---------------------------------------------------------------------------
# Cost tracking — per-model pricing (per million tokens)
# ---------------------------------------------------------------------------

_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_M, output_per_M)
    LLM_MODEL_STRONG: (3.0, 15.0),   # Sonnet
    LLM_MODEL_FAST: (1.0, 5.0),      # Haiku
}

# Backward-compat defaults (Sonnet pricing)
COST_PER_M_INPUT = 3.0
COST_PER_M_OUTPUT = 15.0


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "") -> float:
    """Estimate USD cost from token counts. Uses model-specific pricing if provided."""
    input_price, output_price = _MODEL_PRICING.get(
        model, (COST_PER_M_INPUT, COST_PER_M_OUTPUT)
    )
    return round(
        (input_tokens * input_price + output_tokens * output_price) / 1_000_000,
        4,
    )

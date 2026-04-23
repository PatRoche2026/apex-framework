"""Multi-provider LLM router.

Provider-agnostic factory + cost accounting for the three backends used by
the Truth Tribunal:

    anthropic   -> Claude Sonnet 4 (synthesis) / Haiku 4.5 (specialist)
    openai      -> GPT-4o (synthesis)          / GPT-4o-mini (specialist)
    google      -> Gemini 1.5 Pro (synthesis)  / Gemini 1.5 Flash (specialist)

Roles are bucketed into a "synthesis" tier (CSO + Portfolio Director — heavy
reasoning) or a "specialist" tier (everyone else). The router returns a
LangChain-compatible chat model whose ``.ainvoke()`` interface is identical
across providers, so downstream node code is provider-blind once the model
has been constructed.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

from agents import (
    ANTHROPIC_API_KEY,
    LLM_TEMPERATURE,
    LLM_MODEL_FAST,
    LLM_MODEL_STRONG,
    estimate_cost as _estimate_cost_anthropic,
)

# ---------------------------------------------------------------------------
# Provider + tier definitions
# ---------------------------------------------------------------------------

Provider = Literal["anthropic", "openai", "google"]
Tier = Literal["synthesis", "specialist"]
VALID_PROVIDERS: tuple[Provider, ...] = ("anthropic", "openai", "google")
DEFAULT_PROVIDER: Provider = "anthropic"

# Which roles get the heavy (synthesis-tier) model.
_SYNTHESIS_ROLES: frozenset[str] = frozenset({"cso", "portfolio_director"})

# Per-provider model map. Synthesis tier = the strongest model that's
# production-stable as of 2026-04; specialist = the fast/cheap variant.
_MODELS: dict[Provider, dict[Tier, str]] = {
    "anthropic": {
        "synthesis":  LLM_MODEL_STRONG,  # claude-sonnet-4-20250514
        "specialist": LLM_MODEL_FAST,    # claude-haiku-4-5-20251001
    },
    "openai": {
        "synthesis":  "gpt-4o-2024-11-20",
        "specialist": "gpt-4o-mini-2024-07-18",
    },
    "google": {
        # Gemini 1.5 was deprecated in 2025 — use 2.x.
        "synthesis":  "gemini-2.5-pro",
        "specialist": "gemini-2.0-flash",
    },
}

# USD per 1,000,000 tokens. (input, output).
# Anthropic values mirror those in agents/__init__._MODEL_PRICING.
# OpenAI: https://openai.com/pricing (GPT-4o, GPT-4o-mini — 2024-11)
# Google: https://ai.google.dev/pricing (Gemini 1.5 — prompts <=128K tier)
_PRICING: dict[Provider, dict[Tier, tuple[float, float]]] = {
    "anthropic": {
        "synthesis":  (3.0, 15.0),
        "specialist": (1.0, 5.0),
    },
    "openai": {
        "synthesis":  (2.50, 10.00),
        "specialist": (0.15,  0.60),
    },
    "google": {
        # Gemini 2.5 Pro: $1.25 input / $10 output (≤200K prompts, Apr 2025)
        # Gemini 2.0 Flash: $0.10 input / $0.40 output
        "synthesis":  (1.25, 10.00),
        "specialist": (0.10,  0.40),
    },
}


# ---------------------------------------------------------------------------
# Tier picker
# ---------------------------------------------------------------------------

def tier_for_role(role: str) -> Tier:
    """Return the tier for a canonical agent role."""
    return "synthesis" if role in _SYNTHESIS_ROLES else "specialist"


def model_for(provider: Provider, role: str) -> str:
    """Return the concrete model id this provider/role combination uses."""
    return _MODELS[provider][tier_for_role(role)]


# ---------------------------------------------------------------------------
# Key + availability checks
# ---------------------------------------------------------------------------

def _key_for(provider: Provider) -> Optional[str]:
    if provider == "anthropic":
        return ANTHROPIC_API_KEY or None
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY") or None
    if provider == "google":
        # Accept either variable name (users tend to write one or the other)
        return os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
    return None


def provider_available(provider: Provider) -> bool:
    """Return True iff the provider's API key is set."""
    return bool(_key_for(provider))


def available_providers() -> list[Provider]:
    """Return the list of providers that currently have API keys configured."""
    return [p for p in VALID_PROVIDERS if provider_available(p)]


# ---------------------------------------------------------------------------
# LLM factory — returns a LangChain-compatible chat model
# ---------------------------------------------------------------------------

_llm_cache: dict[tuple[Provider, str, float, int], Any] = {}


def get_llm(
    role: str,
    provider: Provider = DEFAULT_PROVIDER,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = 3000,
) -> Any:
    """Return a LangChain chat model for the given role + provider.

    Raises RuntimeError if the provider's API key is missing. Raises
    ImportError if the optional langchain package for the provider isn't
    installed. The returned object exposes the standard ``.ainvoke(messages)``
    interface used by every existing APEX node.
    """
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {VALID_PROVIDERS}")

    api_key = _key_for(provider)
    if not api_key:
        env_var = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_AI_API_KEY",
        }[provider]
        raise RuntimeError(
            f"Provider '{provider}' requested but {env_var} is not set. "
            f"Add the key to .env or skip this provider in the tribunal."
        )

    model = model_for(provider, role)
    cache_key = (provider, model, temperature, max_tokens)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # ChatGoogleGenerativeAI uses max_output_tokens (not max_tokens).
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            max_output_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    _llm_cache[cache_key] = llm
    return llm


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(
    provider: Provider,
    role: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return USD cost for a single LLM call."""
    tier = tier_for_role(role)
    input_price, output_price = _PRICING[provider][tier]
    return round(
        (input_tokens * input_price + output_tokens * output_price) / 1_000_000,
        4,
    )


def estimate_cost_from_usage(
    provider: Provider,
    role: str,
    usage: dict[str, Any] | None,
) -> float:
    """Convenience wrapper — pulls input_tokens/output_tokens from the
    LangChain usage_metadata dict (shape differs slightly across providers).
    """
    usage = usage or {}
    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("candidates_token_count")
        or 0
    )
    return estimate_cost(provider, role, input_tokens, output_tokens)


def estimate_cost_from_response(
    provider: Provider,
    role: str,
    response: Any,
) -> float:
    """Robust cost estimator that inspects multiple usage-metadata shapes.

    Different LangChain chat model adapters surface token usage in different
    places:
      - Anthropic / OpenAI:  ``response.usage_metadata`` (standardized)
      - Google (langchain-google-genai): usage sometimes lands on
        ``response.response_metadata['usage_metadata']`` and the keys can be
        ``prompt_token_count`` / ``candidates_token_count``.
      - Some provider wrappers put raw token counts in
        ``response.response_metadata['token_usage']``.

    This helper tries all of them and returns 0.0 if none are populated.
    """
    if response is None:
        return 0.0

    # Path 1: the standardized usage_metadata attribute.
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return estimate_cost_from_usage(provider, role, usage)

    # Path 2: response_metadata dict (google, some openai variants).
    meta = getattr(response, "response_metadata", None) or {}
    for key in ("usage_metadata", "token_usage", "usage"):
        candidate = meta.get(key)
        if isinstance(candidate, dict) and candidate:
            return estimate_cost_from_usage(provider, role, candidate)

    # Path 3: raw keys directly on response_metadata (google sometimes).
    if meta:
        return estimate_cost_from_usage(provider, role, meta)

    return 0.0


# ---------------------------------------------------------------------------
# Convenience: pricing table for the frontend
# ---------------------------------------------------------------------------

def pricing_table() -> dict[str, Any]:
    """Return a JSON-safe pricing table for the frontend / /metrics endpoint."""
    return {
        provider: {
            tier: {
                "model": _MODELS[provider][tier],
                "input_per_million_usd": _PRICING[provider][tier][0],
                "output_per_million_usd": _PRICING[provider][tier][1],
            }
            for tier in ("synthesis", "specialist")
        }
        for provider in VALID_PROVIDERS
    }

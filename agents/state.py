"""APEXState — shared state schema for the multi-agent debate pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Reducer that merges dicts (right overwrites left on key conflict).

    Used for executive_scores so parallel nodes can each write their own
    role's scores without overwriting other roles' scores.
    """
    merged = dict(left)
    merged.update(right)
    return merged


class APEXState(TypedDict):
    """Full state flowing through the APEX LangGraph pipeline."""

    # --- Input ---
    query: str

    # --- Scout stage ---
    scout_data: str                                          # Prose summary of PubMed findings
    scout_sources: list[dict]                                # [{pmid, title, journal, year, abstract_snippet}]

    # --- Assessment stage (parallel) ---
    cso_assessment: str                                      # Scientific Advisor assessment
    cto_assessment: str                                      # Technical Advisor assessment
    cmo_assessment: str                                      # Clinical Advisor assessment
    cbo_assessment: str                                      # Commercial Advisor assessment

    # --- IP Strategy stage (after assessments, before debates) ---
    ip_assessment: str                                       # IP Strategy Advisor assessment

    # --- Debate / Rebuttal stage (parallel) ---
    cso_rebuttal: str
    cto_rebuttal: str
    cmo_rebuttal: str
    cbo_rebuttal: str

    # --- Portfolio Director ---
    portfolio_verdict: str
    confidence_score: int                                    # 0-100, parsed from director output

    # MUST use Annotated + _merge_dicts so parallel nodes merge scores instead of overwriting
    executive_scores: Annotated[dict, _merge_dicts]          # {cso: {scientific_validity: X, ...}, ...}

    # --- User Feedback (Human-in-the-Loop) ---
    ceo_feedback: str                                        # Latest reviewer feedback text
    ceo_feedback_history: Annotated[list[dict], operator.add]  # All feedback entries
    evaluation_round: int                                    # 0=first pass, 1=re-eval after feedback

    # --- Control ---
    debate_round: int                                        # Current round (for conditional loop)

    # --- Audit trail ---
    # MUST use Annotated + operator.add so parallel nodes merge logs instead of overwriting
    activity_log: Annotated[list[dict], operator.add]

    # --- Development Plan (DDP) — populated after reviewer accepts a GO verdict ---
    gene: str                                                # Extracted from query (e.g. "MMP13")
    indication: str                                          # Extracted from query (e.g. "Knee Osteoarthritis")
    cso_ddp_section: str                                     # Scientific Advisor — Target Validation Strategy
    cto_ddp_section: str                                     # Technical Advisor — Modality & Manufacturing Strategy
    cmo_ddp_section: str                                     # Clinical Advisor — Clinical Development Strategy
    cbo_ddp_section: str                                     # Commercial Advisor — Commercial & Strategic Assessment
    ddp_synthesis: str                                       # Portfolio Director — Executive Summary

    # --- Planning pipeline (compiled_planning_graph in graph.py) ---
    planning_triggered: bool                                 # True once reviewer clicks "Accept" on a GO verdict
    cso_plan: str                                            # Scientific Advisor plan
    cto_plan: str                                            # Technical Advisor plan
    cmo_plan: str                                            # Clinical Advisor plan
    cbo_plan: str                                            # Commercial Advisor plan
    ip_attorney_plan: str                                    # IP Strategy Advisor plan
    director_synthesis: str                                  # Portfolio Director — Executive Summary & Integrated Timeline
    ddp_status: str                                          # "pending" | "in_progress" | "complete" | "skipped"

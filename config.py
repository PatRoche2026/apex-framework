"""APEX framework configuration.

Single source of truth for tunable parameters. Customize values here to adapt
APEX to a different domain, scoring strategy, or model backend.
"""

# ---------------------------------------------------------------------------
# Scoring dimensions (must sum to 1.0)
# ---------------------------------------------------------------------------
# The default configuration gives each dimension equal weight (0.20). Replace
# these values to express domain-specific priorities — e.g., if IP posture is
# more important than market size in your use case, shift weight from
# commercial_potential to ip_landscape.

SCORING_DIMENSIONS = {
    "scientific_validity":   {"weight": 0.20, "description": "Evidence strength and biological plausibility"},
    "technical_feasibility": {"weight": 0.20, "description": "Druggability, modality selection, manufacturing"},
    "clinical_path":         {"weight": 0.20, "description": "Regulatory pathway, trial design, endpoints"},
    "commercial_potential":  {"weight": 0.20, "description": "Market size, competitive landscape, deal comparables"},
    "ip_landscape":          {"weight": 0.20, "description": "Patent freedom-to-operate and prosecution strategy"},
}

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
# SYNTHESIS_MODEL is used for the Portfolio Director and any role that needs
# the strongest reasoning capability. SPECIALIST_MODEL handles bulk assessment
# and rebuttal work at lower cost.

SYNTHESIS_MODEL  = "claude-sonnet-4-20250514"
SPECIALIST_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Debate protocol
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD       = 60    # below this, re-run a debate round
MAX_DEBATE_ROUNDS          = 2     # hard upper bound on rebuttal cycles
MAX_CONCURRENT_LLM_CALLS   = 3     # asyncio.Semaphore limit
LLM_CALL_TIMEOUT_SECONDS   = 60    # per-call timeout

# ---------------------------------------------------------------------------
# RAG configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL        = "voyage-3-lite"
CHUNK_SIZE             = 1000
CHUNK_OVERLAP          = 200
CRAG_QUALITY_THRESHOLD = 0.7       # below this, CRAG escalates to live search

# ---------------------------------------------------------------------------
# Demo targets
# ---------------------------------------------------------------------------
# Five well-studied knee osteoarthritis (KOA) genes that are suitable for
# out-of-box demos. All information below is publicly available (PubMed).
# Replace with your own domain-relevant targets as needed.

DEMO_TARGETS = [
    {
        "gene": "MMP13",
        "indication": "Knee Osteoarthritis",
        "description": "Matrix metalloproteinase 13 — the dominant collagenase driving cartilage breakdown in OA.",
    },
    {
        "gene": "ADAMTS5",
        "indication": "Knee Osteoarthritis",
        "description": "Aggrecanase that degrades the cartilage matrix; validated by knockout mouse studies.",
    },
    {
        "gene": "IL6",
        "indication": "Knee Osteoarthritis",
        "description": "Pro-inflammatory cytokine driving synovitis and structural OA progression.",
    },
    {
        "gene": "GDF5",
        "indication": "Knee Osteoarthritis",
        "description": "Growth and differentiation factor 5 — strong GWAS association with OA risk (rs143383).",
    },
    {
        "gene": "FGF18",
        "indication": "Knee Osteoarthritis",
        "description": "Anabolic cartilage growth factor; sprifermin has advanced to Phase III trials.",
    },
]

# ---------------------------------------------------------------------------
# ChromaDB collection names (one per role, plus a shared cross-role pool)
# ---------------------------------------------------------------------------
COLLECTION_NAMES = {
    "scientific": "apex_scientific",
    "technical":  "apex_technical",
    "clinical":   "apex_clinical",
    "commercial": "apex_commercial",
    "ip":         "apex_ip",
    "shared":     "apex_shared",
}

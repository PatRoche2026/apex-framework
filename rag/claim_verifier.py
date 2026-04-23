"""Claim Verification Module — post-session PubMed fact-checking for APEX agent outputs.

Parses [SUPPORTED:], [UNSUPPORTED:], [UNCERTAIN:] tags from agent outputs,
searches PubMed for each claim, and uses Claude Haiku to judge whether
retrieved abstracts support or contradict each claim.

Output: structured verification report for DDP appendix.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

from agents import ANTHROPIC_API_KEY, LLM_MODEL_FAST, PUBMED_RATE_LIMIT_DELAY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerification:
    """Result of verifying a single factual claim."""
    claim: str
    source_agent: str
    source_tag: str  # SUPPORTED, UNSUPPORTED, or UNCERTAIN
    verdict: str  # Verified, Unverified, or Contradicted
    pmids: list[str] = field(default_factory=list)
    justification: str = ""


@dataclass
class VerificationReport:
    """Full verification report across all agents."""
    claims: list[ClaimVerification] = field(default_factory=list)
    total_verified: int = 0
    total_unverified: int = 0
    total_contradicted: int = 0

    def to_markdown(self) -> str:
        """Format as markdown for DDP appendix."""
        lines = [
            "## Appendix: Claim Verification Report",
            "",
            f"**Total claims checked:** {len(self.claims)}",
            f"**Verified:** {self.total_verified} | "
            f"**Unverified:** {self.total_unverified} | "
            f"**Contradicted:** {self.total_contradicted}",
            "",
        ]

        if not self.claims:
            lines.append("No factual claims were extracted for verification.")
            return "\n".join(lines)

        lines.append("| # | Agent | Claim | Verdict | PMIDs | Justification |")
        lines.append("|---|-------|-------|---------|-------|---------------|")

        for i, c in enumerate(self.claims, 1):
            pmid_str = ", ".join(c.pmids) if c.pmids else "—"
            verdict_icon = {
                "Verified": "Verified",
                "Unverified": "Unverified",
                "Contradicted": "**CONTRADICTED**",
            }.get(c.verdict, c.verdict)
            # Escape pipes in claim text for markdown table
            claim_text = c.claim.replace("|", "/")
            justification = c.justification.replace("|", "/")
            lines.append(
                f"| {i} | {c.source_agent} | {claim_text} | "
                f"{verdict_icon} | {pmid_str} | {justification} |"
            )

        # Highlight contradictions
        contradicted = [c for c in self.claims if c.verdict == "Contradicted"]
        if contradicted:
            lines.append("")
            lines.append("### Contradicted Claims Requiring Review")
            for c in contradicted:
                lines.append(
                    f"- **{c.source_agent}:** \"{c.claim}\" — {c.justification} "
                    f"(PMIDs: {', '.join(c.pmids)})"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claim extraction — parse [SUPPORTED:], [UNSUPPORTED:], [UNCERTAIN:] tags
# ---------------------------------------------------------------------------

# Pattern matches [TAG: claim1, claim2, ...] across one or more lines
_TAG_PATTERN = re.compile(
    r"\[(SUPPORTED|UNSUPPORTED|UNCERTAIN):\s*(.+?)\]",
    re.IGNORECASE | re.DOTALL,
)

# Map agent role names to display names
_AGENT_NAMES = {
    "cso": "Scientific Advisor",
    "cto": "Technical Advisor",
    "cmo": "Clinical Advisor",
    "cbo": "Commercial Advisor",
}


def extract_claims(agent_output: str, agent_role: str) -> list[dict[str, str]]:
    """Parse [SUPPORTED:], [UNSUPPORTED:], [UNCERTAIN:] tags from agent output.

    Returns list of {"claim": str, "tag": str, "agent": str}.
    """
    claims: list[dict[str, str]] = []
    agent_name = _AGENT_NAMES.get(agent_role, agent_role)

    for match in _TAG_PATTERN.finditer(agent_output):
        tag = match.group(1).upper()
        raw_claims = match.group(2).strip()

        # Split on commas, but respect claims that may contain commas in context
        # Use a simple heuristic: split on ", " followed by a lowercase or uppercase letter
        # that starts a new claim
        individual_claims = re.split(r",\s*(?=[A-Z])", raw_claims)

        for claim_text in individual_claims:
            claim_text = claim_text.strip().rstrip(",").strip()
            if claim_text and claim_text.lower() != "none":
                claims.append({
                    "claim": claim_text,
                    "tag": tag,
                    "agent": agent_name,
                })

    return claims


# ---------------------------------------------------------------------------
# PubMed search for claim verification
# ---------------------------------------------------------------------------

async def _search_pubmed_for_claim(claim: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search PubMed for papers relevant to a factual claim.

    Reuses the existing search_pubmed from agents.scout (sync), wrapped
    in asyncio.to_thread for async compatibility.
    """
    from agents.scout import search_pubmed

    # Condense the claim to a keyword query — take first 200 chars
    # PubMed handles natural language queries well
    query = claim[:200]

    try:
        results = await asyncio.to_thread(search_pubmed, query, max_results)
        return results
    except Exception as exc:
        logger.warning("PubMed search failed for claim '%s': %s", claim[:80], exc)
        return []


# ---------------------------------------------------------------------------
# Haiku LLM judgment — SUPPORTS / CONTRADICTS / INSUFFICIENT
# ---------------------------------------------------------------------------

_JUDGMENT_PROMPT = """\
You are a biomedical fact-checker. A biotech executive made the following claim \
during a drug target evaluation:

CLAIM: {claim}

Below are PubMed abstracts retrieved for this claim. Determine whether the \
published literature SUPPORTS, CONTRADICTS, or provides INSUFFICIENT evidence \
for this claim.

{abstracts}

Respond with EXACTLY this format (no other text):
VERDICT: SUPPORTS | CONTRADICTS | INSUFFICIENT
JUSTIFICATION: [One sentence explaining your judgment, citing specific PMIDs]"""


async def _judge_claim_against_abstracts(
    claim: str, papers: list[dict[str, Any]]
) -> tuple[str, str]:
    """Use Claude Haiku to judge whether PubMed abstracts support or contradict a claim.

    Returns (verdict, justification) where verdict is one of:
    SUPPORTS, CONTRADICTS, INSUFFICIENT.
    """
    # Format abstracts for the prompt
    abstract_lines = []
    for p in papers:
        abstract_lines.append(
            f"PMID {p['pmid']}: {p['title']}\n"
            f"Journal: {p['journal']} ({p['year']})\n"
            f"Abstract: {p['abstract'][:500]}"
        )
    abstracts_text = "\n\n".join(abstract_lines)

    prompt = _JUDGMENT_PROMPT.format(claim=claim, abstracts=abstracts_text)

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = await asyncio.to_thread(
            client.messages.create,
            model=LLM_MODEL_FAST,
            max_tokens=150,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Parse verdict
        verdict_match = re.search(r"VERDICT:\s*(SUPPORTS|CONTRADICTS|INSUFFICIENT)", text, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() if verdict_match else "INSUFFICIENT"

        # Parse justification
        just_match = re.search(r"JUSTIFICATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        justification = just_match.group(1).strip() if just_match else "No justification provided."

        return verdict, justification

    except Exception as exc:
        logger.warning("Haiku judgment failed for claim '%s': %s", claim[:80], exc)
        return "INSUFFICIENT", f"LLM judgment failed: {str(exc)[:100]}"


# ---------------------------------------------------------------------------
# Main verification pipeline
# ---------------------------------------------------------------------------

async def verify_claims(
    agent_outputs: dict[str, str],
    max_pubmed_results: int = 3,
) -> VerificationReport:
    """Run the full claim verification pipeline.

    Args:
        agent_outputs: {role: raw_output_string} for each executive agent.
            e.g. {"cso": "...", "cto": "...", "cmo": "...", "cbo": "..."}
        max_pubmed_results: Max PubMed papers to retrieve per claim.

    Returns:
        VerificationReport with all claims checked and scored.
    """
    # Step 1: Extract all claims from all agent outputs
    all_claims: list[dict[str, str]] = []
    for role, output in agent_outputs.items():
        if output:
            all_claims.extend(extract_claims(output, role))

    if not all_claims:
        logger.info("No claims extracted from agent outputs — skipping verification.")
        return VerificationReport()

    logger.info("Extracted %d claims for verification.", len(all_claims))

    # Step 2: Verify each claim (sequential to respect PubMed rate limits)
    report = VerificationReport()

    for claim_info in all_claims:
        claim_text = claim_info["claim"]
        agent_name = claim_info["agent"]
        source_tag = claim_info["tag"]

        # Search PubMed
        papers = await _search_pubmed_for_claim(claim_text, max_pubmed_results)
        pmids = [p["pmid"] for p in papers if p.get("pmid")]

        if not papers:
            # No PubMed results → Unverified
            verification = ClaimVerification(
                claim=claim_text,
                source_agent=agent_name,
                source_tag=source_tag,
                verdict="Unverified",
                pmids=[],
                justification="No PubMed results found for this claim.",
            )
        else:
            # Haiku judges support vs contradiction
            llm_verdict, justification = await _judge_claim_against_abstracts(
                claim_text, papers
            )

            verdict_map = {
                "SUPPORTS": "Verified",
                "CONTRADICTS": "Contradicted",
                "INSUFFICIENT": "Unverified",
            }
            verdict = verdict_map.get(llm_verdict, "Unverified")

            verification = ClaimVerification(
                claim=claim_text,
                source_agent=agent_name,
                source_tag=source_tag,
                verdict=verdict,
                pmids=pmids,
                justification=justification,
            )

        report.claims.append(verification)

        # Rate limit between claims
        await asyncio.sleep(PUBMED_RATE_LIMIT_DELAY)

    # Tally
    report.total_verified = sum(1 for c in report.claims if c.verdict == "Verified")
    report.total_unverified = sum(1 for c in report.claims if c.verdict == "Unverified")
    report.total_contradicted = sum(1 for c in report.claims if c.verdict == "Contradicted")

    logger.info(
        "Verification complete: %d verified, %d unverified, %d contradicted.",
        report.total_verified, report.total_unverified, report.total_contradicted,
    )

    return report

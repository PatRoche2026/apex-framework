"""ClinicalTrials.gov MCP Server — exposes clinical trial search and detail
retrieval as FastMCP tools.

Run standalone:
    python -m mcp_servers.clinicaltrials_mcp
    # or
    fastmcp run mcp_servers/clinicaltrials_mcp.py
"""

from __future__ import annotations

import httpx
from fastmcp import FastMCP

mcp = FastMCP("ClinicalTrials")

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"
_HEADERS = {"User-Agent": "APEX-BiotechAgent/1.0 (patroche@mit.edu)"}
_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Tool 1: search_trials
# ---------------------------------------------------------------------------

@mcp.tool()
def search_trials(query: str, max_results: int = 5) -> str:
    """Search ClinicalTrials.gov for clinical studies matching a query.

    Returns NCT IDs, titles, phases, status, and conditions for top results.

    Args:
        query: Clinical trial search query (e.g. "OSMR ulcerative colitis").
        max_results: Maximum number of studies to return (default 5, max 20).
    """
    max_results = min(max_results, 20)

    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
            response = client.get(
                CTGOV_API,
                params={
                    "query.term": query,
                    "pageSize": max_results,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

        studies = data.get("studies", [])
        if not studies:
            return f"No clinical trials found for: {query}"

        lines = [f"ClinicalTrials.gov results for '{query}' ({len(studies)} studies):"]
        for study in studies:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            conditions_mod = proto.get("conditionsModule", {})

            nct_id = ident.get("nctId", "N/A")
            title = ident.get("briefTitle", "No title")
            status = status_mod.get("overallStatus", "Unknown")
            phase_list = design.get("phases", [])
            phase = ", ".join(phase_list) if phase_list else "N/A"
            conditions = ", ".join(conditions_mod.get("conditions", []))

            lines.append(
                f"\nNCT ID: {nct_id}\n"
                f"Title: {title}\n"
                f"Status: {status} | Phase: {phase}\n"
                f"Conditions: {conditions or 'N/A'}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"ClinicalTrials.gov search failed: {str(e)[:300]}"


# ---------------------------------------------------------------------------
# Tool 2: get_trial_details
# ---------------------------------------------------------------------------

@mcp.tool()
def get_trial_details(nct_id: str) -> str:
    """Fetch detailed protocol information for a specific clinical trial.

    Returns sponsor, enrollment, arms, interventions, eligibility criteria,
    endpoints, and study dates.

    Args:
        nct_id: ClinicalTrials.gov NCT identifier (e.g. "NCT04567890").
    """
    nct_id = nct_id.strip().upper()

    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
            response = client.get(
                f"{CTGOV_API}/{nct_id}",
                params={"format": "json"},
            )
            response.raise_for_status()
            study = response.json()

        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        eligibility = proto.get("eligibilityModule", {})
        outcomes_mod = proto.get("outcomesModule", {})
        description_mod = proto.get("descriptionModule", {})

        lines = [f"Clinical Trial Details: {nct_id}"]
        lines.append(f"Title: {ident.get('briefTitle', 'N/A')}")
        lines.append(f"Official Title: {ident.get('officialTitle', 'N/A')}")
        lines.append(f"Status: {status_mod.get('overallStatus', 'N/A')}")

        phase_list = design.get("phases", [])
        lines.append(f"Phase: {', '.join(phase_list) if phase_list else 'N/A'}")

        # Sponsor
        lead = sponsor_mod.get("leadSponsor", {})
        lines.append(f"Sponsor: {lead.get('name', 'N/A')} ({lead.get('class', '')})")

        # Enrollment
        enrollment = design.get("enrollmentInfo", {})
        lines.append(
            f"Enrollment: {enrollment.get('count', 'N/A')} "
            f"({enrollment.get('type', '')})"
        )

        # Description
        brief_summary = description_mod.get("briefSummary", "")
        if brief_summary:
            lines.append(f"\nBrief Summary:\n{brief_summary[:1000]}")

        # Arms & Interventions
        arms = arms_mod.get("armGroups", [])
        if arms:
            lines.append("\nArms:")
            for arm in arms:
                lines.append(
                    f"  - {arm.get('label', 'N/A')} ({arm.get('type', '')}): "
                    f"{arm.get('description', 'N/A')[:200]}"
                )

        interventions = arms_mod.get("interventions", [])
        if interventions:
            lines.append("\nInterventions:")
            for intv in interventions:
                lines.append(
                    f"  - {intv.get('type', 'N/A')}: {intv.get('name', 'N/A')} — "
                    f"{intv.get('description', 'N/A')[:200]}"
                )

        # Primary Outcomes
        primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
        if primary_outcomes:
            lines.append("\nPrimary Outcomes:")
            for out in primary_outcomes:
                lines.append(
                    f"  - {out.get('measure', 'N/A')} "
                    f"[{out.get('timeFrame', 'N/A')}]"
                )

        # Eligibility
        elig_criteria = eligibility.get("eligibilityCriteria", "")
        if elig_criteria:
            lines.append(f"\nEligibility Criteria:\n{elig_criteria[:1500]}")

        return "\n".join(lines)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Trial {nct_id} not found on ClinicalTrials.gov."
        return f"ClinicalTrials.gov request failed: {str(e)[:300]}"
    except Exception as e:
        return f"Trial detail retrieval failed for {nct_id}: {str(e)[:300]}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

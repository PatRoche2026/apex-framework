"""@Mention parser — routes a directed query to a single agent.

Parses leading or embedded ``@role`` mentions like ``@cso``, ``@scientific``,
or ``@ip`` and returns the canonical role key plus the query text with the
mention stripped.

Canonical role keys used by the rest of the codebase:
    cso, cto, cmo, cbo, ip_attorney, portfolio_director

The alias map below uses role-category synonyms only (no persona names) so
public installations stay self-documenting. If you fork APEX and name your
advisors after real people, extend ``_ALIASES`` with lowercase first names.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional

# ---------------------------------------------------------------------------
# Alias map — lowercase aliases -> canonical role keys
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # Scientific Advisor (CSO)
    "cso":         "cso",
    "scientific":  "cso",
    "science":     "cso",
    # Technical Advisor (CTO)
    "cto":         "cto",
    "technical":   "cto",
    "tech":        "cto",
    # Clinical Advisor (CMO)
    "cmo":         "cmo",
    "clinical":    "cmo",
    "medical":     "cmo",
    # Commercial Advisor (CBO)
    "cbo":         "cbo",
    "commercial":  "cbo",
    "business":    "cbo",
    # IP Strategy Advisor
    "ip":          "ip_attorney",
    "ip_attorney": "ip_attorney",
    "attorney":    "ip_attorney",
    "patent":      "ip_attorney",
    "legal":       "ip_attorney",
    # Portfolio Director
    "director":    "portfolio_director",
    "portfolio":   "portfolio_director",
}

# Match @alias anywhere in the query. Allows underscores and hyphens so
# ``@ip_attorney`` resolves. Case-insensitive.
_MENTION_RE = re.compile(r"@([a-zA-Z][a-zA-Z0-9_\-]*)", re.IGNORECASE)


class Mention(NamedTuple):
    role: Optional[str]        # canonical role key, or None if no valid mention
    query: str                 # query with the mention stripped
    raw_alias: Optional[str]   # the literal alias text found (for logging)


def parse_mention(text: str) -> Mention:
    """Parse the first @mention in ``text`` and return the canonical role.

    Returns ``Mention(role=None, query=text.strip(), raw_alias=None)`` if no
    valid @mention is present. If multiple @mentions appear, only the first
    is honoured (routing is 1-to-1, not broadcast).

    Examples:
        parse_mention("@CSO what is the GWAS evidence for MMP13?")
        -> Mention(role='cso', query='what is the GWAS evidence for MMP13?', raw_alias='CSO')

        parse_mention("hey @clinical, thoughts on the Phase II design?")
        -> Mention(role='cmo', query='hey , thoughts on the Phase II design?', raw_alias='clinical')

        parse_mention("plain text with no mention")
        -> Mention(role=None, query='plain text with no mention', raw_alias=None)
    """
    if not text:
        return Mention(role=None, query="", raw_alias=None)

    for match in _MENTION_RE.finditer(text):
        alias_raw = match.group(1)
        role = _ALIASES.get(alias_raw.lower())
        if role:
            stripped = (text[:match.start()] + text[match.end():]).strip()
            stripped = re.sub(r"\s{2,}", " ", stripped)
            return Mention(role=role, query=stripped, raw_alias=alias_raw)

    return Mention(role=None, query=text.strip(), raw_alias=None)


def list_aliases() -> dict[str, list[str]]:
    """Return canonical role -> list of accepted aliases, for /help surfaces."""
    out: dict[str, list[str]] = {}
    for alias, role in _ALIASES.items():
        out.setdefault(role, []).append(alias)
    return out

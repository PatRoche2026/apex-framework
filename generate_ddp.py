"""Generate a professional Drug Development Plan (DDP) PDF from APEX planning output.

Usage:
    from generate_ddp import generate_ddp_pdf
    path = generate_ddp_pdf(
        gene="OSMR", indication="ulcerative colitis",
        verdict="GO", confidence=78.0,
        cso_plan="...", cto_plan="...", cmo_plan="...", cbo_plan="...",
        director_synthesis="...", session_id="abc123",
        executive_scores={...},
    )
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF

# ---------------------------------------------------------------------------
# Font registration — DejaVu Sans (bundled TTF, full Unicode incl. Greek)
# ---------------------------------------------------------------------------

FONT_DIR = Path(__file__).parent / "fonts"
FONT_FAMILY = "DejaVu"
_FONT_FILES = {
    "": FONT_DIR / "DejaVuSans.ttf",
    "B": FONT_DIR / "DejaVuSans-Bold.ttf",
    "I": FONT_DIR / "DejaVuSans-Oblique.ttf",
}

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

DARK_BLUE = (26, 26, 46)
BODY_BLACK = (0, 0, 0)
GREY = (100, 100, 100)
LIGHT_GREY = (220, 220, 220)
BG_STRIPE = (248, 248, 252)
GO_GREEN = (34, 139, 34)
AMBER = (255, 140, 0)
WHITE = (255, 255, 255)
IP_CYAN = (6, 182, 212)            # #06b6d4 — IP Strategy Advisor accent color

MARGIN = 15
LINE_HEIGHT = 5.5


# ---------------------------------------------------------------------------
# Scoring rows — dimension labels + advisor roles + weights (from config.py)
# ---------------------------------------------------------------------------

def _build_weight_rows() -> list[tuple[str, str, str, float, tuple[int, int, int]]]:
    """Construct KPI Scorecard rows from config.SCORING_DIMENSIONS.

    Returns list of (dimension_key, display_label, role_title, weight, row_color) tuples.
    """
    from config import SCORING_DIMENSIONS
    role_titles = {
        "scientific_validity":   "Scientific Advisor",
        "technical_feasibility": "Technical Advisor",
        "clinical_path":         "Clinical Advisor",
        "commercial_potential":  "Commercial Advisor",
        "ip_landscape":          "IP Strategy Advisor",
    }
    display_labels = {
        "scientific_validity":   "Scientific Validity",
        "technical_feasibility": "Technical Feasibility",
        "clinical_path":         "Clinical Path",
        "commercial_potential":  "Commercial Potential",
        "ip_landscape":          "IP Landscape",
    }
    rows = []
    for dim, meta in SCORING_DIMENSIONS.items():
        color = IP_CYAN if dim == "ip_landscape" else BODY_BLACK
        rows.append((
            dim,
            display_labels.get(dim, dim.replace("_", " ").title()),
            role_titles.get(dim, "Advisor"),
            meta["weight"],
            color,
        ))
    return rows

V3_WEIGHTS = _build_weight_rows()


# ---------------------------------------------------------------------------
# Minimal sanitizer — DejaVu handles full Unicode, so we only strip zero-width
# control chars that can break layout. Greek, em-dashes, smart quotes,
# superscripts, etc. all pass through untouched.
# ---------------------------------------------------------------------------

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200B\u200C\u200D\uFEFF"), None)


def _sanitize(text: str) -> str:
    """Strip zero-width chars and BOM. Preserves all other Unicode."""
    if not text:
        return text
    # Normalise NBSP -> regular space so justification doesn't get stuck
    text = text.replace(" ", " ")
    return text.translate(_ZERO_WIDTH)


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting for plain-text PDF rendering."""
    if not text:
        return ""
    # Remove heading markers (##, ###, etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers (**text**, *text*, __text__, _text_)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    # Remove inline code backticks
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _sanitize(text).strip()


# ---------------------------------------------------------------------------
# DDP PDF class
# ---------------------------------------------------------------------------

class _DDPPDF(FPDF):
    """Custom FPDF subclass with DDP header/footer and DejaVu font registered."""

    def __init__(self, gene: str, indication: str):
        super().__init__()
        self.gene = gene
        self.indication = indication
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=20)

        # Register DejaVu as the default font family (TTF → full Unicode)
        for style, path in _FONT_FILES.items():
            self.add_font(FONT_FAMILY, style=style, fname=str(path))

    def header(self):
        """Thin top rule on non-cover pages."""
        if self.page_no() > 1:
            self.set_draw_color(*LIGHT_GREY)
            self.set_line_width(0.3)
            self.line(MARGIN, 12, self.w - MARGIN, 12)

    def footer(self):
        """Page number centered at bottom; cover page gets custom footer."""
        self.set_y(-15)
        self.set_font(FONT_FAMILY, "I", 8)
        self.set_text_color(*GREY)
        if self.page_no() == 1:
            self.cell(0, 10, "Generated by APEX v2.0 | MIT MAS.664 AI Studio", align="C")
        else:
            self.cell(0, 10, f"Page {self.page_no()}", align="C")


# ---------------------------------------------------------------------------
# Section page renderer
# ---------------------------------------------------------------------------

def _render_section(
    pdf: _DDPPDF,
    header: str,
    subheader: str,
    body: str,
) -> None:
    """Render a full section page: header → subheader → rule → body."""
    pdf.add_page()

    # Section header — single line, explicit cursor control
    pdf.set_font(FONT_FAMILY, "B", 16)
    pdf.set_text_color(*DARK_BLUE)
    pdf.set_xy(MARGIN, 20)
    pdf.cell(0, 10, _sanitize(header), align="L", new_x="LMARGIN", new_y="NEXT")

    # Subheader — single line
    pdf.set_font(FONT_FAMILY, "I", 11)
    pdf.set_text_color(*GREY)
    pdf.set_x(MARGIN)
    pdf.cell(0, 7, _sanitize(subheader), align="L", new_x="LMARGIN", new_y="NEXT")

    # Horizontal rule
    y_rule = pdf.get_y() + 2
    pdf.set_draw_color(*DARK_BLUE)
    pdf.set_line_width(0.5)
    pdf.line(MARGIN, y_rule, pdf.w - MARGIN, y_rule)
    pdf.set_xy(MARGIN, y_rule + 4)

    # Body text — reset x explicitly before multi_cell
    pdf.set_font(FONT_FAMILY, "", 11)
    pdf.set_text_color(*BODY_BLACK)
    cleaned = _strip_markdown(body) if body else "(No content generated)"
    pdf.set_x(MARGIN)
    pdf.multi_cell(pdf.w - 2 * MARGIN, LINE_HEIGHT, cleaned, align="L")


# ---------------------------------------------------------------------------
# KPI Scorecard page
# ---------------------------------------------------------------------------

def _render_kpi_scorecard(pdf: _DDPPDF, executive_scores: dict[str, Any]) -> None:
    """Render the V3 Weighted Composite Score scorecard table (page 2).

    Each row shows: Dimension | Analyst | Weight | Score | Weighted contribution.
    Composite row sums the weighted contributions — the math is internally
    consistent on the page (score × weight = weighted; sum = composite).
    """
    pdf.add_page()

    # -- Page header --
    pdf.set_font(FONT_FAMILY, "B", 18)
    pdf.set_text_color(*DARK_BLUE)
    pdf.set_xy(MARGIN, 20)
    pdf.cell(0, 10, "KPI Scorecard", align="L", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(FONT_FAMILY, "I", 11)
    pdf.set_text_color(*GREY)
    pdf.set_x(MARGIN)
    pdf.cell(0, 7, "V3 Weighted Composite Score", align="L", new_x="LMARGIN", new_y="NEXT")

    # Rule under title
    y_rule = pdf.get_y() + 2
    pdf.set_draw_color(*DARK_BLUE)
    pdf.set_line_width(0.5)
    pdf.line(MARGIN, y_rule, pdf.w - MARGIN, y_rule)
    pdf.ln(6)

    # -- Table geometry --
    # Column widths (sum = 180 which matches page-width - 2*MARGIN for A4 = 210-2*15)
    COL_DIM, COL_ANALYST, COL_WEIGHT, COL_SCORE, COL_WEIGHTED = 56, 58, 22, 22, 22
    ROW_H = 10
    HEAD_H = 9

    # Header row
    pdf.set_font(FONT_FAMILY, "B", 10)
    pdf.set_fill_color(*DARK_BLUE)
    pdf.set_text_color(*WHITE)
    pdf.set_x(MARGIN)
    pdf.cell(COL_DIM,      HEAD_H, "Dimension",   border=0, align="L", fill=True)
    pdf.cell(COL_ANALYST,  HEAD_H, "Analyst",     border=0, align="L", fill=True)
    pdf.cell(COL_WEIGHT,   HEAD_H, "Weight",      border=0, align="C", fill=True)
    pdf.cell(COL_SCORE,    HEAD_H, "Score",       border=0, align="C", fill=True)
    pdf.cell(COL_WEIGHTED, HEAD_H, "Weighted",    border=0, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

    # Data rows
    per_dim = (executive_scores or {}).get("per_dimension", {}) or {}
    row_weighted_sum = 0.0
    score_total = 0.0  # for display in composite "Score" column (weighted avg)
    pdf.set_font(FONT_FAMILY, "", 10)
    for i, (dim_key, label, analyst, weight, row_color) in enumerate(V3_WEIGHTS):
        score = float(per_dim.get(dim_key, 0) or 0)
        weighted = score * weight
        row_weighted_sum += weighted
        score_total += score * weight  # same value — kept explicit for clarity

        stripe = (i % 2 == 0)
        if stripe:
            pdf.set_fill_color(*BG_STRIPE)
        else:
            pdf.set_fill_color(*WHITE)

        pdf.set_x(MARGIN)
        # Dimension cell — colored text for IP row
        pdf.set_text_color(*row_color)
        pdf.set_font(FONT_FAMILY, "B" if row_color == IP_CYAN else "", 10)
        pdf.cell(COL_DIM, ROW_H, _sanitize(label), border=0, align="L", fill=stripe)

        # Analyst — grey italic, cyan for IP
        pdf.set_font(FONT_FAMILY, "I", 10)
        pdf.set_text_color(*(row_color if row_color == IP_CYAN else GREY))
        pdf.cell(COL_ANALYST, ROW_H, _sanitize(analyst), border=0, align="L", fill=stripe)

        # Weight / Score / Weighted — numeric columns
        pdf.set_font(FONT_FAMILY, "", 10)
        pdf.set_text_color(*BODY_BLACK)
        pdf.cell(COL_WEIGHT,   ROW_H, f"{weight*100:.0f}%",    border=0, align="C", fill=stripe)
        pdf.cell(COL_SCORE,    ROW_H, f"{score:.1f} / 10",     border=0, align="C", fill=stripe)
        pdf.cell(COL_WEIGHTED, ROW_H, f"{weighted:.2f}",       border=0, align="C", fill=stripe,
                 new_x="LMARGIN", new_y="NEXT")

    # Separator line before composite
    y_sep = pdf.get_y() + 1
    pdf.set_draw_color(*DARK_BLUE)
    pdf.set_line_width(0.3)
    pdf.line(MARGIN, y_sep, pdf.w - MARGIN, y_sep)
    pdf.ln(2)

    # -- COMPOSITE row --
    composite = round(row_weighted_sum, 2)
    pdf.set_font(FONT_FAMILY, "B", 11)
    pdf.set_fill_color(*DARK_BLUE)
    pdf.set_text_color(*WHITE)
    pdf.set_x(MARGIN)
    pdf.cell(COL_DIM,      ROW_H + 2, "COMPOSITE",      border=0, align="L", fill=True)
    pdf.cell(COL_ANALYST,  ROW_H + 2, "Weighted total", border=0, align="L", fill=True)
    pdf.cell(COL_WEIGHT,   ROW_H + 2, "100%",           border=0, align="C", fill=True)
    pdf.cell(COL_SCORE,    ROW_H + 2, f"{composite:.2f} / 10",  border=0, align="C", fill=True)
    pdf.cell(COL_WEIGHTED, ROW_H + 2, f"{composite:.2f}",       border=0, align="C", fill=True,
             new_x="LMARGIN", new_y="NEXT")

    # -- Footnote: show API's weighted_total if it differs from math --
    api_total = (executive_scores or {}).get("weighted_total")
    pdf.ln(6)
    pdf.set_text_color(*GREY)
    pdf.set_font(FONT_FAMILY, "I", 9)
    pdf.set_x(MARGIN)
    pdf.multi_cell(
        pdf.w - 2 * MARGIN, 4.5,
        "Weights per V3 scoring model (sum = 100%). Per-dimension scores are "
        "averaged across all executives who scored that dimension; the IP "
        "Landscape dimension is contributed by the IP Strategy Advisor only.",
        align="L",
    )
    if api_total is not None and abs(float(api_total) - composite) > 0.05:
        pdf.ln(1)
        pdf.set_x(MARGIN)
        pdf.multi_cell(
            pdf.w - 2 * MARGIN, 4.5,
            f"Director's narrative composite: {float(api_total):.2f} / 10 "
            f"(qualitative adjustment of the {composite:.2f} arithmetic composite).",
            align="L",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_ddp_pdf(
    gene: str,
    indication: str,
    verdict: str,
    confidence: float,
    cso_plan: str,
    cto_plan: str,
    cmo_plan: str,
    cbo_plan: str,
    director_synthesis: str,
    session_id: str,
    ip_attorney_plan: str = "",
    executive_scores: dict[str, Any] | None = None,
) -> str:
    """Generate a Drug Development Plan PDF from APEX planning output.

    Args:
        gene:               Gene symbol (e.g. "OSMR")
        indication:         Disease indication (e.g. "ulcerative colitis")
        verdict:            Portfolio Director verdict ("GO", "CONDITIONAL GO", "NO-GO")
        confidence:         Confidence score 0–100
        cso_plan:           Scientific Advisor Target Validation section
        cto_plan:           Technical Advisor Modality & Manufacturing section
        cmo_plan:           Clinical Advisor Development Strategy section
        cbo_plan:           Commercial Advisor Strategic Assessment section
        director_synthesis: Portfolio Director Executive Summary & Timeline
        session_id:         APEX session ID (used in filename)
        ip_attorney_plan:   IP Strategy Advisor Prosecution Plan (optional
                            — empty string renders no IP section)
        executive_scores:   Full scoring dict from state (must contain per_dimension;
                            weighted_total is read if present but the scorecard computes
                            its own composite from per-dim × weight for internal
                            mathematical consistency).

    Returns:
        Absolute path to the generated PDF file.
    """
    # Ensure output directory exists
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Build filename (sanitize spaces, lowercase indication)
    safe_indication = indication.lower().replace(" ", "_")
    filename = f"ddp_{gene}_{safe_indication}_{session_id[:8]}.pdf"
    output_path = reports_dir / filename

    # Light sanitize (strip zero-width chars only — DejaVu handles full Unicode)
    gene = _sanitize(gene)
    indication = _sanitize(indication)
    verdict = _sanitize(verdict)

    # Init PDF
    pdf = _DDPPDF(gene=gene, indication=indication)
    pdf.set_title(f"Drug Development Plan — {gene} for {indication}")
    pdf.set_author("APEX v2.0 | MIT MAS.664 AI Studio")

    # ------------------------------------------------------------------
    # PAGE 1 — Cover
    # ------------------------------------------------------------------
    pdf.add_page()
    pdf.set_y(40)

    # Main title
    pdf.set_font(FONT_FAMILY, "B", 28)
    pdf.set_text_color(*DARK_BLUE)
    pdf.cell(0, 14, "DRUG DEVELOPMENT PLAN", align="C", new_x="LMARGIN", new_y="NEXT")

    # Gene + indication subtitle
    pdf.set_font(FONT_FAMILY, "", 16)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 10, f"{gene}  |  {indication.title()}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(12)

    # Verdict badge
    verdict_upper = verdict.upper().strip()
    if "NO-GO" in verdict_upper:
        badge_color = (160, 160, 160)
    elif "CONDITIONAL" in verdict_upper:
        badge_color = AMBER
    else:
        badge_color = GO_GREEN

    badge_w = 70
    badge_h = 14
    badge_x = (pdf.w - badge_w) / 2
    badge_y = pdf.get_y()

    pdf.set_fill_color(*badge_color)
    pdf.set_draw_color(*badge_color)
    pdf.rect(badge_x, badge_y, badge_w, badge_h, style="F")

    pdf.set_xy(badge_x, badge_y + 2)
    pdf.set_font(FONT_FAMILY, "B", 13)
    pdf.set_text_color(*WHITE)
    pdf.cell(badge_w, badge_h - 4, verdict_upper, align="C")

    # Confidence score below badge
    pdf.set_xy(MARGIN, badge_y + badge_h + 6)
    pdf.set_font(FONT_FAMILY, "", 12)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 8, f"Confidence Score: {confidence:.0f}%", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_draw_color(*LIGHT_GREY)
    pdf.set_line_width(0.4)
    pdf.line(MARGIN + 20, pdf.get_y(), pdf.w - MARGIN - 20, pdf.get_y())
    pdf.ln(8)

    # Date
    pdf.set_font(FONT_FAMILY, "", 11)
    pdf.set_text_color(*GREY)
    date_str = datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 8, date_str, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # Prepared by block
    pdf.set_font(FONT_FAMILY, "I", 10)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6, "Prepared by APEX Executive Team", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Scientific · Technical · Clinical · Commercial · IP Strategy Advisors",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Synthesized by the Portfolio Director", align="C", new_x="LMARGIN", new_y="NEXT")

    # ------------------------------------------------------------------
    # PAGE 2 — KPI Scorecard (NEW)
    # ------------------------------------------------------------------
    _render_kpi_scorecard(pdf, executive_scores or {})

    # ------------------------------------------------------------------
    # PAGE 3 — Executive Summary (Director synthesis)
    # ------------------------------------------------------------------
    _render_section(
        pdf,
        header="Executive Summary",
        subheader="Portfolio Director",
        body=director_synthesis,
    )

    # ------------------------------------------------------------------
    # PAGE 4 — Target Validation Strategy (CSO)
    # ------------------------------------------------------------------
    _render_section(
        pdf,
        header="Section 1: Target Validation Strategy",
        subheader="Scientific Advisor",
        body=cso_plan,
    )

    # ------------------------------------------------------------------
    # PAGE 5 — Modality & Manufacturing Strategy (CTO)
    # ------------------------------------------------------------------
    _render_section(
        pdf,
        header="Section 2: Modality & Manufacturing Strategy",
        subheader="Technical Advisor",
        body=cto_plan,
    )

    # ------------------------------------------------------------------
    # PAGE 6 — Clinical Development Strategy (CMO)
    # ------------------------------------------------------------------
    _render_section(
        pdf,
        header="Section 3: Clinical Development Strategy",
        subheader="Clinical Advisor",
        body=cmo_plan,
    )

    # ------------------------------------------------------------------
    # PAGE 7 — Commercial & Strategic Assessment (CBO)
    # ------------------------------------------------------------------
    _render_section(
        pdf,
        header="Section 4: Commercial & Strategic Assessment",
        subheader="Commercial Advisor",
        body=cbo_plan,
    )

    # ------------------------------------------------------------------
    # PAGE 8 — IP Strategy & Prosecution Plan (Chief IP Counsel)
    # ------------------------------------------------------------------
    if ip_attorney_plan and ip_attorney_plan.strip():
        _render_section(
            pdf,
            header="Section 5: IP Strategy & Prosecution Plan",
            subheader="IP Strategy Advisor",
            body=ip_attorney_plan,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pdf.output(str(output_path))
    return str(output_path.resolve())

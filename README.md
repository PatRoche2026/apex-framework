# APEX — Agentic Pipeline for Executive Decisions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Live Demo](https://img.shields.io/badge/demo-apex--biotech.lovable.app-purple.svg)](https://apex-biotech.lovable.app)

**APEX** is a multi-agent coordination framework that simulates a structured executive debate over the viability of a biomedical research target. A research scout gathers evidence from public literature, five domain specialists (science, technology, clinical, commercial, IP) assess the target from their respective vantage points, debate each other through a round of rebuttals, and a portfolio director synthesizes a GO / CONDITIONAL GO / NO-GO verdict with a confidence score.

The framework is **domain-agnostic**: swap the knowledge corpus and adjust the agent prompts and it can evaluate anything from a biotech target to a software architecture proposal. The default configuration demonstrates the protocol on publicly-studied knee osteoarthritis (KOA) drug targets.

> Built for **MIT Media Lab MAS.664 — AI Studio: AI Agents and Agentic Web** (Spring 2026). Demo Day: May 7, 2026.

**[Try the live demo →](https://apex-biotech.lovable.app)**

<!-- Screenshot placeholder — add docs/screenshot.png to enable:
![APEX demo](docs/screenshot.png)
-->


---

## What APEX demonstrates

APEX treats **agent coordination** as the first-class problem. The agents are not competing — they are constructively disagreeing through a structured protocol designed to surface weak claims, pull in missing evidence, and converge on a well-reasoned verdict that a human can inspect and override.

Key protocol elements:

- **Parallel independent assessment** removes groupthink — each specialist evaluates the target without seeing peers' views first.
- **Structured rebuttal round** forces each specialist to challenge at least one peer's claim with specific evidence.
- **Synthesis by a dedicated director** — not by majority vote. The director weighs each dimension, reconciles disagreements, and produces a single decision.
- **Confidence-based re-run loop** — if synthesis confidence falls below a threshold, the system re-debates with the new information surfaced in the first round.
- **Retrieval-augmented evidence** — every claim is anchored in a knowledge base, with self-reflection on claim support.

---

## Architecture

```
                              START
                                │
                                ▼
                        ┌───────────────┐
                        │ Research      │ PubMed search + PMC full-text
                        │ Scout         │ enrichment on top-ranked papers
                        └───────┬───────┘
                                │
                  ┌─────────────┼──────────────┐
                  ▼             ▼              ▼
          ┌─────────────┐ ┌─────────┐  ┌──────────────┐
          │ Scientific  │ │Technical│  │ Clinical /   │   ← parallel fan-out
          │ Advisor     │ │Advisor  │  │ Commercial / │     (5 roles assess
          └──────┬──────┘ └────┬────┘  │ IP Advisors  │      independently)
                 │             │       └──────┬───────┘
                 └─────────────┼──────────────┘
                               ▼
                       ┌───────────────┐
                       │ Debate Router │  (fan-in / fan-out barrier)
                       └───────┬───────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼         ← parallel fan-out
          ┌───────────┐ ┌────────────┐ ┌────────┐       (rebuttals — each
          │ Rebuttal  │ │ Rebuttal   │ │ ...    │        challenges peers)
          └─────┬─────┘ └─────┬──────┘ └───┬────┘
                │             │            │
                └─────────────┼────────────┘
                              ▼
                      ┌───────────────┐
                      │ Portfolio     │ Weighted synthesis →
                      │ Director      │ GO / CONDITIONAL GO / NO-GO
                      └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │ Consensus     │ confidence < 60% ──▶ re-debate
                      │ Check         │ confidence ≥ 60% ──▶ END
                      └───────────────┘
```

After a GO verdict is accepted, a second graph runs in parallel to generate a **Drug Development Plan (DDP)** PDF — each specialist drafts their section and the director synthesizes the final report.

---

## Multi-LLM Truth Tribunal

Running a debate through a single foundation-model family exposes you to that model's blind spots — its training-data gaps, its priors, its refusal patterns. APEX includes a **Truth Tribunal** layer that runs the same evaluation against up to three model families in parallel (Claude / GPT-4o / Gemini), then reconciles their verdicts with a meta-judge.

```
         POST /evaluate/tribunal
                │
   ┌────────────┼────────────┐
   ▼            ▼            ▼          ← asyncio.gather(return_exceptions=True)
┌────────┐ ┌────────┐ ┌────────┐
│ APEX   │ │ APEX   │ │ APEX   │
│ graph  │ │ graph  │ │ graph  │           same prompts, same RAG,
│ on     │ │ on     │ │ on     │           different provider
│Claude  │ │GPT-4o  │ │Gemini  │
└───┬────┘ └───┬────┘ └───┬────┘
    │          │          │
    └──────────┼──────────┘
               ▼
       ┌───────────────┐
       │  Meta-judge   │   synthesizes 2-3 stack verdicts,
       │ (synthesis-   │   flags singleton claims (likely
       │  tier model)  │   hallucination vs novel insight),
       └───────┬───────┘   issues consensus + agreement score
               ▼
       Consensus verdict + per-provider cost breakdown
```

**Graceful degradation.** Providers without API keys are silently skipped. Providers that error mid-run are captured in the `errors` field but do not fail the tribunal as long as at least `min_survivors` (default 2) succeed. If fewer than `min_survivors` succeed, the endpoint returns `503` with per-provider diagnostics.

**Tiered routing via `agents/llm_router.py`.** Roles are bucketed into a synthesis tier (CSO, Portfolio Director) and a specialist tier (everyone else). Each provider supplies both: Anthropic uses Claude Sonnet 4 / Haiku 4.5, OpenAI uses GPT-4o / GPT-4o-mini, Google uses Gemini 2.5 Pro / Gemini 2.0 Flash. Cost accounting is per-provider and surfaces in the tribunal response.

**Single-provider mode still works.** Pass `"provider": "openai"` to the regular `/evaluate` endpoint to run one evaluation against a single backend — no tribunal overhead.

---

## Direct agent queries (@mention routing)

Not every question needs a full 90-second multi-agent debate. The `/ask` endpoint routes a single directed question to one specialist and returns in under 10 seconds.

```
POST /ask      { "query": "@scientific what is the GWAS evidence for MMP13?" }

   ┌────────────────────┐
   │ @mention parser    │    resolves @scientific → cso,
   │ agents/mention.py  │    @clinical → cmo, @ip → ip_attorney, etc.
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ RAG retrieve (k=5) │    role-specific ChromaDB collection
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ Single-agent LLM   │    short 150-300 word answer, max_tokens=900
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ Claim verify       │    extracts [SUPPORTED]/[UNSUPPORTED]/[UNCERTAIN]
   │ (regex — fast)     │    reflection tokens for inline UI rendering
   └────────────────────┘
```

**Session context.** Pass an existing `session_id` from a prior `/evaluate` run and the selected agent receives its own earlier assessment as context, so the answer builds on its prior analysis instead of starting fresh.

**WebSocket variant.** `/ws/ask` streams `node_complete` events (`rag_retrieve` → `single_agent_llm` → `claim_verify`) so the UI can render progressive feedback.

**Aliases.** The @mention parser accepts role synonyms (`@scientific`, `@technical`, `@clinical`, `@commercial`, `@ip`, `@director`) in addition to the canonical role keys. See `/ask/aliases` for the full list.

---

## Agent Coordination Protocol

| Phase | What happens | Why it matters |
|-------|-------------|----------------|
| **1. Scout** | Issue three query variants against PubMed, rank hits, enrich top 3 with PMC full text | Grounds the debate in current peer-reviewed evidence, not LLM pretraining |
| **2. Independent assessment** | Five advisors evaluate the target in parallel, each using role-specific RAG context | Prevents anchoring bias; reveals where specialists genuinely disagree |
| **3. IP analysis** | Dedicated IP Strategy Advisor runs patent landscape + FTO analysis via Lens.org | IP posture is often the deciding factor in target selection — it deserves its own first-class role |
| **4. Adversarial debate** | Each advisor issues a rebuttal, forced to cite specific evidence and name at least one peer claim they challenge | Surfaces under-specified claims and drives evidence retrieval |
| **5. Synthesis** | Portfolio Director weighs each dimension (configurable in `config.py`), writes a structured verdict with reasoning | Produces a single decision with traceable reasoning — not a vote count |
| **6. Confidence loop** | If the verdict confidence is below threshold and round limit is not yet reached, re-enter rebuttals with the synthesized concerns | Reruns are evidence-informed, not random |

---

## Technology stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph 0.2+ (StateGraph) |
| Backend API | FastAPI 0.115+ with WebSocket streaming |
| LLM (multi-provider) | Claude Sonnet 4 / Haiku 4.5 · GPT-4o / GPT-4o-mini · Gemini 2.5 Pro / Gemini 2.0 Flash |
| RAG / vector store | ChromaDB 0.5+ (persistent, cosine similarity) |
| Embeddings | Voyage AI `voyage-3-lite` |
| Research APIs | PubMed / NCBI Entrez, ClinicalTrials.gov v2, Open Targets GraphQL, Lens.org Patent |
| Tool protocol | FastMCP (4 servers: PubMed, ClinicalTrials, UniProt, bioRxiv) |
| PDF generation | fpdf2 with DejaVu Sans (Unicode) |
| Deployment | Docker + Railway |
| Language | Python 3.11+ |

---

## RAG pipeline

```
Query ──▶ HyDE ──▶ CRAG retrieval ──▶ Self-RAG reflection ──▶ Output
         │         │                   │
         │         │                   └─ tags claims as
         │         │                      [SUPPORTED] / [UNSUPPORTED] / [UNCERTAIN]
         │         │
         │         └─ if retrieval confidence < 0.7:
         │            escalate to live PubMed / MCP tools
         │
         └─ generate hypothetical ideal abstract,
            embed with Voyage AI, search ChromaDB
```

All retrieved evidence is attributed to its source document; the director's final synthesis preserves citation provenance through the DDP PDF.

---

## Quick start

### Prerequisites

- Python 3.11+
- At least one LLM provider key — [Anthropic](https://console.anthropic.com/), [OpenAI](https://platform.openai.com/), or [Google AI Studio](https://aistudio.google.com/)
- [Voyage AI API key](https://www.voyageai.com/) (free tier is sufficient for the sample corpus)
- (Optional) [Lens.org API token](https://www.lens.org/lens/user/subscriptions) for patent analysis
- (Optional) All three provider keys — enables the Multi-LLM Truth Tribunal

### Install

```bash
git clone https://github.com/PatRoche2026/apex-framework.git
cd apex-framework
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and VOYAGE_API_KEY at minimum
```

### Ingest the sample knowledge base

```bash
python -m rag.ingest
```

This embeds the 5 sample KOA target markdowns (under `knowledge/shared/`) into six ChromaDB collections. First run takes ~30 seconds.

### Run the server

```bash
uvicorn server:app --reload --port 8000
```

### Test

```bash
curl http://localhost:8000/health
curl http://localhost:8000/personas
curl http://localhost:8000/targets         # returns the 5 DEMO_TARGETS from config.py

# Run a full evaluation (takes ~90 seconds)
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query": "MMP13 knee osteoarthritis"}'

# Route a directed question to one agent (<10s)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "@scientific what is the GWAS evidence for MMP13?"}'

# Run the Multi-LLM Truth Tribunal (needs anthropic + openai + google keys;
# takes ~3-4 min for 3 parallel evaluations + meta-judge)
curl -X POST http://localhost:8000/evaluate/tribunal \
  -H "Content-Type: application/json" \
  -d '{"query": "MMP13 knee osteoarthritis"}'
```

Open `http://localhost:8000/docs` in a browser for the interactive FastAPI documentation.

---

## API reference

### REST

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Root info |
| `/health` | GET | Health check + session count |
| `/personas` | GET | Agent metadata (role titles, colors) |
| `/targets` | GET | Demo target catalog (from `config.DEMO_TARGETS`) |
| `/metrics` | GET | Framework-level stats |
| `/evaluate` | POST | Synchronous full evaluation (optional `provider` field) |
| `/evaluate/tribunal` | POST | Multi-LLM Truth Tribunal — parallel evaluation on anthropic/openai/google + meta-judge synthesis |
| `/ask` | POST | Direct @mention query to a single agent (target latency <10s) |
| `/ask/aliases` | GET | Alias → canonical-role map for @mention UI |
| `/results/{session_id}` | GET | Fetch completed evaluation |
| `/sessions` | GET | List all sessions |
| `/export/{session_id}` | GET | Download Markdown report |
| `/download/ddp/{session_id}` | GET | Download DDP PDF |
| `/feedback/{session_id}` | POST | Submit feedback + re-evaluate, or `action:"approve"` to trigger DDP |
| `/reject/{session_id}` | POST | Mark session rejected |
| `/batch/evaluate` | POST | Multi-target batch evaluation |
| `/batch/status/{batch_id}` | GET | Batch progress |

### WebSocket (streaming)

| Route | Description |
|-------|-------------|
| `/ws/evaluate` | Stream evaluation events node-by-node |
| `/ws/feedback/{session_id}` | Stream re-evaluation after feedback |
| `/ws/plan/{session_id}` | Stream DDP planning pipeline |
| `/ws/ask` | Stream the 3 nodes of a single-agent @mention query |

---

## Scoring system

Each advisor scores the target on 5 dimensions (0-10). The portfolio director computes a weighted composite.

| Dimension | Default weight | Scored by |
|-----------|---------------|-----------|
| Scientific Validity | 20% | Scientific Advisor |
| Technical Feasibility | 20% | Technical Advisor |
| Clinical Path | 20% | Clinical Advisor |
| Commercial Potential | 20% | Commercial Advisor |
| IP Landscape | 20% | IP Strategy Advisor |

Weights are configurable in `config.py` — if IP is especially load-bearing in your domain, give it 30%; if you want to downweight commercial considerations for research-stage triage, cut it to 10%. Weights must sum to 1.0.

Final verdict categories:
- **GO** — proceed to full development plan
- **CONDITIONAL GO** — proceed with caveats (director lists specific conditions)
- **NO-GO** — target does not warrant further investment

---

## Project structure

```
apex-framework/
├── README.md
├── LICENSE
├── config.py                 # Scoring weights, models, DEMO_TARGETS, collection names
├── server.py                 # FastAPI app (REST + WebSocket)
├── generate_ddp.py           # DDP PDF generation (fpdf2 + DejaVu Unicode font)
├── requirements.txt
├── Dockerfile
├── Procfile
├── .env.example
├── agents/
│   ├── __init__.py           # Shared config, concurrency semaphore, role list
│   ├── state.py              # APEXState TypedDict (with provider field)
│   ├── graph.py              # build_graph() + build_planning_graph()
│   ├── scout.py              # Research Scout node
│   ├── executives.py         # Factory for 4 advisor assessment + rebuttal nodes
│   ├── ip_attorney.py        # IP Strategy Advisor node (Lens.org integration)
│   ├── director.py           # Portfolio Director (synthesis + scoring)
│   ├── planning.py           # DDP planning nodes (5 parallel + synthesis)
│   ├── tools.py              # PubMed, ClinicalTrials, Open Targets, Lens.org
│   ├── prompts.py            # Generic biomedical system prompts
│   ├── llm_router.py         # Multi-provider factory + cost accounting
│   ├── tribunal.py           # Truth Tribunal: 3 parallel runs + meta-judge
│   ├── mention.py            # @mention parser (aliases → canonical role)
│   └── ask_graph.py          # Single-agent Q&A subgraph for @mention routing
├── rag/
│   ├── store.py              # ChromaDB collection setup
│   ├── embeddings.py         # Voyage AI wrapper
│   ├── retriever.py          # retrieve_context(role, query, k)
│   ├── ingest.py             # Chunk + batch-embed knowledge files
│   ├── crag.py               # Quality-graded retrieval + live search fallback
│   ├── claim_verifier.py     # Self-RAG reflection tokens
│   └── hyde.py               # Hypothetical Document Embeddings
├── mcp_servers/              # FastMCP servers
│   ├── pubmed_mcp.py
│   ├── clinicaltrials_mcp.py
│   ├── uniprot_mcp.py
│   └── biorxiv_mcp.py
├── knowledge/                # RAG corpus — one subdir per role (+ shared/)
│   └── shared/               # 5 public KOA target summaries (cross-role)
│       ├── MMP13_osteoarthritis.md
│       ├── ADAMTS5_osteoarthritis.md
│       ├── IL6_osteoarthritis.md
│       ├── GDF5_osteoarthritis.md
│       └── FGF18_osteoarthritis.md
│   # Add knowledge/cso/, knowledge/cto/, knowledge/cmo/, knowledge/cbo/,
│   # knowledge/ip_attorney/ folders for role-specific content.
├── fonts/                    # DejaVu Sans (Unicode support for PDF)
└── docs/                     # GitHub Pages documentation
```

---

## Customizing for your domain

1. **Define your agents.** Edit `agents/prompts.py` — rename role titles, rewrite system prompts to your domain's evaluation criteria. Keep the six roles (or prune `ip_attorney` if IP analysis is not relevant to your domain).
2. **Curate your knowledge.** Replace `knowledge/shared/` with your own markdown files (one per topic). Re-run `python -m rag.ingest`.
3. **Tune scoring weights.** Edit `config.SCORING_DIMENSIONS` to reflect what matters in your domain (weights must sum to 1.0).
4. **Add external tools.** Extend `agents/tools.py` with your own domain APIs (market data, regulatory databases, etc.). Extend `mcp_servers/` for protocol-managed tools.
5. **Tune the debate protocol.** Adjust `MAX_DEBATE_ROUNDS`, `CONFIDENCE_THRESHOLD`, and `MAX_CONCURRENT_LLM_CALLS` in `config.py`.

---

## Limitations

- Session storage is in-memory (lost on server restart). For production, add Redis or PostgreSQL.
- No user authentication — add as needed.
- Rate limits: concurrent LLM calls are capped at 3 by default; increase only if your API tier supports it.
- WebSocket proxies (e.g., Railway) may time out on long evaluations. The built-in 60-second ping keepalive mitigates this but is not bulletproof.
- Voyage AI embeddings require an internet connection; no local embedding fallback is included.

---

## Course context

Built for **MIT Media Lab MAS.664 — AI Studio: AI Agents and Agentic Web** (Spring 2026). The course theme is **agent coordination protocols** — how multiple agents can cooperate and constructively disagree to produce decisions a single agent could not.

Demo Day: **May 7, 2026** at MIT Media Lab.

---

## Author

Pat Ovando Roche — MIT Sloan Fellows MBA 2026.
[LinkedIn](https://www.linkedin.com/in/patovandoroche) · [GitHub](https://github.com/PatRoche2026)

---

## License

MIT — see [LICENSE](./LICENSE).

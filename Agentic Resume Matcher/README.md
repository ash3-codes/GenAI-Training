# Agentic Resume Matcher

An AI-powered recruitment support platform that ingests resumes and job descriptions, parses them into structured schemas using LLMs, embeds them into a vector database, and ranks candidates using a multi-stage scoring pipeline — all orchestrated through LangGraph and surfaced via a Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Pipeline Details](#pipeline-details)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Query Pipeline](#query-pipeline)
- [Scoring System](#scoring-system)
- [Developer Tools](#developer-tools)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Agentic Resume Matcher solves a core recruiting problem: matching hundreds of resumes to a job description quickly and accurately, without manual screening.

**What it does:**
- Parses PDF, DOCX, and TXT resumes into structured candidate profiles using GPT-4o
- Embeds both resumes and job descriptions into a 1536-dimensional vector space
- Retrieves top candidates via hybrid search (vector + BM25 keyword + metadata filtering)
- Re-ranks candidates using a local CrossEncoder model
- Computes a weighted ATS score and a final fused score for each candidate
- Displays ranked results in a Streamlit dashboard with per-candidate score breakdowns

**Scale:** Designed for corpora of 800+ resumes. Parallel LLM parsing (10 concurrent workers) and a checkpoint system that skips already-indexed files keep re-ingestion under 25 minutes.

---

## Architecture

```
HR User (Streamlit UI)
        │
        ▼
┌───────────────────────────────────────────────────┐
│              LangGraph Orchestrator               │
│                                                   │
│  INGESTION PIPELINE                               │
│  load_documents → checkpoint_filter               │
│    → parse_resume (parallel, 10 workers)          │
│    → validate_schema → expand_skills              │
│    → embed_and_store (batch=50)                   │
│                                                   │
│  QUERY PIPELINE                                   │
│  parse_jd → embed_jd → hybrid_retrieve            │
│    → rerank_candidates → score_ats                │
│    → fuse_scores → aggregate_results              │
└───────────────────────────────────────────────────┘
        │                         │
        ▼                         ▼
  Qdrant Cloud               Azure OpenAI
  (resumes_index,            (gpt-4o parsing,
   jd_index)                  text-embedding-3-small)
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| LangGraph orchestration | Modular nodes, explicit state, easy to extend or swap individual steps |
| Qdrant Cloud vector DB | Managed, scalable, supports payload filtering alongside vector search |
| Hybrid retrieval (vector + BM25 + RRF) | Pure semantic search misses exact skill keywords; BM25 catches them |
| Local CrossEncoder reranker | More accurate than bi-encoder for pairwise ranking; zero API cost |
| Skill Knowledge Graph | Infers base languages from frameworks (NumPy → Python) to fix retrieval gaps |
| Checkpoint filter | Skips already-indexed resumes before the LLM parse step — the biggest speed win |

---

## Project Structure

```
agentic_resume_matcher/
│
├── app/
│   ├── main.py                  # Streamlit entry point (Match, Ingest, Analytics pages)
│   └── pages/
│       ├── app.py               # Account / login page
│       └── dashboard.py         # Search history & export
│
├── config/
│   └── settings.py              # Singleton config + client factories
│
├── graphs/
│   ├── state.py                 # Shared LangGraph PipelineState TypedDict
│   ├── ingestion_graph.py       # Compiled ingestion pipeline
│   └── query_graph.py           # Compiled query pipeline
│
├── nodes/
│   ├── load_documents.py        # PDF / DOCX / TXT loader
│   ├── checkpoint_filter.py     # Skip already-indexed resumes (pre-parse)
│   ├── parse_resume.py          # GPT-4o structured extraction (parallel)
│   ├── validate_schema.py       # Pydantic validation + LLM repair (3 retries)
│   ├── expand_skills.py         # Knowledge graph skill inference
│   ├── embed_and_store.py       # Azure Embeddings + Qdrant upsert (batch=50)
│   ├── parse_jd.py              # GPT-4o JD structured extraction
│   ├── embed_jd.py              # JD embedding
│   ├── hybrid_retrieve.py       # Vector + BM25 + RRF fusion
│   ├── rerank_candidates.py     # CrossEncoder reranking
│   ├── score_ats.py             # Weighted ATS scoring
│   ├── fuse_scores.py           # Final score fusion
│   └── aggregate_results.py     # Result normalisation + summary
│
├── retrieval/
│   ├── vector_search.py         # Qdrant cosine similarity search
│   ├── bm25_search.py           # BM25 keyword search over payloads
│   └── rrf.py                   # Reciprocal Rank Fusion
│
├── schemas/
│   ├── resume_schema.py         # ResumeSchema Pydantic model
│   └── jd_schema.py             # JobDescriptionSchema Pydantic model
│
├── skills/
│   └── skills_graph.json        # 102 skill → parent mappings
│
├── utils/
│   ├── config_loader.py         # Typed config accessors (cached)
│   ├── embed_template.py        # Resume → embedding text builder
│   └── logger.py                # Structured JSON logging + NodeTimer
│
├── data/
│   ├── resumes/                 # Drop resume files here (PDF/DOCX/TXT)
│   └── jd/                      # Drop JD files here (TXT/PDF)
│
├── logs/
│   └── pipeline.log             # Structured JSON node execution logs
│
├── inspect_pipeline.py          # Dev tool: run + inspect each node individually
├── config.yaml                  # All tunable parameters
├── requirements.txt
└── .env                         # Secrets (never commit)
```

---

## Prerequisites

- Python 3.10+
- An **Azure OpenAI** resource with:
  - A chat deployment (GPT-4o recommended)
  - An embeddings deployment (`text-embedding-3-small`)
- A **Qdrant Cloud** account with a cluster URL and API key
- Node.js (only needed if regenerating the `.docx` documentation)

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd agentic_resume_matcher

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template and fill in your secrets
cp .env.example .env
```

---

## Configuration

### `.env` — Secrets

```env
# Azure OpenAI
ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
DEPLOYMENT_NAME=gpt-4o
DEPLOYMENT_NAME_EMBEDDING=text-embedding-3-small
API_VERSION=2024-02-01
API_VERSION_EMBEDDING=2024-02-01

# Qdrant Cloud
QDARNT_ENDPOINT=https://<your-cluster>.qdrant.io
QDARNT_API_KEY=<your-api-key>
```

> Note: The `QDARNT_` prefix (with the typo) is intentional — it matches the existing `.env` variable names already in use.

### `config.yaml` — Tunable Parameters

All pipeline parameters live here. No code changes needed to tune the system.

```yaml
ats_weights:
  skills:         0.40   # Increase to prioritise skill match
  experience:     0.30
  projects:       0.15
  education:      0.10
  certifications: 0.05

final_score_weights:
  semantic_similarity: 0.40
  skill_match:         0.30
  experience_score:    0.20
  ats_score:           0.10

retrieval:
  top_k_vector: 20   # Candidates retrieved per method
  top_k_bm25:   20
  top_k_final:   5   # Candidates shown in UI
```

---

## Running the Application

```bash
# From the project root
streamlit run app/main.py
```

Default credentials:
| Username | Password | Role |
|---|---|---|
| `admin` | `hr2025` | Admin |
| `recruiter` | `match123` | Recruiter |

### Workflow

1. **Ingest Resumes** — Upload files or point to `data/resumes/`. The pipeline parses, validates, expands skills, embeds, and stores each resume in Qdrant. Already-indexed files are skipped automatically.
2. **Match Candidates** — Paste or upload a job description. The query pipeline retrieves, reranks, and scores candidates, displaying the top results with full score breakdowns.
3. **Analytics** — View index statistics, collection health, and reset/re-ingest controls.
4. **Dashboard** — Browse search history, compare past runs, and export results to CSV.

---

## Pipeline Details

### Ingestion Pipeline

```
load_documents → checkpoint_filter → parse_resume → validate_schema
                                                              ↓
                                              expand_skills → embed_and_store
```

| Node | What it does |
|---|---|
| `load_documents` | Reads PDF/DOCX/TXT; extracts raw text + metadata |
| `checkpoint_filter` | Queries Qdrant for existing `file_name`s; drops already-indexed docs before LLM |
| `parse_resume` | GPT-4o extracts name, skills (with years), experience, education, certifications. **10 concurrent workers.** |
| `validate_schema` | Pydantic validation. Invalid docs are repaired via LLM (up to 3 retries) then discarded |
| `expand_skills` | Knowledge graph infers parent skills (e.g. FastAPI → Python, Pandas → Python) |
| `embed_and_store` | Builds embedding text, calls Azure Embeddings in batches of 50, upserts to Qdrant |

### Query Pipeline

```
parse_jd → embed_jd → hybrid_retrieve → rerank_candidates
                                                  ↓
                           aggregate_results ← fuse_scores ← score_ats
```

| Node | What it does |
|---|---|
| `parse_jd` | GPT-4o extracts title, required skills (with min years), experience requirements |
| `embed_jd` | Embeds the structured JD into a 1536-dim vector |
| `hybrid_retrieve` | Vector search (top 20) + BM25 keyword search (top 20), fused via RRF |
| `rerank_candidates` | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) scores each JD×candidate pair |
| `score_ats` | Weighted ATS score: skills (40%) + experience (30%) + projects (15%) + education (10%) + certs (5%) |
| `fuse_scores` | Final score: semantic (40%) + skill match (30%) + experience (20%) + ATS (10%) |
| `aggregate_results` | Trims to `top_k_final`, builds `results_summary`, normalises all fields |

---

## Scoring System

### ATS Score

Computed per candidate against the JD's required skills and experience:

```
ATS = 0.40 × skill_match
    + 0.30 × experience_score
    + 0.15 × project_score
    + 0.10 × education_score
    + 0.05 × certification_score
```

`skill_match` gives partial credit: if a candidate has a skill but fewer years than required, they score `candidate_years / required_years`.

### Final Score

```
Final = 0.40 × semantic_similarity   (vector cosine)
      + 0.30 × skill_match_rrf        (RRF-fused rank score)
      + 0.20 × rerank_score           (CrossEncoder, sigmoid-normalised)
      + 0.10 × ats_score
```

All weights are adjustable in `config.yaml` — no code changes required.

---

## Developer Tools

### Inspect Pipeline (`inspect_pipeline.py`)

Run each node individually on a single resume + JD and see the full structured output after every step:

```bash
# Auto-selects first resume and JD found
python inspect_pipeline.py

# Point to specific files
python inspect_pipeline.py --resume data/resumes/alice.pdf --jd data/jd/senior_engineer.txt
```

Output includes: parsed fields, skill expansion diff, Qdrant vector sample, ATS sub-score bars, final ranking table, and per-node latency summary.

### Validate Config

```bash
python config/settings.py
```

Checks all required environment variables are set and all weight groups sum to 1.0.

### Structured Logs

Every node execution is logged to `logs/pipeline.log` in JSON format with node name, status, latency, and extra metadata (stored count, parsed count, etc.).

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `stored=0, skipped_dupes=1` after reset | Use the Analytics page Reset button — it waits for Qdrant Cloud to confirm deletion before re-ingesting |
| `name: "John Doe"` in Qdrant payload | Resume text didn't contain a readable name; pipeline now falls back to filename. Re-ingest after the schema fix. |
| 429 rate limit errors during ingestion | Reduce `_PARSE_WORKERS` in `nodes/parse_resume.py` from 10 to 5 |
| Qdrant timeout errors | Already retried 3× with backoff. Check Qdrant Cloud cluster status. |
| Pydantic serializer warnings in terminal | Suppressed at module level in parse nodes — safe to ignore if they still appear |
| `pythonjsonlogger` import error | Install: `pip install python-json-logger` or the mock in `inspect_pipeline.py` handles it automatically |

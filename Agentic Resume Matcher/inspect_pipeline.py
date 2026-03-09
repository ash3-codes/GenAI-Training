"""
inspect_pipeline.py
-------------------
Run every node one-by-one on a single resume + JD and print
the full structured output to the terminal after each step.

Usage (run from project root):
    python inspect_pipeline.py
    python inspect_pipeline.py --resume data/resumes/alice.pdf --jd data/jd/jd1.txt

If no files are given it auto-picks the first resume and JD it finds.
If data/ is completely empty it creates a built-in sample pair so you
can run the script even before ingesting anything.

Nodes covered (in order):
    1. parse_resume_node
    2. validate_schema_node
    3. expand_skills_node
    4. embed_and_store_node
    5. parse_jd_node
    6. embed_jd_node
    7. score_ats_node
    8. fuse_scores_node
    9. aggregate_results_node

hybrid_retrieve_node and rerank_candidates_node run silently between
nodes 6 and 7 — they are needed to produce reranked_candidates so that
score_ats_node has something to score.
"""

# ── bootstrap: patch pythonjsonlogger with a real formatter so the logger
# handler can call stream.write(str) without a MagicMock crash ──────────────
import sys, types, logging

def _patch_pythonjsonlogger():
    """Install a real no-op module so logging handlers stay functional."""
    pkg = types.ModuleType("pythonjsonlogger")
    sub = types.ModuleType("pythonjsonlogger.json")

    class _NoOpJsonFormatter(logging.Formatter):
        """Drop-in that formats log records as plain text (not JSON).
        Avoids the MagicMock.write() crash while keeping logging functional."""
        def format(self, record):
            return super().format(record)

    sub.JsonFormatter = _NoOpJsonFormatter
    pkg.json = sub
    sys.modules.setdefault("pythonjsonlogger", pkg)
    sys.modules.setdefault("pythonjsonlogger.json", sub)

_patch_pythonjsonlogger()

import warnings
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse, json, time, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── ANSI palette ──────────────────────────────────────────────────────────────
RST = "\033[0m"
BLD = "\033[1m"
DIM = "\033[2m"
CYN = "\033[96m"   # node headers
GRN = "\033[92m"   # keys / success
YLW = "\033[93m"   # values
MAG = "\033[95m"   # section dividers
RED = "\033[91m"   # errors
BLU = "\033[94m"   # sub-labels
W   = 76           # line width


# ── helpers ───────────────────────────────────────────────────────────────────
def section(title, char="─"):
    pad   = max(0, W - len(title) - 4)
    left  = pad // 2
    right = pad - left
    print(f"\n{MAG}{char * left}  {BLD}{title}{RST}{MAG}  {char * right}{RST}")


def node_banner(name, ms):
    ms_str = f"{ms:.0f} ms"
    dots   = max(1, W - len(name) - len(ms_str) - 5)
    print(f"\n{CYN}{BLD}  {name}{RST}  {DIM}{'·' * dots} {ms_str}{RST}")
    print(f"{DIM}{'─' * W}{RST}")


def field(key, val, depth=1):
    pad = "  " * depth
    if isinstance(val, (dict, list)):
        s     = json.dumps(val, indent=2, default=str)
        lines = s.split("\n")
        print(f"{pad}{GRN}{key}{RST}: {YLW}{lines[0]}{RST}")
        for ln in lines[1:]:
            print(f"{pad}  {YLW}{ln}{RST}")
    else:
        s = str(val)
        if len(s) > 110:
            s = s[:107] + "…"
        print(f"{pad}{GRN}{key}{RST}: {YLW}{s}{RST}")


def show_list(label, items, fmt=str, limit=7, depth=1):
    pad = "  " * depth
    if not items:
        print(f"{pad}{GRN}{label}{RST}: {DIM}(empty){RST}")
        return
    print(f"{pad}{GRN}{label}{RST}: {DIM}({len(items)} items){RST}")
    for item in items[:limit]:
        print(f"{pad}  {YLW}{fmt(item)}{RST}")
    if len(items) > limit:
        print(f"{pad}  {DIM}… +{len(items) - limit} more{RST}")


def bar(val, width=22):
    val   = max(0.0, min(1.0, float(val)))
    n     = int(val * width)
    return f"[{BLU}{'█' * n}{DIM}{'·' * (width - n)}{RST}]"


def vec_info(label, vec, depth=1):
    pad = "  " * depth
    if not vec:
        print(f"{pad}{GRN}{label}{RST}: {RED}None / empty{RST}")
        return
    import statistics as _st
    sample = "[" + ", ".join(f"{v:.4f}" for v in vec[:5]) + ", …]"
    print(f"{pad}{GRN}{label}{RST}:")
    print(f"{pad}  {DIM}dims   : {len(vec)}{RST}")
    print(f"{pad}  {DIM}sample : {sample}{RST}")
    print(f"{pad}  {DIM}range  : {min(vec):.4f} … {max(vec):.4f}  "
          f"mean {_st.mean(vec):.4f}{RST}")


# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--resume", default=None, help="Path to resume file (pdf/docx/txt)")
ap.add_argument("--jd",     default=None, help="Path to JD file (txt/pdf)")
args = ap.parse_args()

# ── locate / create sample files ─────────────────────────────────────────────
from config.settings import RESUMES_DIR, JD_DIR

RESUMES_DIR.mkdir(parents=True, exist_ok=True)
JD_DIR.mkdir(parents=True, exist_ok=True)


def _pick(cli_arg, directory, exts, label):
    """Return an existing file or create a built-in sample."""
    if cli_arg:
        p = Path(cli_arg)
        if not p.exists():
            print(f"{RED}File not found: {cli_arg}{RST}")
            sys.exit(1)
        return p
    for ext in exts:
        hits = sorted(directory.glob(f"*{ext}"))
        real = [f for f in hits if "sample_" not in f.name]
        if real:
            print(f"  {DIM}Auto-selected {label}: {real[0].name}{RST}")
            return real[0]
    # nothing found — write a built-in sample
    if label == "resume":
        dest = directory / "sample_resume.txt"
        dest.write_text(SAMPLE_RESUME, encoding="utf-8")
    else:
        dest = directory / "sample_jd.txt"
        dest.write_text(SAMPLE_JD, encoding="utf-8")
    print(f"  {YLW}No {label} found — created built-in sample: {dest.name}{RST}")
    return dest


SAMPLE_RESUME = """
Alice Johnson
alice.johnson@email.com | +1-555-0100 | San Francisco, CA
LinkedIn: linkedin.com/in/alicejohnson | GitHub: github.com/alice

SUMMARY
Senior software engineer with 6 years of experience building scalable
backend systems. Passionate about distributed systems and ML pipelines.

EDUCATION
B.Tech in Computer Science — Stanford University, 2018  GPA: 3.8

SKILLS
Python (5 years), FastAPI (3 years), PostgreSQL (4 years),
Docker (3 years), Kubernetes (2 years), AWS (3 years),
PyTorch (2 years), NumPy (4 years), Pandas (4 years),
Redis (2 years), Kafka (1 year), Git (5 years)

EXPERIENCE
Senior Software Engineer — FinTech Corp (Jan 2022 – Present, 30 months)
  - Built real-time fraud detection pipeline processing 10k events/sec
  - Migrated monolith to microservices reducing latency by 40%
  Technologies: Python, FastAPI, Kafka, PostgreSQL, Docker, Kubernetes

Software Engineer — DataCo (Jul 2019 – Dec 2021, 30 months)
  - Developed ETL pipelines ingesting 50M records/day
  - Built REST APIs serving 5M daily active users
  Technologies: Python, Django, PostgreSQL, AWS, Redis

Junior Developer — StartupXYZ (Jun 2018 – Jun 2019, 12 months)
  - Full-stack features using React + Flask
  Technologies: Python, Flask, React, MySQL

CERTIFICATIONS
AWS Certified Solutions Architect — Associate (2022)
Google Professional Data Engineer (2023)
""".strip()

SAMPLE_JD = """
Job Title: Senior Python Backend Engineer
Company: TechCorp Inc.
Domain: FinTech

About the Role:
We are looking for a Senior Python Backend Engineer with strong
experience in building scalable APIs and data pipelines.

Requirements:
- 5+ years of software engineering experience
- Strong Python skills (4+ years)
- Experience with FastAPI or Django REST Framework (2+ years)
- PostgreSQL or similar relational database (3+ years)
- Docker and container orchestration (Kubernetes preferred)
- Cloud experience (AWS or GCP)
- Experience with message queues (Kafka, RabbitMQ)

Nice to Have:
- Machine learning pipeline experience
- PyTorch or TensorFlow
- Redis caching

Education: Bachelor's degree in Computer Science or related field

Minimum Experience: 5 years
""".strip()


section("PIPELINE NODE INSPECTOR", "═")
print(f"{DIM}Locating files…{RST}")

resume_path = _pick(args.resume, RESUMES_DIR, [".pdf", ".docx", ".txt"], "resume")
jd_path     = _pick(args.jd,     JD_DIR,      [".txt", ".pdf"],          "JD")

print(f"  {GRN}Resume{RST}: {resume_path}")
print(f"  {GRN}JD    {RST}: {jd_path}")

# ── load raw documents ────────────────────────────────────────────────────────
section("LOADING FILES")
from nodes.load_documents import load_single_document

resume_doc = load_single_document(resume_path)
jd_doc     = load_single_document(jd_path)

field("resume.file_name",  resume_doc["file_name"])
field("resume.file_type",  resume_doc["file_type"])
field("resume.char_count", resume_doc["char_count"])
print(f"  {GRN}resume.text preview{RST}:")
for ln in textwrap.wrap(resume_doc["text"][:300], W - 6):
    print(f"      {DIM}{ln}{RST}")
print()
field("jd.file_name",  jd_doc["file_name"])
field("jd.char_count", jd_doc["char_count"])

# shared mutable state dict threaded through every node
state: dict = {
    "raw_resume_texts": [resume_doc],
    "node_logs":        [],
    "failed_docs":      [],
}


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 1 — parse_resume_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 1 / 9  ·  parse_resume_node", "═")
from nodes.parse_resume import parse_resume_node

t0  = time.time()
out = parse_resume_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("parse_resume_node", ms)

parsed_list: list = state.get("parsed_resumes") or []
if not parsed_list:
    failed = state.get("failed_docs") or []
    print(f"{RED}  parse_resume_node returned no results.{RST}")
    for fd in failed:
        print(f"  {RED}{fd}{RST}")
    sys.exit(1)

pr = parsed_list[0]
field("candidate_id",            pr.get("candidate_id"))
field("name",                    pr.get("name"))
field("email",                   pr.get("email"))
field("phone",                   pr.get("phone"))
field("location",                pr.get("location"))
field("total_experience_months", pr.get("total_experience_months"))
field("file_name",               pr.get("file_name"))
show_list("education", pr.get("education") or [],
    fmt=lambda e: (f"{e.get('degree','?')} @ {e.get('university','?')} "
                   f"({e.get('graduation_year','?')})"))
show_list("skills", pr.get("skills") or [],
    fmt=lambda s: f"{str(s.get('skill','?')):<22}  {s.get('experience_years','?')} yr(s)",
    limit=10)
show_list("experience", pr.get("experience") or [],
    fmt=lambda e: (f"{e.get('role','?')} @ "
                   f"{e.get('client') or e.get('project_title','?')} "
                   f"[{e.get('duration_months','?')} mo]"),
    limit=5)
show_list("certifications", pr.get("certifications") or [], limit=5)


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 2 — validate_schema_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 2 / 9  ·  validate_schema_node", "═")
from nodes.validate_schema import validate_schema_node

t0  = time.time()
out = validate_schema_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("validate_schema_node", ms)

validated = state.get("parsed_resumes") or []
rejected  = [f for f in (state.get("failed_docs") or [])
             if f.get("stage") == "validate_schema_node"]

field("input_resumes",     len(parsed_list))
field("validated",         len(validated))
field("rejected",          len(rejected))

if validated:
    v = validated[0]
    print(f"\n  {BLU}Validated resume:{RST}")
    field("candidate_id",  v.get("candidate_id"), depth=2)
    field("name",          v.get("name"),          depth=2)
    field("skills_count",  len(v.get("skills") or []), depth=2)
    field("schema_valid",  "True — all required Pydantic fields present", depth=2)

for r in rejected:
    print(f"  {RED}REJECTED  {r.get('file_name')} — {r.get('reason','')}{RST}")

if not validated:
    print(f"{RED}Nothing validated — cannot continue.{RST}")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 3 — expand_skills_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 3 / 9  ·  expand_skills_node", "═")
from nodes.expand_skills import expand_skills_node

before = [s.get("skill") for s in (validated[0].get("skills") or [])]

t0  = time.time()
out = expand_skills_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("expand_skills_node", ms)

expanded_list: list = state.get("expanded_resumes") or []
after = [s.get("skill") for s in (expanded_list[0].get("skills") or [])] \
        if expanded_list else []
added = [s for s in after if s not in before]

field("skills_before_expansion", len(before))
field("skills_after_expansion",  len(after))
field("skills_added_by_graph",   len(added))

show_list("before", before, limit=10)
print()
show_list("after",  after,  limit=12)

if added:
    print(f"\n  {GRN}Inferred by knowledge graph:{RST}")
    for s in added:
        print(f"    {YLW}+ {s}{RST}")
else:
    print(f"\n  {DIM}  No new skills inferred (base skills already present){RST}")

if not expanded_list:
    print(f"{RED}expand_skills_node returned nothing — cannot continue.{RST}")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 4 — embed_and_store_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 4 / 9  ·  embed_and_store_node", "═")
from nodes.embed_and_store import embed_and_store_node

t0  = time.time()
out = embed_and_store_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("embed_and_store_node", ms)

# NodeTimer appends a log entry — read the latest one
embed_log = next(
    (l for l in reversed(state.get("node_logs") or [])
     if l.get("node") == "embed_and_store_node"),
    {}
)
stored   = embed_log.get("stored_count",   0)
dupes    = embed_log.get("skipped_dupes",  0)
failures = embed_log.get("embed_failures", 0)
to_embed = embed_log.get("resumes_to_embed", 0)

field("resumes_submitted", to_embed)
field("stored_in_qdrant",  stored)
field("skipped_dupes",     dupes)
field("embed_failures",    failures)

# If stored, pull the point back from Qdrant and show its vector
cid = expanded_list[0].get("candidate_id") if expanded_list else None
if stored and cid:
    from config.settings import get_qdrant_client, QDRANT_RESUME_COLLECTION
    import uuid as _uuid
    try:
        pts = get_qdrant_client().retrieve(
            collection_name=QDRANT_RESUME_COLLECTION,
            ids=[str(_uuid.UUID(cid))],
            with_payload=True,
            with_vectors=True,
        )
        if pts:
            pt      = pts[0]
            payload = pt.payload or {}
            print(f"\n  {BLU}Qdrant point confirmed:{RST}")
            field("point_id",    pt.id,                          depth=2)
            field("name",        payload.get("name"),             depth=2)
            field("file_name",   payload.get("file_name"),        depth=2)
            field("skills",      (payload.get("skills") or [])[:8], depth=2)
            field("exp_months",  payload.get("total_experience_months"), depth=2)
            vec_info("vector", pt.vector or [], depth=2)
    except Exception as exc:
        print(f"  {DIM}Could not retrieve stored point: {exc}{RST}")
elif dupes:
    print(f"\n  {YLW}  Duplicate — resume already in index, skipped re-embedding.{RST}")


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 5 — parse_jd_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 5 / 9  ·  parse_jd_node", "═")
from nodes.parse_jd import parse_jd_node

state["jd_raw_text"]  = jd_doc["text"]
state["jd_file_name"] = jd_doc["file_name"]

t0  = time.time()
out = parse_jd_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("parse_jd_node", ms)

pjd: dict = state.get("parsed_jd") or {}
if not pjd:
    print(f"{RED}  parse_jd_node returned nothing.{RST}")
    sys.exit(1)

field("jd_id",                  pjd.get("jd_id"))
field("title",                  pjd.get("title"))
field("company",                pjd.get("company"))
field("domain",                 pjd.get("domain"))
field("experience_min_years",   pjd.get("experience_min_years"))
field("experience_max_years",   pjd.get("experience_max_years"))
field("education_requirement",  pjd.get("education_requirement"))
show_list("required_skills", pjd.get("required_skills") or [],
    fmt=lambda s: f"{str(s.get('skill','?')):<22}  min {s.get('min_years','?')} yr(s)",
    limit=10)
show_list("nice_to_have_skills", pjd.get("nice_to_have_skills") or [],
    fmt=lambda s: s.get("skill", "?") if isinstance(s, dict) else str(s),
    limit=6)


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 6 — embed_jd_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 6 / 9  ·  embed_jd_node", "═")
from nodes.embed_jd import embed_jd_node

t0  = time.time()
out = embed_jd_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("embed_jd_node", ms)

jd_vec = state.get("jd_embedding") or []
vec_info("jd_embedding", jd_vec)


# ── silent: hybrid_retrieve + rerank ─────────────────────────────────────────
section("RUNNING: hybrid_retrieve_node + rerank_candidates_node  (silent)")
from nodes.hybrid_retrieve   import hybrid_retrieve_node
from nodes.rerank_candidates import rerank_candidates_node

state.update(hybrid_retrieve_node(state))
n_ret = len(state.get("retrieved_candidates") or [])
print(f"  {DIM}Retrieved {n_ret} candidate(s) via vector + BM25 + RRF fusion{RST}")

state.update(rerank_candidates_node(state))
n_rr = len(state.get("reranked_candidates") or [])
print(f"  {DIM}Reranked to top {n_rr} candidate(s) via CrossEncoder{RST}")

if n_rr == 0:
    print(f"\n  {YLW}No candidates in index yet — score_ats / fuse / aggregate will show empty results.")
    print(f"  Tip: ingest more resumes first, then re-run this script.{RST}")


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 7 — score_ats_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 7 / 9  ·  score_ats_node", "═")
from nodes.score_ats import (
    score_ats_node,
    score_skills, score_experience, score_education,
    score_certifications, score_projects,
)
from schemas.jd_schema import JobDescriptionSchema

t0  = time.time()
out = score_ats_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("score_ats_node", ms)

ats_scores: dict   = state.get("ats_scores") or {}
reranked:   list   = state.get("reranked_candidates") or []

field("candidates_scored", len(ats_scores))

# Build JD object once for sub-score breakdown
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _clean_jd = {k: v for k, v in pjd.items() if k != "parsed"}
    jd_obj = JobDescriptionSchema(**_clean_jd)

req_skill_map = jd_obj.get_required_skill_map()   # {skill_name: min_years}

from config.settings import ATS_WEIGHTS
print(f"\n  {BLU}ATS weight breakdown:{RST}")
for wk, wv in ATS_WEIGHTS.items():
    print(f"    {GRN}{wk:<18}{RST}: {wv:.0%}")

print()
for cid, total_score in list(ats_scores.items())[:5]:
    cand    = next((c for c in reranked if c.get("candidate_id") == cid), {})
    payload = cand.get("payload") or {}
    name    = payload.get("name") or cand.get("name") or cid[:12] + "…"

    sk_sc  = score_skills(payload, req_skill_map)
    ex_sc  = score_experience(payload, jd_obj.experience_min_years or 0)
    ed_sc  = score_education(payload)
    ce_sc  = score_certifications(payload)
    pr_sc  = score_projects(payload)

    print(f"  {BLD}{name}{RST}")
    print(f"    {GRN}{'ats_total':<18}{RST}  {YLW}{total_score:.4f}{RST}  {bar(total_score)}")
    print(f"    {GRN}{'skill_match':<18}{RST}  {sk_sc:.4f}  {bar(sk_sc)}")
    print(f"    {GRN}{'experience':<18}{RST}  {ex_sc:.4f}  {bar(ex_sc)}")
    print(f"    {GRN}{'education':<18}{RST}  {ed_sc:.4f}  {bar(ed_sc)}")
    print(f"    {GRN}{'certifications':<18}{RST}  {ce_sc:.4f}  {bar(ce_sc)}")
    print(f"    {GRN}{'projects':<18}{RST}  {pr_sc:.4f}  {bar(pr_sc)}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 8 — fuse_scores_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 8 / 9  ·  fuse_scores_node", "═")
from nodes.fuse_scores      import fuse_scores_node
from config.settings        import FINAL_SCORE_WEIGHTS

t0  = time.time()
out = fuse_scores_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("fuse_scores_node", ms)

print(f"  {BLU}Fusion formula:{RST}")
for wk, wv in FINAL_SCORE_WEIGHTS.items():
    print(f"    {GRN}{wk:<30}{RST}: {wv:.0%}")

fused: list = state.get("final_scores") or []
print()
field("candidates_fused", len(fused))
print()

for c in fused[:5]:
    name    = c.get("name") or (c.get("payload") or {}).get("name") or "?"
    fs      = float(c.get("final_score")     or 0)
    sem     = float(c.get("semantic_score")  or 0)
    ats_s   = float(c.get("ats_score")       or 0)
    rr_norm = float(c.get("rerank_score")    or 0)   # already sigmoid-normalised
    rr_raw  = float(c.get("rerank_score_raw") or c.get("rerank_score") or 0)
    bm25    = float(c.get("bm25_score")      or 0)
    rrf     = float(c.get("rrf_score")       or 0)
    rank    = c.get("final_rank") or "?"
    srcs    = c.get("sources") or []

    print(f"  {BLD}#{rank}  {name}{RST}")
    print(f"    {GRN}{'final_score':<22}{RST}  {YLW}{fs:.4f}{RST}  {bar(fs)}")
    print(f"    {GRN}{'semantic_score':<22}{RST}  {sem:.4f}  {bar(sem)}")
    print(f"    {GRN}{'ats_score':<22}{RST}  {ats_s:.4f}  {bar(ats_s)}")
    print(f"    {GRN}{'rerank_score (norm)':<22}{RST}  {rr_norm:.4f}  {bar(rr_norm)}")
    print(f"    {GRN}{'rerank_score (raw)':<22}{RST}  {rr_raw:.4f}")
    print(f"    {GRN}{'bm25_score':<22}{RST}  {bm25:.4f}")
    print(f"    {GRN}{'rrf_score':<22}{RST}  {rrf:.4f}")
    print(f"    {GRN}{'sources':<22}{RST}  {srcs}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
#  NODE 9 — aggregate_results_node
# ═════════════════════════════════════════════════════════════════════════════
section("NODE 9 / 9  ·  aggregate_results_node", "═")
from nodes.aggregate_results import aggregate_results_node

t0  = time.time()
out = aggregate_results_node(state)
ms  = (time.time() - t0) * 1000
state.update(out)
node_banner("aggregate_results_node", ms)

final:   list = state.get("final_scores")    or []
summary: dict = state.get("results_summary") or {}

print(f"  {BLU}results_summary:{RST}")
for k, v in summary.items():
    field(k, f"{v:.3f}" if isinstance(v, float) else v, depth=2)

print(f"\n  {BLU}Final ranked table:{RST}")
hdr = (f"  {'#':<4}  {'Name':<28}  "
       f"{'Final':>6}  {'ATS':>6}  {'Sem':>6}  {'Rerank':>7}  {'Exp':>5}")
print(hdr)
print(f"  {DIM}{'─' * 70}{RST}")

for c in final:
    rank   = c.get("final_rank") or 0
    name   = (c.get("name") or "?")[:26]
    fs     = float(c.get("final_score")    or 0)
    ats_s  = float(c.get("ats_score")      or 0)
    sem    = float(c.get("semantic_score") or 0)
    rr     = float(c.get("rerank_score")   or 0)
    exp_mo = float((c.get("payload") or {}).get("total_experience_months") or 0)
    exp_yr = round(exp_mo / 12, 1)
    col    = YLW if fs > 0.7 else (GRN if fs > 0.5 else DIM)
    print(f"  {col}#{rank:<3}  {name:<28}  "
          f"{fs:>6.3f}  {ats_s:>6.3f}  {sem:>6.3f}  "
          f"{rr:>7.3f}  {exp_yr:>4}y{RST}")


# ═════════════════════════════════════════════════════════════════════════════
#  TIMING SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("TIMING SUMMARY", "═")

logs     = state.get("node_logs") or []
total_ms = 0.0

print(f"  {'Node':<40}  {'Status':<10}  {'Latency':>8}")
print(f"  {DIM}{'─' * 64}{RST}")

for lg in logs:
    nm   = str(lg.get("node")       or "")
    st   = str(lg.get("status")     or "")
    lat  = float(lg.get("latency_ms") or 0)
    total_ms += lat
    sc   = GRN if st == "success" else RED
    print(f"  {nm:<40}  {sc}{st:<10}{RST}  {lat:>7.0f} ms")

print(f"  {DIM}{'─' * 64}{RST}")
print(f"  {'TOTAL':<40}  {'':10}  {total_ms:>7.0f} ms")

failed_docs = [f for f in (state.get("failed_docs") or []) if f.get("stage")]
if failed_docs:
    print(f"\n  {RED}Failed documents ({len(failed_docs)}):{RST}")
    for fd in failed_docs:
        print(f"    {RED}{fd.get('file_name')}  [{fd.get('stage')}]  "
              f"{str(fd.get('reason',''))[:80]}{RST}")
else:
    print(f"\n  {GRN}No failed documents.{RST}")

section("", "═")
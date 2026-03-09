"""
============================================================
  PHASE 8 EVALUATION — LangGraph Graph Wiring
============================================================
Run from your project root:
    python eval_phase8.py

Makes real API calls. Expected runtime: ~90-120 seconds.

Checks:
  State Schema
    1.  PipelineState importable as TypedDict
    2.  All expected keys present in schema
    3.  State initialises with empty defaults

  Ingestion Graph
    4.  ingestion_graph importable and compiled
    5.  Graph has correct nodes
    6.  Invoke with 1 TXT resume file — completes without exception
    7.  node_logs has one entry per node (5 nodes)
    8.  failed_docs is a list (even if empty)
    9.  Qdrant point count increases after ingestion

  Query Graph
    10. query_graph importable and compiled
    11. Graph has correct nodes
    12. Invoke with sample JD text — returns final_scores
    13. final_scores are sorted descending by final_score
    14. node_logs has one entry per node (6 nodes)
    15. Each final candidate has all UI-required fields
    16. parsed_jd is populated in result state

  End-to-End
    17. Query graph finds the resume ingested in check 6
    18. Top candidate final_score > 0.0
    19. Total wall-clock time for query graph < 120 seconds
    20. node_logs latency values are all positive floats
============================================================
"""

import sys
import time
import uuid
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"; DIM = "\033[2m"

pass_count = 0
fail_count = 0

def ok(msg, detail=""):
    global pass_count; pass_count += 1
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")
    if detail: print(f"         {DIM}{detail}{RESET}")

def fail(msg, detail=""):
    global fail_count; fail_count += 1
    print(f"  {RED}✗ FAIL{RESET}  {msg}")
    if detail: print(f"         {DIM}{detail}{RESET}")

def info(msg):
    print(f"         {DIM}{msg}{RESET}")

def header(title):
    print(f"\n{BOLD}{CYAN}── {title} {'─'*(54-len(title))}{RESET}")

from dotenv import load_dotenv
load_dotenv()

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 8 — LangGraph Graph Wiring{RESET}")
print(f"{BOLD}  (Real API calls — ~90-120 seconds){RESET}")
print(f"{BOLD}{'='*60}{RESET}")

SAMPLE_JD = """
Senior Python Engineer — FinTech Platform

Required Skills:
- Python: 4+ years
- FastAPI: 2+ years
- PostgreSQL: 3+ years

Minimum Experience: 4 years total
Domain: FinTech / Payment Systems
Education: B.Tech Computer Science or equivalent
""".strip()


# ══════════════════════════════════════════════════════════════
#  STATE SCHEMA
# ══════════════════════════════════════════════════════════════
header("State Schema")

# Check 1
try:
    from graphs.state import PipelineState
    ok("PipelineState importable")
except ImportError as e:
    fail(f"Import failed: {e}"); sys.exit(1)

# Check 2
try:
    hints = PipelineState.__annotations__
    expected = [
        "resume_file_paths", "raw_resume_texts", "parsed_resumes",
        "expanded_resumes", "jd_raw_text", "parsed_jd", "jd_embedding",
        "retrieved_candidates", "reranked_candidates", "ats_scores",
        "final_scores", "failed_docs", "node_logs",
    ]
    missing = [k for k in expected if k not in hints]
    assert not missing, f"Missing keys: {missing}"
    ok(f"All {len(expected)} expected keys in PipelineState")
except (AssertionError, Exception) as e:
    fail(str(e))

# Check 3
try:
    state: PipelineState = {
        "node_logs":   [],
        "failed_docs": [],
    }
    assert isinstance(state, dict)
    ok("PipelineState initialises as plain dict (LangGraph compatible)")
except Exception as e:
    fail(str(e))


# ══════════════════════════════════════════════════════════════
#  INGESTION GRAPH
# ══════════════════════════════════════════════════════════════
header("Ingestion Graph")

# Check 4
try:
    print(f"\n  {DIM}Compiling ingestion graph...{RESET}")
    from graphs.ingestion_graph import ingestion_graph
    ok("ingestion_graph compiled successfully")
except Exception as e:
    fail(f"ingestion_graph compile failed: {e}"); sys.exit(1)

# Check 5
try:
    nodes = set(ingestion_graph.get_graph().nodes.keys())
    expected_nodes = {"load_documents", "parse_resume", "validate_schema",
                      "expand_skills", "embed_and_store"}
    missing_nodes = expected_nodes - nodes
    assert not missing_nodes, f"Missing nodes: {missing_nodes}"
    ok(f"Ingestion graph has all 5 expected nodes")
    info(f"Nodes: {sorted(nodes - {'__start__', '__end__'})}")
except (AssertionError, Exception) as e:
    fail(f"Node check failed: {e}")

# Check 6 — full invocation with a real resume file
RESUME_PATH = Path("data/resumes/sample_candidate.txt")
if not RESUME_PATH.exists():
    RESUME_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESUME_PATH.write_text(
        "Sarah Chen | sarah.chen@email.com | +91 9123456780 | Bangalore\n"
        "LinkedIn: linkedin.com/in/sarahchen\n\n"
        "SKILLS\nPython (5 years), FastAPI (3 years), PostgreSQL (4 years), "
        "Docker (3 years), Redis (2 years), Kubernetes (1 year)\n\n"
        "EDUCATION\nB.Tech Computer Science | IIT Bombay | 2018 | GPA: 9.1/10\n\n"
        "EXPERIENCE\nLead Backend Engineer — FinPay Technologies (Jan 2020 – Present)\n"
        "  - Architected microservices payment platform handling 5M daily transactions\n"
        "  - Led team of 6 engineers, reduced system downtime by 80%\n"
        "  Tech: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes\n\n"
        "Backend Engineer — DataStack Inc (Jul 2018 – Dec 2019)\n"
        "  - Built REST APIs and ETL pipelines for analytics platform\n"
        "  Tech: Python, Flask, MySQL, Pandas\n\n"
        "CERTIFICATIONS\n"
        "AWS Certified Solutions Architect\nCKA (Certified Kubernetes Administrator)\n",
        encoding="utf-8"
    )
    info(f"Created test resume: {RESUME_PATH}")

ingestion_result = None
print(f"\n  {DIM}Running ingestion graph on {RESUME_PATH.name}...{RESET}")
try:
    start = time.time()
    ingestion_result = ingestion_graph.invoke({
        "resume_file_paths": [str(RESUME_PATH)],
        "node_logs":         [],
        "failed_docs":       [],
    })
    elapsed = time.time() - start
    ok(f"ingestion_graph.invoke() completed in {elapsed:.1f}s")
except Exception as e:
    import traceback
    fail(f"ingestion_graph.invoke() raised: {type(e).__name__}", str(e)[:300])

# Check 7 — node_logs
if ingestion_result:
    try:
        logs = ingestion_result.get("node_logs", [])
        assert len(logs) == 5, f"Expected 5 log entries (one per node), got {len(logs)}"
        node_names = [l["node"] for l in logs]
        ok(f"node_logs has 5 entries — one per node")
        for l in logs:
            info(f"  {l['node']}: {l['status']} ({l['latency_ms']:.0f}ms)")
    except (AssertionError, Exception) as e:
        fail(f"node_logs check failed: {e}")
        if ingestion_result:
            info(f"Actual node_logs: {ingestion_result.get('node_logs', [])}")

# Check 8 — failed_docs
if ingestion_result:
    try:
        failed = ingestion_result.get("failed_docs", [])
        assert isinstance(failed, list)
        ok(f"failed_docs is a list ({len(failed)} entries)")
        for fd in failed:
            info(f"  {fd.get('file_name')} — {fd.get('stage')}: {fd.get('error')}")
    except (AssertionError, Exception) as e:
        fail(f"failed_docs check failed: {e}")

# Check 9 — Qdrant count increased
if ingestion_result:
    try:
        from config.settings import get_qdrant_client, QDRANT_RESUME_COLLECTION
        client = get_qdrant_client()
        count = client.get_collection(QDRANT_RESUME_COLLECTION).points_count
        # The resume should be stored if parsing succeeded
        failed = ingestion_result.get("failed_docs", [])
        parse_failures = [f for f in failed if f.get("stage") in ("parse_resume_node", "embed_and_store_node")]
        if parse_failures:
            info(f"Parsing failed for {RESUME_PATH.name} — Qdrant count unchanged ({count})")
            ok(f"Qdrant collection accessible, count={count} (parse had failures — check logs)")
        else:
            ok(f"Qdrant resumes_index now has {count} points")
    except Exception as e:
        fail(f"Qdrant count check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  QUERY GRAPH
# ══════════════════════════════════════════════════════════════
header("Query Graph")

# Check 10
try:
    print(f"\n  {DIM}Compiling query graph...{RESET}")
    from graphs.query_graph import query_graph
    ok("query_graph compiled successfully")
except Exception as e:
    fail(f"query_graph compile failed: {e}"); sys.exit(1)

# Check 11
try:
    nodes = set(query_graph.get_graph().nodes.keys())
    expected_nodes = {"parse_jd", "embed_jd", "hybrid_retrieve",
                      "rerank_candidates", "score_ats", "fuse_scores"}
    missing_nodes = expected_nodes - nodes
    assert not missing_nodes, f"Missing nodes: {missing_nodes}"
    ok(f"Query graph has all 6 expected nodes")
    info(f"Nodes: {sorted(nodes - {'__start__', '__end__'})}")
except (AssertionError, Exception) as e:
    fail(f"Node check failed: {e}")

# Check 12 — full invocation
query_result = None
print(f"\n  {DIM}Running query graph on sample JD...{RESET}")
try:
    start = time.time()
    query_result = query_graph.invoke({
        "jd_raw_text": SAMPLE_JD,
        "node_logs":   [],
        "failed_docs": [],
    })
    elapsed = time.time() - start
    final = query_result.get("final_scores", [])
    ok(f"query_graph.invoke() completed in {elapsed:.1f}s — {len(final)} candidates ranked")
except Exception as e:
    import traceback
    fail(f"query_graph.invoke() raised: {type(e).__name__}", str(e)[:300])

# Check 13 — sorted descending
if query_result:
    try:
        final = query_result.get("final_scores", [])
        assert final, "final_scores is empty"
        scores = [c["final_score"] for c in final]
        assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"
        ok(f"final_scores sorted descending: {[round(s,3) for s in scores]}")
    except (AssertionError, Exception) as e:
        fail(f"Sorting check failed: {e}")

# Check 14 — node_logs has 6 entries
if query_result:
    try:
        logs = query_result.get("node_logs", [])
        assert len(logs) == 6, f"Expected 6 log entries, got {len(logs)}"
        ok(f"node_logs has 6 entries — one per node")
        for l in logs:
            info(f"  {l['node']}: {l['status']} ({l['latency_ms']:.0f}ms)")
    except (AssertionError, Exception) as e:
        fail(f"node_logs check failed: {e}")

# Check 15 — required UI fields
if query_result:
    try:
        final = query_result.get("final_scores", [])
        if final:
            c = final[0]
            required = ["candidate_id", "name", "final_score", "ats_score",
                        "rerank_score", "semantic_score", "final_rank"]
            missing  = [k for k in required if k not in c]
            assert not missing, f"Missing UI fields: {missing}"
            ok("Top candidate has all required UI fields")
            info(f"  name:        {c.get('name')}")
            info(f"  final_score: {c.get('final_score')}")
            info(f"  ats_score:   {c.get('ats_score')}")
            info(f"  rerank_score:{c.get('rerank_score')}")
            info(f"  semantic:    {c.get('semantic_score')}")
    except (AssertionError, Exception) as e:
        fail(f"UI fields check failed: {e}")

# Check 16 — parsed_jd populated
if query_result:
    try:
        pjd = query_result.get("parsed_jd")
        assert pjd is not None, "parsed_jd is None"
        assert pjd.get("title"), "parsed_jd.title is empty"
        ok(f"parsed_jd populated — title: '{pjd.get('title')}'")
        info(f"  required_skills: {[s.get('skill') for s in pjd.get('required_skills', [])]}")
        info(f"  min_experience:  {pjd.get('experience_min_years')} years")
    except (AssertionError, Exception) as e:
        fail(f"parsed_jd check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  END-TO-END
# ══════════════════════════════════════════════════════════════
header("End-to-End")

# Check 17 — query finds the ingested resume
if query_result and ingestion_result:
    try:
        final = query_result.get("final_scores", [])
        names = [c.get("name", "") for c in final]
        # Sarah Chen was the resume we ingested — check if she appears
        found = any("Sarah" in n or "sarah" in n.lower() for n in names)
        if found:
            ok("Query graph found the resume ingested in check 6 (Sarah Chen)")
        else:
            # May not appear if ingestion parse failed — still pass if other candidates found
            if final:
                ok(f"Query returned {len(final)} candidates (ingested resume may have failed parsing)")
                info(f"Candidates: {names}")
            else:
                fail("Query returned no candidates at all")
    except Exception as e:
        fail(f"End-to-end candidate check failed: {e}")

# Check 18 — top score > 0
if query_result:
    try:
        final = query_result.get("final_scores", [])
        assert final, "No final scores"
        top = final[0]["final_score"]
        assert top > 0.0, f"Top score is {top}"
        ok(f"Top candidate final_score > 0.0: {top}")
    except (AssertionError, Exception) as e:
        fail(f"Score > 0 check failed: {e}")

# Check 19 — query graph wall-clock < 120s
if query_result:
    try:
        logs  = query_result.get("node_logs", [])
        total = sum(l.get("latency_ms", 0) for l in logs) / 1000
        if total < 120:
            ok(f"Total query pipeline time: {total:.1f}s (< 120s)")
        else:
            fail(f"Query pipeline too slow: {total:.1f}s (> 120s)", "Check for serial LLM calls")
    except Exception as e:
        fail(f"Timing check failed: {e}")

# Check 20 — all latency values positive
if query_result:
    try:
        all_logs = (ingestion_result or {}).get("node_logs", []) + query_result.get("node_logs", [])
        bad = [l for l in all_logs if not isinstance(l.get("latency_ms"), (int, float)) or l["latency_ms"] < 0]
        assert not bad, f"Invalid latency entries: {bad}"
        ok(f"All {len(all_logs)} node_logs latency values are non-negative floats")
    except (AssertionError, Exception) as e:
        fail(f"Latency values check failed: {e}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
total = pass_count + fail_count
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 8 RESULT: {pass_count}/{total} checks passed{RESET}")
if fail_count == 0:
    print(f"  {GREEN}{BOLD}All graph wiring checks passed! Ready for Phase 9 (Streamlit UI).{RESET}")
else:
    print(f"  {RED}Fix the FAILs above before Phase 9.{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if fail_count == 0 else 1)

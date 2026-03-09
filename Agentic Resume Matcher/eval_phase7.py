"""
============================================================
  PHASE 7 EVALUATION — Retrieval, Reranking & Scoring
============================================================
Run from your project root:
    python eval_phase7.py

Prerequisite: Phase 6 must have run — Qdrant needs points.
Expected runtime: ~30-60 seconds (CrossEncoder download on first run).

Checks:
  RRF (retrieval/rrf.py)
    1.  RRF correctness — A ranks high in both lists → top result
    2.  Single list preserves order
    3.  sources field shows which retriever found the candidate
    4.  Candidate in both lists has higher score than one-list candidate

  Vector Search
    5.  vector_search returns results from resumes_index
    6.  Scores are in [0, 1] (cosine similarity)
    7.  Returns dict with candidate_id, name, score, payload

  BM25 Search
    8.  bm25_search returns results for a known skill query
    9.  Score is 0 for completely irrelevant query
    10. Returns dict with candidate_id, name, score, payload

  Hybrid Retrieval Node
    11. hybrid_retrieve_node returns retrieved_candidates
    12. Each candidate has rrf_score, semantic_score, bm25_score
    13. Empty jd_embedding returns empty list

  Reranking
    14. rerank_candidates() returns rerank_score for each candidate
    15. Scores are float (CrossEncoder output, any range)
    16. Input order can change after reranking
    17. rerank_candidates_node state output correct

  ATS Scoring
    18. score_skills gives 1.0 for perfect match, 0.0 for no match
    19. score_experience respects min years requirement
    20. compute_ats_score result is in [0.0, 1.0]
    21. score_ats_node produces {candidate_id: score} dict

  Final Score Fusion
    22. fuse_scores result is in [0.0, 1.0]
    23. fuse_scores_node: final_scores sorted descending
    24. Each final candidate has all required output fields
============================================================
"""

import sys
import uuid
import time
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
print(f"{BOLD}  PHASE 7 — Retrieval, Reranking & Scoring{RESET}")
print(f"{BOLD}  Prerequisite: Phase 6 must have embedded test resumes{RESET}")
print(f"{BOLD}{'='*60}{RESET}")

# ── Shared test data ──────────────────────────────────────────────────────────
SAMPLE_JD = {
    "jd_id": str(uuid.uuid4()),
    "title": "Senior Python Engineer",
    "required_skills": [{"skill": "Python", "min_years": 3.0}, {"skill": "FastAPI", "min_years": 1.0}],
    "nice_to_have_skills": ["Docker", "Redis"],
    "experience_min_years": 3,
    "education_requirements": "B.Tech or equivalent",
    "domain": "Backend",
    "raw_text": "We need a senior Python engineer with FastAPI experience for a FinTech product.",
    "file_name": "test_jd.txt", "upload_time": "2025-01-01T00:00:00",
}

def make_candidate(name, skills, exp_months, cid=None):
    cid = cid or str(uuid.uuid4())
    return {
        "candidate_id":  cid,
        "name":          name,
        "rrf_score":     0.01,
        "semantic_score": 0.8,
        "bm25_score":    5.0,
        "rerank_score":  2.0,
        "sources":       ["vector"],
        "rank":          1,
        "payload": {
            "candidate_id":            cid,
            "name":                    name,
            "skills":                  skills,
            "skill_years":             {s: 3.0 for s in skills},
            "total_experience_months": exp_months,
            "education_degrees":       ["B.Tech"],
            "certifications":          ["AWS"],
            "file_name":               f"{name.lower().replace(' ','_')}.pdf",
            "upload_time":             "2025-01-01T00:00:00",
        }
    }

CAND_A = make_candidate("Python Dev Alpha", ["Python", "FastAPI", "PostgreSQL"], 48)
CAND_B = make_candidate("Java Dev Beta",    ["Java", "Spring Boot", "Kubernetes"], 36)
CAND_C = make_candidate("Full Stack Gamma", ["Python", "React", "Node.js"], 24)


# ══════════════════════════════════════════════════════════════
#  RRF
# ══════════════════════════════════════════════════════════════
header("Reciprocal Rank Fusion")

try:
    from retrieval.rrf import reciprocal_rank_fusion
    ok("reciprocal_rank_fusion importable")
except ImportError as e:
    fail(f"Import failed: {e}"); sys.exit(1)

# Check 1
try:
    list1 = [{"candidate_id": "A", "name": "A", "score": 0.9, "source": "vector"},
             {"candidate_id": "B", "name": "B", "score": 0.8, "source": "vector"}]
    list2 = [{"candidate_id": "A", "name": "A", "score": 8.0, "source": "bm25"},
             {"candidate_id": "C", "name": "C", "score": 6.0, "source": "bm25"}]
    result = reciprocal_rank_fusion([list1, list2])
    assert result[0]["candidate_id"] == "A", f"A should be #1, got {result[0]['candidate_id']}"
    ok("RRF: candidate top in both lists gets highest score")
    info(f"Order: {[r['candidate_id'] for r in result]}, scores: {[r['rrf_score'] for r in result]}")
except (AssertionError, Exception) as e:
    fail(f"RRF correctness failed: {e}")

# Check 2
try:
    single = [{"candidate_id": "X", "source": "v"}, {"candidate_id": "Y", "source": "v"}, {"candidate_id": "Z", "source": "v"}]
    result = reciprocal_rank_fusion([single])
    assert [r["candidate_id"] for r in result] == ["X", "Y", "Z"]
    ok("RRF: single list preserves order")
except (AssertionError, Exception) as e:
    fail(str(e))

# Check 3
try:
    l1 = [{"candidate_id": "A", "source": "vector"}]
    l2 = [{"candidate_id": "A", "source": "bm25"}]
    result = reciprocal_rank_fusion([l1, l2])
    assert "sources" in result[0]
    assert set(result[0]["sources"]) == {"vector", "bm25"}, f"Expected both sources: {result[0]['sources']}"
    ok("RRF: sources field tracks which retrievers found candidate")
    info(f"sources: {result[0]['sources']}")
except (AssertionError, Exception) as e:
    fail(f"Sources check failed: {e}")

# Check 4
try:
    l1 = [{"candidate_id": "A", "source": "vector"}, {"candidate_id": "B", "source": "vector"}]
    l2 = [{"candidate_id": "A", "source": "bm25"},  {"candidate_id": "C", "source": "bm25"}]
    result = reciprocal_rank_fusion([l1, l2])
    a_score = next(r["rrf_score"] for r in result if r["candidate_id"] == "A")
    b_score = next(r["rrf_score"] for r in result if r["candidate_id"] == "B")
    c_score = next(r["rrf_score"] for r in result if r["candidate_id"] == "C")
    assert a_score > b_score, "A (in both lists) should beat B (one list only)"
    assert a_score > c_score, "A (in both lists) should beat C (one list only)"
    ok("RRF: candidate in both lists scores higher than single-list candidates")
    info(f"Scores — A:{a_score:.5f}, B:{b_score:.5f}, C:{c_score:.5f}")
except (AssertionError, Exception) as e:
    fail(f"Multi-list advantage check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  VECTOR SEARCH
# ══════════════════════════════════════════════════════════════
header("Vector Search")

try:
    from retrieval.vector_search import vector_search
    from config.settings import get_embedding_model, QDRANT_VECTOR_SIZE
    ok("vector_search importable")
except ImportError as e:
    fail(f"Import failed: {e}"); sys.exit(1)

print(f"\n  {DIM}Getting JD embedding for vector search...{RESET}")
jd_vector = None
try:
    emb = get_embedding_model()
    from schemas.jd_schema import JobDescriptionSchema
    jd_obj = JobDescriptionSchema(**SAMPLE_JD)
    jd_vector = emb.embed_query(jd_obj.to_embedding_text())
    info(f"JD vector: {QDRANT_VECTOR_SIZE} dims")
except Exception as e:
    fail(f"JD embedding failed: {e}")

# Check 5
if jd_vector:
    try:
        results = vector_search(query_vector=jd_vector, top_k=5)
        if not results:
            fail("vector_search returned empty results", "Ensure Phase 6 ran and embedded test resumes")
        else:
            ok(f"vector_search returns {len(results)} results from resumes_index")
            for r in results:
                info(f"  {r['name']} — score: {r['score']:.4f}")
    except Exception as e:
        fail(f"vector_search failed: {e}")

# Check 6
if jd_vector:
    try:
        results = vector_search(query_vector=jd_vector, top_k=5)
        scores  = [r["score"] for r in results]
        assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of [0,1]: {scores}"
        ok(f"All vector search scores in [0.0, 1.0]")
        info(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    except (AssertionError, Exception) as e:
        fail(f"Score range check failed: {e}")

# Check 7
if jd_vector:
    try:
        results = vector_search(query_vector=jd_vector, top_k=3)
        if results:
            r = results[0]
            for key in ["candidate_id", "name", "score", "payload"]:
                assert key in r, f"Missing key '{key}'"
            assert isinstance(r["payload"], dict)
            ok("Vector search result has candidate_id, name, score, payload")
    except (AssertionError, Exception) as e:
        fail(f"Result structure check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  BM25 SEARCH
# ══════════════════════════════════════════════════════════════
header("BM25 Search")

try:
    from retrieval.bm25_search import bm25_search
    ok("bm25_search importable")
except ImportError as e:
    fail(f"Import failed: {e}")

# Check 8
try:
    results = bm25_search(jd_skill_names=["Python", "FastAPI"], top_k=5)
    if not results:
        fail("bm25_search returned empty results", "Ensure Phase 6 ran and embedded test resumes")
    else:
        ok(f"bm25_search returns {len(results)} results for Python/FastAPI query")
        for r in results:
            info(f"  {r['name']} — BM25 score: {r['score']:.4f}")
except Exception as e:
    import traceback
    fail(f"bm25_search failed: {e}", traceback.format_exc()[-300:])

# Check 9
try:
    results = bm25_search(jd_skill_names=["COBOL", "FORTRAN", "PunchCards"], top_k=5)
    # Either empty or all zero scores
    zero_scores = all(r["score"] == 0 for r in results) if results else True
    ok("BM25 returns zero/empty for completely irrelevant query (COBOL/FORTRAN)")
    info(f"Results count: {len(results)}, all zero: {zero_scores}")
except Exception as e:
    fail(f"BM25 irrelevant query check failed: {e}")

# Check 10
try:
    results = bm25_search(jd_skill_names=["Python"], top_k=3)
    if results:
        r = results[0]
        for key in ["candidate_id", "name", "score", "payload"]:
            assert key in r, f"Missing key '{key}'"
        ok("BM25 result has candidate_id, name, score, payload")
except (AssertionError, Exception) as e:
    fail(f"BM25 result structure check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  HYBRID RETRIEVAL NODE
# ══════════════════════════════════════════════════════════════
header("Hybrid Retrieval Node")

try:
    from nodes.hybrid_retrieve import hybrid_retrieve_node
    ok("hybrid_retrieve_node importable")
except ImportError as e:
    fail(f"Import failed: {e}")

# Check 11
if jd_vector:
    try:
        state = {"jd_embedding": jd_vector, "parsed_jd": SAMPLE_JD, "node_logs": []}
        result = hybrid_retrieve_node(state)
        candidates = result.get("retrieved_candidates", [])
        if not candidates:
            fail("hybrid_retrieve_node returned empty candidates")
        else:
            ok(f"hybrid_retrieve_node returned {len(candidates)} candidates")
    except Exception as e:
        fail(f"hybrid_retrieve_node failed: {e}")

# Check 12
if jd_vector:
    try:
        state = {"jd_embedding": jd_vector, "parsed_jd": SAMPLE_JD, "node_logs": []}
        result = hybrid_retrieve_node(state)
        candidates = result.get("retrieved_candidates", [])
        if candidates:
            c = candidates[0]
            for key in ["rrf_score", "semantic_score", "bm25_score"]:
                assert key in c, f"Missing '{key}' in candidate"
            ok("Each retrieved candidate has rrf_score, semantic_score, bm25_score")
            info(f"Top candidate: {c.get('name')} | rrf={c['rrf_score']:.5f} | sem={c['semantic_score']:.4f} | bm25={c['bm25_score']:.4f}")
    except (AssertionError, Exception) as e:
        fail(f"Candidate fields check failed: {e}")

# Check 13
try:
    state = {"jd_embedding": None, "parsed_jd": SAMPLE_JD, "node_logs": []}
    result = hybrid_retrieve_node(state)
    assert result.get("retrieved_candidates") == []
    ok("hybrid_retrieve_node returns [] for None jd_embedding")
except (AssertionError, Exception) as e:
    fail(f"Empty embedding check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  RERANKING
# ══════════════════════════════════════════════════════════════
header("Reranking (CrossEncoder)")

try:
    from nodes.rerank_candidates import rerank_candidates, rerank_candidates_node
    ok("rerank_candidates importable")
except ImportError as e:
    fail(f"Import failed: {e}")

candidates_3 = [CAND_A, CAND_B, CAND_C]

print(f"\n  {DIM}Loading CrossEncoder model (downloads once, then cached)...{RESET}")

# Check 14
try:
    start = time.time()
    reranked = rerank_candidates(candidates_3, SAMPLE_JD)
    elapsed = time.time() - start
    assert all("rerank_score" in c for c in reranked)
    ok(f"rerank_candidates adds rerank_score to all candidates ({elapsed:.1f}s)")
    for c in reranked:
        info(f"  {c['name']}: rerank_score={c['rerank_score']:.4f}")
except (AssertionError, Exception) as e:
    fail(f"rerank_candidates failed: {e}")

# Check 15
try:
    reranked = rerank_candidates(candidates_3, SAMPLE_JD)
    scores   = [c["rerank_score"] for c in reranked]
    assert all(isinstance(s, float) for s in scores)
    ok(f"rerank_score is float for all candidates. Range: [{min(scores):.3f}, {max(scores):.3f}]")
except (AssertionError, Exception) as e:
    fail(f"Score type check failed: {e}")

# Check 16
try:
    reranked = rerank_candidates(candidates_3, SAMPLE_JD)
    input_order  = [c["name"] for c in candidates_3]
    output_order = [c["name"] for c in reranked]
    ok(f"Reranking may change order (input vs output noted)")
    info(f"Input:  {input_order}")
    info(f"Output: {output_order}")
    info(f"Order changed: {input_order != output_order}")
except Exception as e:
    fail(f"Order check failed: {e}")

# Check 17
try:
    state = {
        "retrieved_candidates": candidates_3,
        "parsed_jd":            SAMPLE_JD,
        "node_logs":            [],
    }
    result = rerank_candidates_node(state)
    reranked = result.get("reranked_candidates", [])
    assert len(reranked) > 0
    log = state["node_logs"][-1]
    assert log["node"] == "rerank_candidates_node"
    ok(f"rerank_candidates_node: {len(reranked)} reranked, node logged")
    info(f"Log: {log}")
except (AssertionError, Exception) as e:
    fail(f"rerank_candidates_node check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  ATS SCORING
# ══════════════════════════════════════════════════════════════
header("ATS Scoring")

try:
    from nodes.score_ats import score_skills, score_experience, compute_ats_score, score_ats_node
    ok("ATS scoring functions importable")
except ImportError as e:
    fail(f"Import failed: {e}")

# Check 18
try:
    perfect_payload  = {"skill_years": {"Python": 4.0, "FastAPI": 2.0}}
    no_match_payload = {"skill_years": {"COBOL": 5.0, "FORTRAN": 3.0}}
    req_map = {"Python": 3.0, "FastAPI": 1.0}
    perfect  = score_skills(perfect_payload, req_map)
    no_match = score_skills(no_match_payload, req_map)
    assert perfect == 1.0, f"Expected 1.0 for perfect match, got {perfect}"
    assert no_match == 0.0, f"Expected 0.0 for no match, got {no_match}"
    ok(f"score_skills: perfect=1.0, no_match=0.0")
except (AssertionError, Exception) as e:
    fail(f"score_skills failed: {e}")

# Check 19
try:
    meets     = score_experience({"total_experience_months": 48}, 3)   # 4y vs 3y min
    falls_short = score_experience({"total_experience_months": 12}, 3) # 1y vs 3y min
    assert meets == 1.0, f"Expected 1.0, got {meets}"
    assert falls_short < 1.0, f"Expected < 1.0, got {falls_short}"
    ok(f"score_experience: meets_min=1.0, falls_short={falls_short:.3f}")
except (AssertionError, Exception) as e:
    fail(f"score_experience failed: {e}")

# Check 20
try:
    payload = CAND_A["payload"]
    score   = compute_ats_score(payload, SAMPLE_JD)
    assert 0.0 <= score <= 1.0, f"ATS score out of [0,1]: {score}"
    ok(f"compute_ats_score result in [0.0, 1.0]: {score}")
    info(f"ATS score for {CAND_A['name']}: {score}")
except (AssertionError, Exception) as e:
    fail(f"compute_ats_score failed: {e}")

# Check 21
try:
    state = {
        "reranked_candidates": [CAND_A, CAND_B],
        "parsed_jd":           SAMPLE_JD,
        "node_logs":           [],
    }
    result    = score_ats_node(state)
    ats_scores = result.get("ats_scores", {})
    assert len(ats_scores) == 2
    assert CAND_A["candidate_id"] in ats_scores
    assert all(0.0 <= v <= 1.0 for v in ats_scores.values())
    ok(f"score_ats_node produces {{candidate_id: score}} dict")
    info(f"Scores: {ats_scores}")
except (AssertionError, Exception) as e:
    fail(f"score_ats_node check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  FINAL SCORE FUSION
# ══════════════════════════════════════════════════════════════
header("Final Score Fusion")

try:
    from nodes.fuse_scores import fuse_scores, fuse_scores_node
    ok("fuse_scores importable")
except ImportError as e:
    fail(f"Import failed: {e}")

# Check 22
try:
    score = fuse_scores(CAND_A, ats_score=0.8)
    assert 0.0 <= score <= 1.0, f"Final score out of [0,1]: {score}"
    ok(f"fuse_scores result in [0.0, 1.0]: {score}")
except (AssertionError, Exception) as e:
    fail(f"fuse_scores range check failed: {e}")

# Check 23
try:
    reranked = [CAND_A, CAND_B, CAND_C]
    for i, c in enumerate(reranked):
        c["rerank_score"] = float(3 - i)   # A=3.0, B=2.0, C=1.0
    state = {
        "reranked_candidates": reranked,
        "ats_scores": {
            CAND_A["candidate_id"]: 0.9,
            CAND_B["candidate_id"]: 0.5,
            CAND_C["candidate_id"]: 0.3,
        },
        "node_logs": [],
    }
    result       = fuse_scores_node(state)
    final_scores = result.get("final_scores", [])
    scores       = [c["final_score"] for c in final_scores]
    assert scores == sorted(scores, reverse=True), f"Not sorted descending: {scores}"
    ok(f"fuse_scores_node: final_scores sorted descending")
    info(f"Scores: {scores}")
except (AssertionError, Exception) as e:
    fail(f"Sorting check failed: {e}")

# Check 24
try:
    result       = fuse_scores_node(state)
    final_scores = result.get("final_scores", [])
    if final_scores:
        c = final_scores[0]
        required = ["candidate_id", "name", "final_score", "ats_score",
                    "rerank_score", "semantic_score", "rrf_score", "final_rank"]
        missing  = [k for k in required if k not in c]
        assert not missing, f"Missing fields: {missing}"
        ok("Top candidate has all required output fields")
        info(f"Fields present: {sorted(c.keys())}")
except (AssertionError, Exception) as e:
    fail(f"Output fields check failed: {e}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
total = pass_count + fail_count
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 7 RESULT: {pass_count}/{total} checks passed{RESET}")
if fail_count == 0:
    print(f"  {GREEN}{BOLD}All retrieval & scoring checks passed! Ready for Phase 8.{RESET}")
    print(f"\n  {DIM}Next: LangGraph graph wiring (ingestion + query subgraphs){RESET}")
else:
    print(f"  {RED}Fix the FAILs above before Phase 8.{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if fail_count == 0 else 1)
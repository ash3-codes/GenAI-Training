"""
============================================================
  PHASE 6 EVALUATION — Embedding & Qdrant Storage
============================================================
Run from your project root:
    python eval_phase6.py

Makes real API calls to AzureOpenAI + Qdrant.
Expected runtime: ~30-45 seconds.

Checks:
  Collection Setup
    1.  ensure_resume_collection creates resumes_index
    2.  ensure_jd_collection creates jd_index
    3.  Both collections have correct vector size (1536) and cosine distance
    4.  Re-running ensure_* is idempotent (no error if already exists)

  embed_and_store_resumes()
    5.  Embeds 2 sample resumes without error
    6.  Qdrant point count increases by 2
    7.  Upsert is idempotent — running again keeps count at 2
    8.  Stored point has correct payload keys
    9.  Payload skills list is filterable (list of strings)
    10. Retrieved point ID matches candidate_id

  embed_and_store_node (LangGraph)
    11. Node runs on 2 expanded_resumes, updates failed_docs
    12. node_logs entry has correct structure

  embed_jd_node
    13. embed_jd_node importable
    14. Embeds sample JD, returns 1536-dim vector in state
    15. Empty parsed_jd returns None without crashing
    16. Vector is list of floats (not None, not empty)
============================================================
"""

import sys
import uuid
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

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

# Suppress Pydantic serializer warnings from LangChain throughout this eval
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 6 — Embedding & Qdrant Storage{RESET}")
print(f"{BOLD}  (Real API calls — ~30-45 seconds){RESET}")
print(f"{BOLD}{'='*60}{RESET}")

# ── Sample data ───────────────────────────────────────────────────────────────

def make_resume(name: str, skills: list, role: str, file_name: str) -> dict:
    """Build a minimal valid ResumeSchema dict for testing."""
    return {
        "candidate_id":  str(uuid.uuid4()),
        "name":          name,
        "email":         f"{name.lower().replace(' ', '.')}@test.com",
        "phone":         None, "linkedin": None, "github": None, "location": "Bangalore",
        "skills":        [{"skill": s, "experience_years": float(i+2)} for i, s in enumerate(skills)],
        "education":     [{"degree": "B.Tech", "university": "IIT Test", "graduation_year": 2020, "gpa": None}],
        "experience":    [{"role": role, "summary": f"Worked on {role.lower()} projects using {skills[0]}",
                           "project_title": None, "client": None, "duration_months": 24, "technologies": skills}],
        "certifications": [],
        "file_name":     file_name,
        "upload_time":   "2025-01-01T00:00:00",
        "total_experience_months": 24,
    }

RESUME_1 = make_resume("Test Candidate Alpha", ["Python", "FastAPI", "PostgreSQL"], "Backend Engineer", "test_alpha.pdf")
RESUME_2 = make_resume("Test Candidate Beta",  ["Java", "Spring Boot", "Kubernetes"], "Java Developer",  "test_beta.pdf")

SAMPLE_JD = {
    "jd_id": str(uuid.uuid4()),
    "title": "Senior Python Engineer",
    "required_skills": [{"skill": "Python", "min_years": 3.0}, {"skill": "FastAPI", "min_years": 1.0}],
    "nice_to_have_skills": ["Docker", "Redis"],
    "experience_min_years": 3,
    "education_requirements": "B.Tech or equivalent",
    "domain": "Backend",
    "raw_text": "We need a senior Python engineer with FastAPI experience.",
    "file_name": "test_jd.txt",
    "upload_time": "2025-01-01T00:00:00",
}

# ── Load dependencies ─────────────────────────────────────────────────────────
try:
    from config.settings import (
        get_qdrant_client, get_embedding_model,
        QDRANT_RESUME_COLLECTION, QDRANT_JD_COLLECTION, QDRANT_VECTOR_SIZE,
    )
    from nodes.embed_and_store import (
        embed_and_store_resumes, embed_and_store_node,
        ensure_resume_collection, ensure_jd_collection,
    )
    from nodes.embed_jd import embed_jd_node
    ok("All modules imported")
except ImportError as e:
    fail(f"Import failed: {e}")
    sys.exit(1)

client = get_qdrant_client()

# ══════════════════════════════════════════════════════════════
#  COLLECTION SETUP
# ══════════════════════════════════════════════════════════════
header("Collection Setup")

# Check 1
try:
    ensure_resume_collection(client, QDRANT_RESUME_COLLECTION, QDRANT_VECTOR_SIZE)
    colls = [c.name for c in client.get_collections().collections]
    assert QDRANT_RESUME_COLLECTION in colls, f"Collection {QDRANT_RESUME_COLLECTION} not found"
    ok(f"resumes_index collection exists: {QDRANT_RESUME_COLLECTION}")
except (AssertionError, Exception) as e:
    fail(f"ensure_resume_collection failed: {e}")

# Check 2
try:
    ensure_jd_collection(client, QDRANT_JD_COLLECTION, QDRANT_VECTOR_SIZE)
    colls = [c.name for c in client.get_collections().collections]
    assert QDRANT_JD_COLLECTION in colls
    ok(f"jd_index collection exists: {QDRANT_JD_COLLECTION}")
except (AssertionError, Exception) as e:
    fail(f"ensure_jd_collection failed: {e}")

# Check 3 — correct vector params
try:
    info_obj = client.get_collection(QDRANT_RESUME_COLLECTION)
    dim  = info_obj.config.params.vectors.size
    dist = info_obj.config.params.vectors.distance
    assert dim == QDRANT_VECTOR_SIZE, f"Expected {QDRANT_VECTOR_SIZE}, got {dim}"
    assert "Cosine" in str(dist), f"Expected Cosine, got {dist}"
    ok(f"resumes_index: {dim}-dim, {dist} distance")
except (AssertionError, Exception) as e:
    fail(f"Collection params check failed: {e}")

# Check 4 — idempotent
try:
    ensure_resume_collection(client, QDRANT_RESUME_COLLECTION, QDRANT_VECTOR_SIZE)
    ensure_resume_collection(client, QDRANT_RESUME_COLLECTION, QDRANT_VECTOR_SIZE)
    ok("ensure_resume_collection is idempotent (no error on re-run)")
except Exception as e:
    fail(f"Idempotency check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  EMBED AND STORE
# ══════════════════════════════════════════════════════════════
header("embed_and_store_resumes()")

# Get baseline count before our test inserts
try:
    baseline = client.get_collection(QDRANT_RESUME_COLLECTION).points_count
    info(f"Baseline point count in {QDRANT_RESUME_COLLECTION}: {baseline}")
except Exception:
    baseline = 0

# Check 5 — embed 2 resumes
print(f"\n  {DIM}Embedding 2 sample resumes...{RESET}")
stored_count = 0
try:
    start = time.time()
    stored_count, failures = embed_and_store_resumes([RESUME_1, RESUME_2])
    elapsed = time.time() - start
    assert stored_count == 2, f"Expected 2 stored, got {stored_count}"
    assert len(failures) == 0, f"Unexpected failures: {failures}"
    ok(f"Embedded and stored 2 resumes in {elapsed:.1f}s")
    info(f"stored_count={stored_count}, failures={len(failures)}")
except (AssertionError, Exception) as e:
    fail(f"embed_and_store_resumes failed: {e}")

# Check 6 — point count increased
try:
    new_count = client.get_collection(QDRANT_RESUME_COLLECTION).points_count
    expected = baseline + 2
    assert new_count >= expected, f"Expected at least {expected} points, got {new_count}"
    ok(f"Qdrant point count increased: {baseline} → {new_count}")
except (AssertionError, Exception) as e:
    fail(f"Point count check failed: {e}")

# Check 7 — upsert is idempotent
try:
    count_before = client.get_collection(QDRANT_RESUME_COLLECTION).points_count
    embed_and_store_resumes([RESUME_1, RESUME_2])  # Re-run same resumes
    count_after  = client.get_collection(QDRANT_RESUME_COLLECTION).points_count
    assert count_after == count_before, \
        f"Upsert should be idempotent: count was {count_before}, now {count_after}"
    ok(f"Upsert is idempotent — count unchanged at {count_after}")
except (AssertionError, Exception) as e:
    fail(f"Idempotency check failed: {e}")

# Check 8 — payload keys
try:
    point_id = str(uuid.UUID(RESUME_1["candidate_id"]))
    results  = client.retrieve(
        collection_name=QDRANT_RESUME_COLLECTION,
        ids=[point_id],
        with_payload=True,
    )
    assert results, f"No point found for ID {point_id}"
    payload = results[0].payload
    required_keys = ["candidate_id", "name", "skills", "skill_years",
                     "total_experience_months", "file_name", "upload_time"]
    missing = [k for k in required_keys if k not in payload]
    assert not missing, f"Missing payload keys: {missing}"
    ok("Stored point has all required payload keys")
    info(f"name: {payload['name']}")
    info(f"skills: {payload['skills']}")
    info(f"skill_years: {payload['skill_years']}")
except (AssertionError, Exception) as e:
    fail(f"Payload check failed: {e}")

# Check 9 — skills is list of strings (filterable)
try:
    results = client.retrieve(
        collection_name=QDRANT_RESUME_COLLECTION,
        ids=[str(uuid.UUID(RESUME_1["candidate_id"]))],
        with_payload=True,
    )
    skills = results[0].payload.get("skills", [])
    assert isinstance(skills, list), f"Expected list, got {type(skills)}"
    assert all(isinstance(s, str) for s in skills), f"Skills should be strings: {skills}"
    ok("Payload skills is a list of strings (Qdrant-filterable)")
    info(f"Skills: {skills}")
except (AssertionError, Exception) as e:
    fail(f"Skills type check failed: {e}")

# Check 10 — retrieved point ID matches candidate_id
try:
    point_id = str(uuid.UUID(RESUME_2["candidate_id"]))
    results  = client.retrieve(
        collection_name=QDRANT_RESUME_COLLECTION,
        ids=[point_id],
        with_payload=True,
    )
    assert results, "Point not found"
    payload_cid = results[0].payload.get("candidate_id")
    assert payload_cid == RESUME_2["candidate_id"], \
        f"ID mismatch: Qdrant has {payload_cid}, expected {RESUME_2['candidate_id']}"
    ok("Retrieved point ID matches candidate_id in payload")
    info(f"candidate_id: {payload_cid}")
except (AssertionError, Exception) as e:
    fail(f"ID match check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  LANGGRAPH NODE
# ══════════════════════════════════════════════════════════════
header("embed_and_store_node (LangGraph)")

# Check 11
try:
    state = {
        "expanded_resumes": [RESUME_1, RESUME_2],
        "failed_docs":      [{"file_name": "pre-existing", "error": "test", "reason": "was already here"}],
        "node_logs":        [],
    }
    result = embed_and_store_node(state)
    assert "failed_docs" in result
    # Pre-existing entry preserved
    assert any(fd["file_name"] == "pre-existing" for fd in result["failed_docs"]), \
        "Pre-existing failed_docs entry was lost"
    ok("embed_and_store_node runs, preserves pre-existing failed_docs")
    info(f"failed_docs count: {len(result['failed_docs'])} (includes pre-existing)")
except (AssertionError, Exception) as e:
    fail(f"embed_and_store_node test failed: {e}")

# Check 12 — node_logs
try:
    assert len(state["node_logs"]) >= 1
    log = state["node_logs"][-1]
    assert log["node"] == "embed_and_store_node"
    assert log["status"] == "success"
    assert "stored_count" in log
    ok("node_logs entry has correct structure")
    info(f"Log: {log}")
except (AssertionError, Exception) as e:
    fail(f"node_logs check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  JD EMBEDDING
# ══════════════════════════════════════════════════════════════
header("embed_jd_node")

# Check 13
try:
    from nodes.embed_jd import embed_jd_node
    ok("embed_jd_node importable")
except ImportError as e:
    fail(f"Import failed: {e}")

# Check 14 — real embedding
print(f"\n  {DIM}Embedding sample JD...{RESET}")
try:
    start = time.time()
    state = {"parsed_jd": SAMPLE_JD, "node_logs": []}
    result = embed_jd_node(state)
    elapsed = time.time() - start
    vec = result.get("jd_embedding")
    assert vec is not None, "jd_embedding is None"
    assert len(vec) == QDRANT_VECTOR_SIZE, f"Expected {QDRANT_VECTOR_SIZE} dims, got {len(vec)}"
    ok(f"JD embedded in {elapsed:.1f}s — {len(vec)}-dim vector")
except (AssertionError, Exception) as e:
    fail(f"embed_jd_node failed: {e}")

# Check 15 — empty parsed_jd
try:
    state = {"parsed_jd": None, "node_logs": []}
    result = embed_jd_node(state)
    assert result.get("jd_embedding") is None
    ok("embed_jd_node handles None parsed_jd gracefully")
except (AssertionError, Exception) as e:
    fail(f"Empty JD handling failed: {e}")

# Check 16 — vector is list of floats
try:
    state = {"parsed_jd": SAMPLE_JD, "node_logs": []}
    result = embed_jd_node(state)
    vec = result.get("jd_embedding", [])
    assert isinstance(vec, list) and len(vec) > 0
    assert all(isinstance(v, float) for v in vec[:5]), \
        f"First 5 values should be floats: {vec[:5]}"
    ok("JD embedding vector is a list of floats")
    info(f"First 3 values: {[round(v, 4) for v in vec[:3]]}")
except (AssertionError, Exception) as e:
    fail(f"Vector type check failed: {e}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
total = pass_count + fail_count
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 6 RESULT: {pass_count}/{total} checks passed{RESET}")
if fail_count == 0:
    print(f"  {GREEN}{BOLD}All embedding checks passed! Ready for Phase 7.{RESET}")
    print(f"\n  {DIM}Qdrant now has test points in resumes_index.")
    print(f"  These will be used in Phase 7 (retrieval) testing.{RESET}")
else:
    print(f"  {RED}Fix the FAILs above before Phase 7.{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if fail_count == 0 else 1)

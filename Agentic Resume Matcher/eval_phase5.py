"""
============================================================
  PHASE 5 EVALUATION — Resume Parsing, Validation, JD Parsing
============================================================
Run from your project root:
    python eval_phase5.py

This phase makes REAL AzureOpenAI API calls.
Expected runtime: ~60-90 seconds (6 LLM calls total).

Checks:
  Resume Parsing (parse_resume_node)
    1.  parse_resume_node importable
    2.  parse_resume_text() returns valid dict for a sample resume
    3.  Parsed dict contains all required ResumeSchema fields
    4.  candidate_id is a fresh UUID (not from LLM)
    5.  file_name is preserved from input (not overwritten by LLM)
    6.  Skills extracted (at least 1 skill found)
    7.  Experience entries extracted
    8.  parse_resume_node state: failed_docs accumulates on error

  Validation (validate_schema_node)
    9.  validate_schema_node importable
    10. Valid dict passes validation without retry
    11. Dict with string experience_years gets repaired and passes
    12. Dict missing name goes to failed_docs (unrecoverable)
    13. Multiple invalid dicts: all end up in failed_docs
    14. Timing: validation of 1 valid doc completes < 5 seconds

  JD Parsing (parse_jd_node)
    15. parse_jd_node importable
    16. parse_jd_text() returns valid dict for a sample JD
    17. required_skills extracted with min_years
    18. experience_min_years is an integer
    19. to_embedding_text() works on parsed result
    20. Empty JD text returns None without crashing
============================================================
"""

import sys
import uuid
import time
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

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 5 — Resume Parsing + Validation + JD Parsing{RESET}")
print(f"{BOLD}  (Makes real API calls — ~60-90 seconds){RESET}")
print(f"{BOLD}{'='*60}{RESET}")

# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_RESUME_TEXT = """
John Smith
john.smith@email.com | +91 9876543210 | Bangalore, India
LinkedIn: linkedin.com/in/johnsmith | GitHub: github.com/jsmith

SKILLS
Python (4 years), FastAPI (2 years), PostgreSQL (3 years),
Docker (2 years), Redis (1.5 years), Git (4 years)

EDUCATION
B.Tech Computer Science | IIT Delhi | 2019 | GPA: 8.7/10

EXPERIENCE
Senior Backend Engineer — TechPay Pvt Ltd (Jan 2021 – Present)
  - Designed and built payment processing REST APIs serving 2M+ daily transactions
  - Reduced P99 API latency from 800ms to 120ms using Redis caching
  - Led migration from monolith to microservices architecture
  Tech: Python, FastAPI, PostgreSQL, Redis, Docker

Backend Engineer — DataCorp India (Jun 2019 – Dec 2020)
  - Built internal ETL pipelines processing 5GB daily data
  - Developed RESTful APIs for internal analytics dashboard
  Tech: Python, Flask, MySQL, Pandas

CERTIFICATIONS
AWS Certified Developer - Associate
PostgreSQL Professional Certificate
""".strip()

SAMPLE_JD_TEXT = """
Senior Python Engineer — FinTech Platform

We are looking for a Senior Python Engineer to join our payments team.

Required Skills:
- Python: 4+ years
- FastAPI or Django: 2+ years
- PostgreSQL or MySQL: 3+ years
- Docker and containerization: 2+ years

Nice to Have:
- Kubernetes
- Redis
- AWS (EC2, RDS, Lambda)
- Experience with payment systems

Requirements:
- 4+ years of total software engineering experience
- B.Tech in Computer Science or equivalent
- Strong understanding of REST API design principles

Domain: FinTech / Payment Systems
""".strip()


# ══════════════════════════════════════════════════════════════
#  RESUME PARSING
# ══════════════════════════════════════════════════════════════
header("Resume Parsing (parse_resume_node)")

# Check 1
try:
    from nodes.parse_resume import parse_resume_node, parse_resume_text
    ok("parse_resume_node and parse_resume_text importable")
except ImportError as e:
    fail(f"Import failed: {e}")
    sys.exit(1)

# Check 2 — actual LLM call
parsed_result = None
print(f"\n  {DIM}Calling AzureOpenAI to parse sample resume...{RESET}")
try:
    start = time.time()
    parsed_result = parse_resume_text(
        resume_text=SAMPLE_RESUME_TEXT,
        file_name="john_smith.txt",
        upload_time="2025-01-01T00:00:00",
    )
    elapsed = time.time() - start
    ok(f"parse_resume_text() succeeded (took {elapsed:.1f}s)")
    info(f"Parsed name:    {parsed_result.get('name')}")
    info(f"Skills count:   {len(parsed_result.get('skills', []))}")
    info(f"Exp entries:    {len(parsed_result.get('experience', []))}")
    info(f"Education:      {len(parsed_result.get('education', []))} entries")
    info(f"Certifications: {parsed_result.get('certifications')}")
except Exception as e:
    fail(f"parse_resume_text() failed: {type(e).__name__}", str(e)[:200])

# Check 3 — required fields
if parsed_result:
    try:
        required = ["candidate_id", "name", "skills", "education", "experience",
                    "certifications", "file_name", "upload_time"]
        missing = [k for k in required if k not in parsed_result]
        assert not missing, f"Missing fields: {missing}"
        ok("Parsed dict contains all required ResumeSchema fields")
    except AssertionError as e:
        fail(str(e))
else:
    fail("Skipped — parse_resume_text() failed")

# Check 4 — candidate_id is fresh UUID
if parsed_result:
    try:
        cid = parsed_result.get("candidate_id", "")
        uuid.UUID(cid)   # Raises if invalid
        # Run again to confirm it's different
        result2 = parse_resume_text(
            resume_text="Alice Wong | alice@email.com\nSkills: Java 3y\nExperience: Java Dev 2 years",
            file_name="alice.txt",
            upload_time="2025-01-01T00:00:00",
        )
        assert parsed_result["candidate_id"] != result2["candidate_id"], "IDs should be unique per parse"
        ok(f"candidate_id is a fresh unique UUID4 (not from LLM)")
        info(f"ID: {cid}")
    except (ValueError, AssertionError) as e:
        fail(f"candidate_id check failed: {e}")

# Check 5 — file_name preserved
if parsed_result:
    try:
        assert parsed_result["file_name"] == "john_smith.txt", \
            f"Expected john_smith.txt, got {parsed_result['file_name']}"
        ok("file_name preserved from input (not overwritten by LLM)")
    except AssertionError as e:
        fail(str(e))

# Check 6 — skills extracted
if parsed_result:
    try:
        skills = parsed_result.get("skills", [])
        assert len(skills) >= 1, f"Expected at least 1 skill, got {len(skills)}"
        skill_names = [s.get("skill") for s in skills]
        ok(f"Skills extracted: {len(skills)} skills found")
        info(f"Skills: {skill_names}")
    except AssertionError as e:
        fail(str(e))

# Check 7 — experience entries
if parsed_result:
    try:
        exp = parsed_result.get("experience", [])
        assert len(exp) >= 1, f"Expected at least 1 experience entry, got {len(exp)}"
        ok(f"Experience entries extracted: {len(exp)} entries")
        for e in exp:
            info(f"  Role: {e.get('role')} | Duration: {e.get('duration_months')} months")
    except AssertionError as e:
        fail(str(e))

# Check 8 — node-level: failed_docs accumulates on error
try:
    state = {
        "raw_resume_texts": [
            {
                "file_name":   "bad_file.txt",
                "text":        "",   # Empty text — structured output will likely fail or produce minimal result
                "upload_time": "2025-01-01T00:00:00",
            }
        ],
        "failed_docs": [{"file_name": "previous_fail.txt", "error": "test", "reason": "pre-existing"}],
        "node_logs":   [],
    }
    # We don't assert failure here since LLM might return minimal valid output for empty text
    # Instead verify state structure is always correct
    result = parse_resume_node(state)
    assert "parsed_resumes" in result
    assert "failed_docs" in result
    # Pre-existing failed doc must be preserved
    assert any(fd["file_name"] == "previous_fail.txt" for fd in result["failed_docs"]), \
        "Pre-existing failed_docs entry was lost"
    ok("parse_resume_node state structure correct, pre-existing failed_docs preserved")
    info(f"parsed_resumes: {len(result['parsed_resumes'])}, failed_docs: {len(result['failed_docs'])}")
except Exception as e:
    fail(f"Node state structure check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════
header("Validation (validate_schema_node)")

# Check 9
try:
    from nodes.validate_schema import validate_schema_node, validate_and_repair
    ok("validate_schema_node and validate_and_repair importable")
except ImportError as e:
    fail(f"Import failed: {e}")

# Check 10 — valid dict passes without retry
try:
    import uuid as _uuid
    valid_dict = {
        "candidate_id":  str(_uuid.uuid4()),
        "name":          "Valid Candidate",
        "email":         "valid@email.com",
        "phone":         None,
        "linkedin":      None,
        "github":        None,
        "location":      "Delhi",
        "skills":        [{"skill": "Python", "experience_years": 3.0}],
        "education":     [{"degree": "B.Tech", "university": "IIT", "graduation_year": 2020, "gpa": None}],
        "experience":    [{"role": "Dev", "summary": "Built APIs", "project_title": None,
                          "client": None, "duration_months": 24, "technologies": []}],
        "certifications": [],
        "file_name":     "valid.pdf",
        "upload_time":   "2025-01-01T00:00:00",
    }
    start = time.time()
    result, error = validate_and_repair(valid_dict, max_retries=3, backoff_seconds=0)
    elapsed = time.time() - start
    assert result is not None, f"Valid dict was rejected: {error}"
    assert error is None
    ok(f"Valid dict passes validation without retry ({elapsed*1000:.0f}ms)")
except (AssertionError, Exception) as e:
    fail(f"Valid dict validation failed: {e}")

# Check 11 — string experience_years gets repaired
try:
    repairable_dict = {
        "candidate_id":  str(_uuid.uuid4()),
        "name":          "Repairable Candidate",
        "skills":        [{"skill": "Java", "experience_years": "5"}],  # string, not float
        "education":     [],
        "experience":    [{"role": "Dev", "summary": "Java development"}],
        "certifications": [],
        "file_name":     "repair.pdf",
        "upload_time":   "2025-01-01T00:00:00",
    }
    # Pydantic v2 actually coerces "5" → 5.0 for float fields, so this might pass directly
    result, error = validate_and_repair(repairable_dict, max_retries=3, backoff_seconds=0)
    if result is not None:
        ok("Dict with string experience_years validated (Pydantic coerced it)")
        info(f"Skills: {result.get('skills')}")
    else:
        fail(f"Repairable dict rejected: {error}")
except Exception as e:
    fail(f"Repair test failed: {e}")

# Check 12 — missing name → unrecoverable
try:
    no_name_dict = {
        "candidate_id":  str(_uuid.uuid4()),
        "name":          "",   # Empty name — unrecoverable
        "skills":        [],
        "education":     [],
        "experience":    [],
        "certifications": [],
        "file_name":     "noname.pdf",
        "upload_time":   "2025-01-01T00:00:00",
    }
    result, error = validate_and_repair(no_name_dict, max_retries=3, backoff_seconds=0)
    assert result is None, "Empty name should be unrecoverable"
    assert "name" in error.lower() or "unrecoverable" in error.lower(), \
        f"Error message should mention name: {error}"
    ok("Dict with empty name correctly marked unrecoverable")
    info(f"Reason: {error}")
except (AssertionError, Exception) as e:
    fail(f"Missing name test failed: {e}")

# Check 13 — multiple invalid dicts all go to failed_docs
try:
    bad_dicts = [
        {"name": "",   "skills": [], "education": [], "experience": [],
         "certifications": [], "file_name": f"bad{i}.pdf", "upload_time": "2025-01-01T00:00:00"}
        for i in range(3)
    ]
    state = {"parsed_resumes": bad_dicts, "failed_docs": [], "node_logs": []}
    result = validate_schema_node(state)
    assert len(result["parsed_resumes"]) == 0, \
        f"Expected 0 valid resumes, got {len(result['parsed_resumes'])}"
    assert len(result["failed_docs"]) == 3, \
        f"Expected 3 failed docs, got {len(result['failed_docs'])}"
    ok("All 3 invalid dicts (empty name) correctly moved to failed_docs")
except (AssertionError, Exception) as e:
    fail(f"Multiple invalid dicts test failed: {e}")

# Check 14 — timing: 1 valid doc < 5 seconds
try:
    import uuid as _uuid2
    one_valid = [{
        "candidate_id":  str(_uuid2.uuid4()),
        "name":          "Timing Test",
        "skills":        [{"skill": "Python", "experience_years": 2.0}],
        "education":     [],
        "experience":    [{"role": "Dev", "summary": "Python dev work"}],
        "certifications": [],
        "file_name":     "timing.pdf",
        "upload_time":   "2025-01-01T00:00:00",
    }]
    state = {"parsed_resumes": one_valid, "failed_docs": [], "node_logs": []}
    start = time.time()
    result = validate_schema_node(state)
    elapsed = time.time() - start
    assert len(result["parsed_resumes"]) == 1
    if elapsed < 5.0:
        ok(f"Validation of 1 valid doc completes in {elapsed:.2f}s (< 5s)")
    else:
        fail(f"Validation took {elapsed:.2f}s (expected < 5s)", "Check for unnecessary LLM calls during validation")
except (AssertionError, Exception) as e:
    fail(f"Timing check failed: {e}")


# ══════════════════════════════════════════════════════════════
#  JD PARSING
# ══════════════════════════════════════════════════════════════
header("JD Parsing (parse_jd_node)")

# Check 15
try:
    from nodes.parse_jd import parse_jd_node, parse_jd_text
    ok("parse_jd_node and parse_jd_text importable")
except ImportError as e:
    fail(f"Import failed: {e}")

parsed_jd = None
print(f"\n  {DIM}Calling AzureOpenAI to parse sample JD...{RESET}")

# Check 16 — actual LLM call
try:
    start = time.time()
    parsed_jd = parse_jd_text(
        jd_text=SAMPLE_JD_TEXT,
        file_name="jd_python_senior.txt",
        upload_time="2025-01-01T00:00:00",
    )
    elapsed = time.time() - start
    ok(f"parse_jd_text() succeeded (took {elapsed:.1f}s)")
    info(f"Title:            {parsed_jd.get('title')}")
    info(f"Required skills:  {len(parsed_jd.get('required_skills', []))}")
    info(f"Nice to have:     {parsed_jd.get('nice_to_have_skills')}")
    info(f"Domain:           {parsed_jd.get('domain')}")
    info(f"Min experience:   {parsed_jd.get('experience_min_years')} years")
except Exception as e:
    fail(f"parse_jd_text() failed: {type(e).__name__}", str(e)[:200])

# Check 17 — required_skills with min_years
if parsed_jd:
    try:
        req = parsed_jd.get("required_skills", [])
        assert len(req) >= 1, f"Expected at least 1 required skill, got {len(req)}"
        for s in req:
            assert "skill" in s, f"Skill entry missing 'skill': {s}"
            assert "min_years" in s, f"Skill entry missing 'min_years': {s}"
        ok(f"required_skills extracted with min_years ({len(req)} skills)")
        for s in req:
            info(f"  {s['skill']}: {s['min_years']}y minimum")
    except (AssertionError, Exception) as e:
        fail(f"required_skills check failed: {e}")

# Check 18 — experience_min_years is int
if parsed_jd:
    try:
        exp_min = parsed_jd.get("experience_min_years")
        assert isinstance(exp_min, int), f"Expected int, got {type(exp_min).__name__}: {exp_min}"
        assert exp_min >= 0
        ok(f"experience_min_years is int: {exp_min}")
    except AssertionError as e:
        fail(str(e))

# Check 19 — to_embedding_text works
if parsed_jd:
    try:
        from schemas.jd_schema import JobDescriptionSchema
        jd_obj = JobDescriptionSchema(**parsed_jd)
        text = jd_obj.to_embedding_text()
        assert len(text) > 20
        assert jd_obj.title in text
        ok("to_embedding_text() works on parsed JD result")
        info(f"Preview: {text[:120]}...")
    except Exception as e:
        fail(f"to_embedding_text() failed: {e}")

# Check 20 — empty JD text returns None
try:
    state = {"jd_raw_text": "   ", "node_logs": []}
    result = parse_jd_node(state)
    assert result.get("parsed_jd") is None, \
        f"Expected None for empty JD, got: {result.get('parsed_jd')}"
    ok("Empty JD text returns None without crashing")
except (AssertionError, Exception) as e:
    fail(f"Empty JD handling failed: {e}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
total = pass_count + fail_count
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 5 RESULT: {pass_count}/{total} checks passed{RESET}")
if fail_count == 0:
    print(f"  {GREEN}{BOLD}All parsing checks passed! Ready for Phase 6.{RESET}")
else:
    print(f"  {RED}Fix the FAILs above before Phase 6.{RESET}")
print(f"\n  {DIM}Tip: If you see LangSmith 403 errors printed above the results,")
print(f"  those are harmless — add LANGCHAIN_TRACING_V2=false to .env to suppress.{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if fail_count == 0 else 1)

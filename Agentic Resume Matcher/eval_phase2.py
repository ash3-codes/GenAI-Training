"""
============================================================
  PHASE 2 EVALUATION — Pydantic Schemas
============================================================
Run from your project root:
    python eval_phase2.py

Checks:
  ResumeSchema
    1.  Full valid resume validates without error
    2.  All optional fields accept None / missing
    3.  experience_years coerced to float (int/str input)
    4.  total_experience_months auto-computed correctly
    5.  candidate_id auto-generated as UUID4 if not provided
    6.  Invalid skills list type raises ValidationError
    7.  to_qdrant_payload() returns flat filterable dict
    8.  to_embedding_dict() returns correct structure

  JobDescriptionSchema
    9.  Full valid JD validates without error
    10. required_skills with min_years=0 defaults work
    11. get_required_skill_map() returns {skill: min_years}
    12. get_required_skill_names() returns list of strings
    13. to_embedding_text() produces non-empty string with title
    14. experience_min_years coerced from string

  SkillEntry / EducationEntry / ExperienceEntry
    15. SkillEntry rejects negative experience_years
    16. ExperienceEntry coerces duration_months from float string
    17. EducationEntry graduation_year bounds enforced

  EmbedTemplate
    18. build_resume_embedding_text() contains all key fields
    19. Resume with no skills/experience produces valid (not empty) text
    20. Output is a single string (no newlines breaking it)
============================================================
"""

import sys
import uuid
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
print(f"{BOLD}  PHASE 2 — Schema Validation Evaluation{RESET}")
print(f"{BOLD}{'='*60}{RESET}")


# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from schemas.resume_schema import ResumeSchema, SkillEntry, EducationEntry, ExperienceEntry
    from schemas.jd_schema import JobDescriptionSchema, RequiredSkillEntry
    from utils.embed_template import build_resume_embedding_text
    from pydantic import ValidationError
    ok("All schema modules imported")
except ImportError as e:
    fail(f"Import failed: {e}")
    sys.exit(1)


# ── RESUME SCHEMA ─────────────────────────────────────────────────────────────
header("ResumeSchema")

FULL_RESUME = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "+91 9876543210",
    "linkedin": "linkedin.com/in/janedoe",
    "github": "github.com/janedoe",
    "location": "Bangalore, India",
    "skills": [
        {"skill": "Python", "experience_years": 4.0},
        {"skill": "FastAPI", "experience_years": 2.0},
        {"skill": "PostgreSQL", "experience_years": 3.0},
    ],
    "education": [
        {"degree": "B.Tech", "university": "IIT Delhi", "graduation_year": 2020, "gpa": 8.5}
    ],
    "experience": [
        {
            "role": "Backend Engineer",
            "summary": "Built REST APIs payment processing microservices",
            "project_title": "PayFlow",
            "client": "FinCorp",
            "duration_months": 24,
            "technologies": ["Python", "FastAPI", "PostgreSQL"],
        },
        {
            "role": "Junior Developer",
            "summary": "Developed internal tooling automation scripts",
            "duration_months": 12,
        },
    ],
    "certifications": ["AWS Certified Developer", "Python Professional"],
    "file_name": "jane_doe.pdf",
    "upload_time": "2025-01-01T00:00:00",
}

# Check 1
try:
    r = ResumeSchema(**FULL_RESUME)
    ok("Full valid resume validates successfully")
    info(f"Name: {r.name}, Skills: {len(r.skills)}, Exp entries: {len(r.experience)}")
except ValidationError as e:
    fail("Full valid resume validation failed", str(e))

# Check 2
try:
    minimal = ResumeSchema(name="John", file_name="john.pdf", upload_time="2025-01-01T00:00:00")
    assert minimal.email is None
    assert minimal.phone is None
    assert minimal.skills == []
    assert minimal.experience == []
    ok("Minimal resume (all optionals omitted) validates correctly")
except (ValidationError, AssertionError) as e:
    fail("Minimal resume failed", str(e))

# Check 3
try:
    s_int  = SkillEntry(skill="Python", experience_years=3)
    s_str  = SkillEntry(skill="Java", experience_years="2.5")
    s_none = SkillEntry(skill="Go", experience_years=None)
    assert isinstance(s_int.experience_years, float)
    assert s_str.experience_years == 2.5
    assert s_none.experience_years == 0.0
    ok("experience_years coerced correctly (int/str/None → float)")
    info(f"int→{s_int.experience_years}, str→{s_str.experience_years}, None→{s_none.experience_years}")
except (ValidationError, AssertionError) as e:
    fail("experience_years coercion failed", str(e))

# Check 4
try:
    r = ResumeSchema(**FULL_RESUME)
    expected = 24 + 12  # 36 months
    assert r.total_experience_months == expected, f"Expected {expected}, got {r.total_experience_months}"
    ok(f"total_experience_months auto-computed correctly ({r.total_experience_months} months)")
except (ValidationError, AssertionError) as e:
    fail("total_experience_months computation failed", str(e))

# Check 5
try:
    r1 = ResumeSchema(name="Auto ID", file_name="x.pdf", upload_time="2025-01-01T00:00:00")
    r2 = ResumeSchema(name="Auto ID 2", file_name="y.pdf", upload_time="2025-01-01T00:00:00")
    assert r1.candidate_id != r2.candidate_id, "IDs should be unique"
    uuid.UUID(r1.candidate_id)  # Raises if not valid UUID
    ok(f"candidate_id auto-generated as unique UUID4")
    info(f"Sample ID: {r1.candidate_id}")
except (ValidationError, AssertionError, ValueError) as e:
    fail("candidate_id auto-generation failed", str(e))

# Check 6
try:
    bad = ResumeSchema(
        name="Bad", file_name="b.pdf", upload_time="2025-01-01T00:00:00",
        skills="NOT_A_LIST"   # Wrong type
    )
    fail("Should have raised ValidationError for skills='NOT_A_LIST'")
except ValidationError:
    ok("Invalid skills type (string) correctly raises ValidationError")

# Check 7
try:
    r = ResumeSchema(**FULL_RESUME)
    payload = r.to_qdrant_payload()
    required_keys = ["candidate_id", "name", "skills", "skill_years",
                     "total_experience_months", "file_name", "upload_time"]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        fail(f"to_qdrant_payload() missing keys: {missing}")
    else:
        ok("to_qdrant_payload() returns all required keys")
        info(f"Skills in payload: {payload['skills']}")
        info(f"Skill years: {payload['skill_years']}")
except Exception as e:
    fail(f"to_qdrant_payload() error: {e}")

# Check 8
try:
    r = ResumeSchema(**FULL_RESUME)
    d = r.to_embedding_dict()
    assert "skills" in d and isinstance(d["skills"], list)
    assert "experience" in d and isinstance(d["experience"], list)
    assert "candidate_id" in d
    ok("to_embedding_dict() returns correct structure")
except (AssertionError, Exception) as e:
    fail(f"to_embedding_dict() failed: {e}")


# ── JD SCHEMA ────────────────────────────────────────────────────────────────
header("JobDescriptionSchema")

FULL_JD = {
    "title": "Senior Python Engineer",
    "required_skills": [
        {"skill": "Python", "min_years": 4.0},
        {"skill": "FastAPI", "min_years": 2.0},
        {"skill": "PostgreSQL", "min_years": 2.0},
    ],
    "nice_to_have_skills": ["Docker", "Kubernetes", "Redis"],
    "experience_min_years": 4,
    "education_requirements": "B.Tech or equivalent",
    "domain": "FinTech",
    "raw_text": "We are hiring a Senior Python Engineer for our FinTech product...",
    "file_name": "jd_python_senior.txt",
    "upload_time": "2025-01-01T00:00:00",
}

# Check 9
try:
    jd = JobDescriptionSchema(**FULL_JD)
    ok("Full valid JD validates successfully")
    info(f"Title: {jd.title}, Required skills: {len(jd.required_skills)}")
except ValidationError as e:
    fail("Full JD validation failed", str(e))

# Check 10
try:
    jd = JobDescriptionSchema(
        title="Junior Dev",
        raw_text="We need a junior developer.",
        required_skills=[{"skill": "Python"}],   # min_years omitted → should default to 0.0
    )
    assert jd.required_skills[0].min_years == 0.0
    ok("required_skills with omitted min_years defaults to 0.0")
except (ValidationError, AssertionError) as e:
    fail("Default min_years failed", str(e))

# Check 11
try:
    jd = JobDescriptionSchema(**FULL_JD)
    skill_map = jd.get_required_skill_map()
    assert skill_map["Python"] == 4.0
    assert skill_map["FastAPI"] == 2.0
    ok("get_required_skill_map() returns {skill: min_years}")
    info(f"Skill map: {skill_map}")
except (AssertionError, Exception) as e:
    fail(f"get_required_skill_map() failed: {e}")

# Check 12
try:
    jd = JobDescriptionSchema(**FULL_JD)
    names = jd.get_required_skill_names()
    assert "Python" in names
    assert isinstance(names, list)
    ok("get_required_skill_names() returns list of strings")
    info(f"Skill names: {names}")
except (AssertionError, Exception) as e:
    fail(f"get_required_skill_names() failed: {e}")

# Check 13
try:
    jd = JobDescriptionSchema(**FULL_JD)
    text = jd.to_embedding_text()
    assert len(text) > 20, "Embedding text too short"
    assert jd.title in text, "Title missing from embedding text"
    assert "Python" in text, "Required skill missing from embedding text"
    ok("to_embedding_text() produces valid, content-rich string")
    info(f"Preview: {text[:120]}...")
except (AssertionError, Exception) as e:
    fail(f"to_embedding_text() failed: {e}")

# Check 14
try:
    jd = JobDescriptionSchema(
        title="Dev",
        raw_text="Some JD text",
        experience_min_years="3",   # String input — should coerce
    )
    assert isinstance(jd.experience_min_years, int)
    assert jd.experience_min_years == 3
    ok("experience_min_years coerced from string '3' to int 3")
except (ValidationError, AssertionError) as e:
    fail("experience_min_years coercion failed", str(e))


# ── FIELD-LEVEL VALIDATORS ────────────────────────────────────────────────────
header("Field Validators")

# Check 15
try:
    bad_skill = SkillEntry(skill="Python", experience_years=-1)
    fail("Should have rejected negative experience_years")
except ValidationError:
    ok("SkillEntry rejects negative experience_years (ge=0.0 enforced)")

# Check 16
try:
    e = ExperienceEntry(role="Dev", summary="Built things", duration_months="18.0")
    assert e.duration_months == 18
    ok("ExperienceEntry coerces duration_months from float-string '18.0' → int 18")
except (ValidationError, AssertionError) as e:
    fail(f"ExperienceEntry duration_months coercion failed: {e}")

# Check 17
try:
    bad_edu = EducationEntry(degree="B.Tech", university="XYZ", graduation_year=1800)
    fail("Should have rejected graduation_year=1800 (< 1950)")
except ValidationError:
    ok("EducationEntry rejects graduation_year=1800 (ge=1950 enforced)")


# ── EMBED TEMPLATE ────────────────────────────────────────────────────────────
header("Embedding Text Template")

# Check 18
try:
    r = ResumeSchema(**FULL_RESUME)
    text = build_resume_embedding_text(r)
    for expected in ["Jane Doe", "Python", "B.Tech", "AWS Certified Developer", "Backend Engineer"]:
        assert expected in text, f"'{expected}' missing from embedding text"
    ok("build_resume_embedding_text() contains all key fields")
    info(f"Full text:\n         {text}")
except (AssertionError, Exception) as e:
    fail(f"Embedding text check failed: {e}")

# Check 19
try:
    empty_r = ResumeSchema(name="No Data", file_name="empty.pdf", upload_time="2025-01-01T00:00:00")
    text = build_resume_embedding_text(empty_r)
    assert len(text) > 10, "Even empty resume should produce some text"
    assert "No Data" in text
    ok("Empty resume produces valid non-empty embedding text")
    info(f"Text: {text}")
except (AssertionError, Exception) as e:
    fail(f"Empty resume embedding failed: {e}")

# Check 20
try:
    r = ResumeSchema(**FULL_RESUME)
    text = build_resume_embedding_text(r)
    assert isinstance(text, str)
    # Should be one continuous string (newlines would break embedding quality)
    assert "\n" not in text, "Embedding text should not contain newlines"
    ok("Embedding text is a single continuous string (no newlines)")
except (AssertionError, Exception) as e:
    fail(f"Embedding text format check failed: {e}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
total = pass_count + fail_count
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 2 RESULT: {pass_count}/{total} checks passed{RESET}")
if fail_count == 0:
    print(f"  {GREEN}{BOLD}All schema checks passed! Ready for Phase 3.{RESET}")
else:
    print(f"  {RED}Fix the FAILs above before Phase 3.{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if fail_count == 0 else 1)

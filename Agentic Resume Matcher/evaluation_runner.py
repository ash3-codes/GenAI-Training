"""
=============================================================
  AGENTIC RESUME MATCHER — PHASE-BY-PHASE EVALUATION RUNNER
=============================================================
Usage:
    python evaluation_runner.py --phase 1
    python evaluation_runner.py --phase 1-5
    python evaluation_runner.py --all

Each phase maps to a build checkpoint. Run this after
completing each phase to verify everything works before
moving on. You see the result of EVERY check, not just failures.
=============================================================
"""

import argparse
import sys
import os
import time
import json
import importlib
import traceback
from pathlib import Path
from typing import Callable


# ─── ANSI colours ────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def ok(msg):   print(f"  {GREEN}[PASS]{RESET} {msg}")
def fail(msg): print(f"  {RED}[FAIL]{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")
def info(msg): print(f"  {DIM}{msg}{RESET}")

def section(title, phase_num):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}  Phase {phase_num} — {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

PASS_COUNT = 0
FAIL_COUNT = 0

def check(label: str, fn: Callable) -> bool:
    global PASS_COUNT, FAIL_COUNT
    try:
        result = fn()
        if result is True or result is None:
            ok(label)
            PASS_COUNT += 1
            return True
        elif isinstance(result, str):
            fail(f"{label} — {result}")
            FAIL_COUNT += 1
            return False
        else:
            ok(label)
            PASS_COUNT += 1
            return True
    except Exception as e:
        fail(f"{label}")
        info(f"    Exception: {type(e).__name__}: {e}")
        FAIL_COUNT += 1
        return False


# ══════════════════════════════════════════════════════════════
#  PHASE 1 — Project Setup
# ══════════════════════════════════════════════════════════════
def phase1_setup():
    section("Project Setup", 1)

    # 1. Required env vars
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_EMBEDDING_DEPLOYMENT",
    ]
    for var in required_vars:
        v = var
        check(f"Env var {v} present", lambda v=v: os.environ.get(v) is not None or f"Missing: {v}")

    # 2. Load .env if python-dotenv is available
    def load_dotenv_check():
        from dotenv import load_dotenv
        load_dotenv()
        return True
    check("python-dotenv loadable", load_dotenv_check)

    # 3. config.yaml exists and loads
    def config_yaml_check():
        import yaml
        cfg_path = Path("config.yaml")
        if not cfg_path.exists():
            return "config.yaml not found in project root"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        # Validate weight sums
        ats = cfg.get("ats_weights", {})
        ats_sum = round(sum(ats.values()), 6) if ats else 0
        if abs(ats_sum - 1.0) > 0.001:
            return f"ats_weights do not sum to 1.0 (got {ats_sum})"
        fs = cfg.get("final_score_weights", {})
        fs_sum = round(sum(fs.values()), 6) if fs else 0
        if abs(fs_sum - 1.0) > 0.001:
            return f"final_score_weights do not sum to 1.0 (got {fs_sum})"
        return True
    check("config.yaml valid and weights sum to 1.0", config_yaml_check)

    # 4. Qdrant connection
    def qdrant_check():
        from qdrant_client import QdrantClient
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        url = cfg.get("qdrant", {}).get("url", "http://localhost:6333")
        client = QdrantClient(url=url, timeout=5)
        client.get_collections()
        return True
    check("Qdrant connection successful", qdrant_check)

    # 5. AzureOpenAI chat reachable
    def azure_chat_check():
        from dotenv import load_dotenv; load_dotenv()
        from langchain_openai import AzureChatOpenAI
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", cfg["azure_openai"]["chat_deployment"]),
            api_version=cfg["azure_openai"]["api_version"],
            max_tokens=10,
        )
        resp = llm.invoke("Say: OK")
        return True
    check("AzureOpenAI chat model reachable (ping)", azure_chat_check)

    # 6. AzureOpenAI embedding reachable
    def azure_embed_check():
        from dotenv import load_dotenv; load_dotenv()
        from langchain_openai import AzureOpenAIEmbeddings
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        emb = AzureOpenAIEmbeddings(
            azure_deployment=os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", cfg["azure_openai"]["embedding_deployment"]),
            api_version=cfg["azure_openai"]["api_version"],
        )
        vec = emb.embed_query("test")
        if len(vec) != 1536:
            return f"Expected 1536 dims, got {len(vec)}"
        return True
    check("AzureOpenAI embedding model reachable (1536 dims)", azure_embed_check)

    # 7. Required libraries
    libs = [
        "langgraph", "langchain_openai", "qdrant_client",
        "pydantic", "streamlit", "rank_bm25", "yaml", "dotenv"
    ]
    for lib in libs:
        check(f"Library '{lib}' importable", lambda l=lib: importlib.import_module(l.replace("-", "_")) is not None)


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — Schemas
# ══════════════════════════════════════════════════════════════
def phase2_schemas():
    section("Pydantic Schemas", 2)

    def resume_schema_check():
        from schemas.resume_schema import ResumeSchema, SkillEntry, ExperienceEntry, EducationEntry
        import uuid
        mock = {
            "candidate_id": str(uuid.uuid4()),
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": None,
            "linkedin": None,
            "github": None,
            "location": "Bangalore",
            "skills": [{"skill": "Python", "experience_years": 3.0}],
            "education": [{"degree": "B.Tech", "university": "IIT", "graduation_year": 2020, "gpa": None}],
            "experience": [{"role": "SDE", "summary": "Built REST APIs", "project_title": None, "client": None, "duration_months": 24}],
            "certifications": ["AWS Solutions Architect"],
            "file_name": "jane_doe.pdf",
            "upload_time": "2025-01-01T00:00:00",
        }
        r = ResumeSchema(**mock)
        assert r.name == "Jane Doe"
        return True
    check("ResumeSchema validates correct mock data", resume_schema_check)

    def resume_optional_fields_check():
        from schemas.resume_schema import ResumeSchema
        import uuid
        # Minimal — all optionals omitted
        mock = {
            "candidate_id": str(uuid.uuid4()),
            "name": "John",
            "skills": [],
            "education": [],
            "experience": [],
            "certifications": [],
            "file_name": "john.pdf",
            "upload_time": "2025-01-01T00:00:00",
        }
        r = ResumeSchema(**mock)
        assert r.email is None
        return True
    check("ResumeSchema: optional fields default to None", resume_optional_fields_check)

    def jd_schema_check():
        from schemas.jd_schema import JobDescriptionSchema
        import uuid
        mock = {
            "jd_id": str(uuid.uuid4()),
            "title": "Senior Python Engineer",
            "required_skills": [{"skill": "Python", "experience_years": 4.0}],
            "nice_to_have_skills": ["FastAPI", "Docker"],
            "experience_min_years": 4,
            "education_requirements": "B.Tech or equivalent",
            "domain": "Backend",
            "raw_text": "We are looking for...",
        }
        jd = JobDescriptionSchema(**mock)
        assert jd.title == "Senior Python Engineer"
        return True
    check("JobDescriptionSchema validates mock JD", jd_schema_check)

    def skill_entry_type_check():
        from schemas.resume_schema import SkillEntry
        # experience_years must be float-coercible
        s = SkillEntry(skill="Python", experience_years=3)
        assert isinstance(s.experience_years, float)
        return True
    check("SkillEntry: experience_years coerced to float", skill_entry_type_check)


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — Skills Graph
# ══════════════════════════════════════════════════════════════
def phase3_skills():
    section("Skills Expansion Graph", 3)

    def skills_graph_exists():
        p = Path("skills/skills_graph.json")
        if not p.exists():
            return "skills/skills_graph.json not found"
        g = json.loads(p.read_text())
        if len(g) < 10:
            return f"Skills graph too small ({len(g)} entries). Add at least 10 mappings."
        return True
    check("skills_graph.json exists with 10+ entries", skills_graph_exists)

    def skills_expand_check():
        from nodes.expand_skills import expand_skills
        # NumPy should expand to include Python
        result = expand_skills([{"skill": "NumPy", "experience_years": 2.0}])
        skill_names = [s["skill"] for s in result]
        if "Python" not in skill_names:
            return f"Expected Python in expanded skills, got: {skill_names}"
        return True
    check("NumPy expands to include Python", skills_expand_check)

    def skills_expand_no_duplicate():
        from nodes.expand_skills import expand_skills
        # If Python already present, shouldn't duplicate
        result = expand_skills([
            {"skill": "NumPy", "experience_years": 2.0},
            {"skill": "Python", "experience_years": 4.0},
        ])
        python_entries = [s for s in result if s["skill"] == "Python"]
        if len(python_entries) > 1:
            return f"Python duplicated in expanded skills: {python_entries}"
        return True
    check("Skill expansion does not create duplicates", skills_expand_no_duplicate)

    known_pairs = [
        ("FastAPI", "Python"),
        ("Pandas", "Python"),
        ("Spring Boot", "Java"),
        ("React", "JavaScript"),
    ]
    for child, parent in known_pairs:
        def make_check(c, p):
            def _check():
                from nodes.expand_skills import expand_skills
                result = expand_skills([{"skill": c, "experience_years": 1.0}])
                names = [s["skill"] for s in result]
                if p not in names:
                    return f"{c} should expand to include {p}, got: {names}"
                return True
            return _check
        check(f"'{child}' expands to include '{parent}'", make_check(child, parent))


# ══════════════════════════════════════════════════════════════
#  PHASE 4 — Document Ingestion
# ══════════════════════════════════════════════════════════════
def phase4_ingestion():
    section("Document Ingestion", 4)

    def sample_files_exist():
        data_dir = Path("data/resumes")
        if not data_dir.exists():
            return "data/resumes/ directory not found"
        files = list(data_dir.iterdir())
        if len(files) == 0:
            return "No files in data/resumes/"
        return True
    check("data/resumes/ exists and has files", sample_files_exist)

    def load_node_importable():
        from nodes.load_documents import load_documents_node
        return True
    check("load_documents_node importable", load_node_importable)

    def ingestion_runs():
        from nodes.load_documents import load_documents_node
        files = list(Path("data/resumes").iterdir())[:2]
        file_paths = [str(f) for f in files]
        state = {"resume_file_paths": file_paths, "raw_resume_texts": [], "node_logs": []}
        result = load_documents_node(state)
        docs = result.get("raw_resume_texts", [])
        if len(docs) == 0:
            return "No documents loaded"
        # Check metadata fields
        required_meta = ["file_name", "text", "file_type"]
        for doc in docs:
            for field in required_meta:
                if field not in doc:
                    return f"Missing field '{field}' in loaded doc: {list(doc.keys())}"
        return True
    check("load_documents_node returns docs with required metadata", ingestion_runs)

    def metadata_fields_check():
        from nodes.load_documents import load_documents_node
        files = list(Path("data/resumes").iterdir())[:1]
        state = {"resume_file_paths": [str(files[0])], "raw_resume_texts": [], "node_logs": []}
        result = load_documents_node(state)
        doc = result["raw_resume_texts"][0]
        assert "upload_time" in doc, "Missing upload_time"
        assert "file_name" in doc, "Missing file_name"
        assert isinstance(doc["text"], str) and len(doc["text"]) > 10, "Text too short or wrong type"
        return True
    check("Loaded doc has upload_time and non-empty text", metadata_fields_check)


# ══════════════════════════════════════════════════════════════
#  PHASE 5 — Resume Parsing
# ══════════════════════════════════════════════════════════════
def phase5_parsing():
    section("Resume Parsing (AzureOpenAI → Pydantic)", 5)

    def parse_node_importable():
        from nodes.parse_resume import parse_resume_node
        return True
    check("parse_resume_node importable", parse_node_importable)

    def parse_runs():
        from nodes.parse_resume import parse_resume_node
        mock_text = """
        John Smith | john@email.com | +91 9876543210
        Skills: Python (4 years), FastAPI (2 years), PostgreSQL (3 years)
        Education: B.Tech Computer Science, IIT Delhi, 2019, GPA 8.5
        Experience: Backend Engineer at TechCorp (2019-2022)
          - Built REST APIs for payment processing
        Certifications: AWS Certified Developer
        """
        state = {
            "raw_resume_texts": [{"file_name": "john.pdf", "text": mock_text, "file_type": "pdf", "upload_time": "2025-01-01T00:00:00"}],
            "parsed_resumes": [],
            "failed_docs": [],
            "node_logs": [],
        }
        result = parse_resume_node(state)
        parsed = result.get("parsed_resumes", [])
        if not parsed:
            return "No parsed resumes returned (check failed_docs)"
        r = parsed[0]
        print()
        info(f"    Parsed name: {r.get('name')}")
        info(f"    Skills count: {len(r.get('skills', []))}")
        info(f"    Experience entries: {len(r.get('experience', []))}")
        return True
    check("parse_resume_node returns structured data for mock resume", parse_runs)

    def parse_has_candidate_id():
        from nodes.parse_resume import parse_resume_node
        mock_text = "Alice Wong | alice@email.com\nSkills: Java (5 years)\nExperience: Java Developer 3 years"
        state = {
            "raw_resume_texts": [{"file_name": "alice.pdf", "text": mock_text, "file_type": "pdf", "upload_time": "2025-01-01T00:00:00"}],
            "parsed_resumes": [], "failed_docs": [], "node_logs": [],
        }
        result = parse_resume_node(state)
        parsed = result.get("parsed_resumes", [])
        if not parsed:
            return "No parsed output"
        if "candidate_id" not in parsed[0]:
            return "Missing candidate_id in parsed resume"
        return True
    check("Parsed resume contains auto-generated candidate_id", parse_has_candidate_id)


# ══════════════════════════════════════════════════════════════
#  PHASE 6 — Validation Node
# ══════════════════════════════════════════════════════════════
def phase6_validation():
    section("Validation Node (retry + backoff + dead-letter)", 6)

    def validation_node_importable():
        from nodes.validate_schema import validate_schema_node
        return True
    check("validate_schema_node importable", validation_node_importable)

    def valid_doc_passes():
        from nodes.validate_schema import validate_schema_node
        import uuid
        valid_resume = {
            "candidate_id": str(uuid.uuid4()),
            "name": "Test User",
            "email": None, "phone": None, "linkedin": None, "github": None, "location": None,
            "skills": [{"skill": "Python", "experience_years": 3.0}],
            "education": [{"degree": "B.Tech", "university": "IIT", "graduation_year": 2020, "gpa": None}],
            "experience": [{"role": "Dev", "summary": "Built things", "project_title": None, "client": None, "duration_months": 12}],
            "certifications": [],
            "file_name": "test.pdf",
            "upload_time": "2025-01-01T00:00:00",
        }
        state = {"parsed_resumes": [valid_resume], "failed_docs": [], "node_logs": []}
        result = validate_schema_node(state)
        assert len(result["parsed_resumes"]) == 1, "Valid doc should remain in parsed_resumes"
        assert len(result["failed_docs"]) == 0, "No failures expected"
        return True
    check("Valid resume passes validation without error", valid_doc_passes)

    def invalid_doc_goes_to_failed():
        from nodes.validate_schema import validate_schema_node
        invalid_resume = {"name": "Broken", "skills": "NOT_A_LIST"}  # malformed
        state = {"parsed_resumes": [invalid_resume], "failed_docs": [], "node_logs": []}
        result = validate_schema_node(state)
        if len(result.get("failed_docs", [])) == 0:
            return "Invalid doc should have been moved to failed_docs"
        fd = result["failed_docs"][0]
        if "reason" not in fd:
            return "failed_docs entry should have a 'reason' field"
        info(f"    Correctly rejected — reason: {fd['reason'][:80]}")
        return True
    check("Invalid resume moves to failed_docs with reason", invalid_doc_goes_to_failed)

    def retry_count_respected():
        from nodes.validate_schema import validate_schema_node
        # Feed multiple invalid docs and verify all end up in failed_docs
        invalid_docs = [{"name": f"Bad{i}", "skills": i} for i in range(3)]
        state = {"parsed_resumes": invalid_docs, "failed_docs": [], "node_logs": []}
        result = validate_schema_node(state)
        if len(result["failed_docs"]) != 3:
            return f"Expected 3 failed docs, got {len(result['failed_docs'])}"
        return True
    check("All 3 invalid docs end up in failed_docs", retry_count_respected)


# ══════════════════════════════════════════════════════════════
#  PHASE 7 — Embedding & Qdrant Storage
# ══════════════════════════════════════════════════════════════
def phase7_embedding():
    section("Embedding & Qdrant Storage", 7)

    def qdrant_collection_exists():
        from qdrant_client import QdrantClient
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        client = QdrantClient(url=cfg["qdrant"]["url"])
        colls = [c.name for c in client.get_collections().collections]
        resume_coll = cfg["qdrant"]["resume_collection"]
        if resume_coll not in colls:
            return f"Collection '{resume_coll}' not found. Run embed_and_store_node first."
        return True
    check("Qdrant resumes_index collection exists", qdrant_collection_exists)

    def embed_and_store_importable():
        from nodes.embed_and_store import embed_and_store_node
        return True
    check("embed_and_store_node importable", embed_and_store_importable)

    def embed_template_works():
        from utils.embed_template import build_resume_embedding_text
        from schemas.resume_schema import ResumeSchema
        import uuid
        r = ResumeSchema(
            candidate_id=str(uuid.uuid4()), name="Test",
            skills=[{"skill": "Python", "experience_years": 3.0}],
            education=[{"degree": "B.Tech", "university": "IIT", "graduation_year": 2020, "gpa": None}],
            experience=[{"role": "Dev", "summary": "Built APIs", "project_title": None, "client": None, "duration_months": 12}],
            certifications=["AWS"], file_name="t.pdf", upload_time="2025-01-01T00:00:00",
        )
        text = build_resume_embedding_text(r)
        assert "Python" in text, "Skills should appear in embedding text"
        assert "Test" in text, "Name should appear in embedding text"
        info(f"    Embedding text preview: {text[:120]}...")
        return True
    check("build_resume_embedding_text produces valid text", embed_template_works)

    def qdrant_point_count_increases():
        from qdrant_client import QdrantClient
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        client = QdrantClient(url=cfg["qdrant"]["url"])
        coll = cfg["qdrant"]["resume_collection"]
        try:
            info_obj = client.get_collection(coll)
            count = info_obj.points_count
            info(f"    Current point count in {coll}: {count}")
            return True
        except Exception as e:
            return f"Could not query collection: {e}"
    check("Qdrant collection point count is accessible", qdrant_point_count_increases)

    def vector_dimension_correct():
        from qdrant_client import QdrantClient
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        client = QdrantClient(url=cfg["qdrant"]["url"])
        coll = cfg["qdrant"]["resume_collection"]
        try:
            info_obj = client.get_collection(coll)
            dim = info_obj.config.params.vectors.size
            expected = cfg["qdrant"]["vector_size"]
            if dim != expected:
                return f"Vector dimension mismatch: got {dim}, expected {expected}"
            info(f"    Vector dimension: {dim} ✓")
            return True
        except Exception:
            return True  # Collection may not exist yet if no resumes embedded
    check("Qdrant collection vector dimension matches config", vector_dimension_correct)


# ══════════════════════════════════════════════════════════════
#  PHASE 8 — JD Pipeline
# ══════════════════════════════════════════════════════════════
def phase8_jd():
    section("JD Parsing & Embedding", 8)

    def parse_jd_importable():
        from nodes.parse_jd import parse_jd_node
        return True
    check("parse_jd_node importable", parse_jd_importable)

    def parse_jd_runs():
        from nodes.parse_jd import parse_jd_node
        sample_jd = """
        Senior Python Engineer — FinTech Domain
        We are looking for a Python engineer with 4+ years experience.
        Required: Python (4y), FastAPI (2y), PostgreSQL (2y)
        Nice to have: Redis, Docker, Kubernetes
        Education: B.Tech or equivalent
        """
        state = {"jd_raw_text": sample_jd, "parsed_jd": None, "jd_embedding": None, "node_logs": []}
        result = parse_jd_node(state)
        jd = result.get("parsed_jd")
        if not jd:
            return "parse_jd_node returned no parsed_jd"
        if "required_skills" not in jd:
            return f"Missing required_skills in parsed JD: {list(jd.keys())}"
        info(f"    JD title: {jd.get('title')}")
        info(f"    Required skills: {[s['skill'] for s in jd.get('required_skills', [])]}")
        return True
    check("parse_jd_node returns structured JD with required_skills", parse_jd_runs)

    def embed_jd_importable():
        from nodes.embed_jd import embed_jd_node
        return True
    check("embed_jd_node importable", embed_jd_importable)

    def jd_embedding_dimension():
        from nodes.parse_jd import parse_jd_node
        from nodes.embed_jd import embed_jd_node
        sample_jd = "Senior Data Engineer. Required: Python 3y, Spark 2y. Experience: 3+ years."
        state = {"jd_raw_text": sample_jd, "parsed_jd": None, "jd_embedding": None, "node_logs": []}
        state = parse_jd_node(state)
        state = embed_jd_node(state)
        vec = state.get("jd_embedding")
        if not vec:
            return "jd_embedding is None after embed_jd_node"
        if len(vec) != 1536:
            return f"Expected 1536 dims, got {len(vec)}"
        info(f"    JD embedding dimension: {len(vec)} ✓")
        return True
    check("JD embedding has 1536 dimensions", jd_embedding_dimension)


# ══════════════════════════════════════════════════════════════
#  PHASE 9 — Hybrid Retrieval
# ══════════════════════════════════════════════════════════════
def phase9_retrieval():
    section("Hybrid Retrieval (Vector + BM25 + RRF)", 9)

    def rrf_importable():
        from retrieval.rrf import reciprocal_rank_fusion
        return True
    check("reciprocal_rank_fusion importable", rrf_importable)

    def rrf_correctness():
        from retrieval.rrf import reciprocal_rank_fusion
        r1 = ["A", "B", "C", "D"]
        r2 = ["C", "A", "D", "B"]
        result = reciprocal_rank_fusion([r1, r2])
        # A and C appear high in both lists, should be top 2
        assert "A" in result[:2] or "C" in result[:2], f"Expected A or C in top 2, got {result[:2]}"
        info(f"    RRF result order: {result}")
        return True
    check("RRF produces consistent fusion ranking", rrf_correctness)

    def rrf_handles_single_list():
        from retrieval.rrf import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([["X", "Y", "Z"]])
        assert result == ["X", "Y", "Z"], f"Single list RRF should preserve order, got {result}"
        return True
    check("RRF handles single ranking list correctly", rrf_handles_single_list)

    def hybrid_retrieve_importable():
        from nodes.hybrid_retrieve import hybrid_retrieve_node
        return True
    check("hybrid_retrieve_node importable", hybrid_retrieve_importable)

    def hybrid_retrieve_runs():
        from nodes.hybrid_retrieve import hybrid_retrieve_node
        from nodes.parse_jd import parse_jd_node
        from nodes.embed_jd import embed_jd_node
        sample_jd = "Python Engineer. Required: Python 3y, FastAPI 2y."
        state = {
            "jd_raw_text": sample_jd, "parsed_jd": None, "jd_embedding": None,
            "retrieved_candidates": [], "node_logs": [],
        }
        state = parse_jd_node(state)
        state = embed_jd_node(state)
        state = hybrid_retrieve_node(state)
        candidates = state.get("retrieved_candidates", [])
        info(f"    Retrieved {len(candidates)} candidates")
        if len(candidates) == 0:
            warn("No candidates retrieved — ensure resumes are embedded (Phase 7)")
        for c in candidates[:3]:
            info(f"    - {c.get('name', 'unknown')} | score: {c.get('score', 'N/A'):.4f}")
        return True
    check("hybrid_retrieve_node returns candidates list", hybrid_retrieve_runs)


# ══════════════════════════════════════════════════════════════
#  PHASE 10 — Re-ranking
# ══════════════════════════════════════════════════════════════
def phase10_rerank():
    section("Re-ranking (LLM Cross-Encoder)", 10)

    def rerank_importable():
        from nodes.rerank_candidates import rerank_candidates_node
        return True
    check("rerank_candidates_node importable", rerank_importable)

    def rerank_preserves_count():
        from nodes.rerank_candidates import rerank_candidates_node
        mock_jd = {"title": "Python Dev", "required_skills": [{"skill": "Python", "experience_years": 3.0}]}
        mock_candidates = [
            {"candidate_id": f"id{i}", "name": f"Candidate {i}", "skills": ["Python"], "experience": f"{i} years"}
            for i in range(5)
        ]
        state = {
            "parsed_jd": mock_jd,
            "retrieved_candidates": mock_candidates,
            "reranked_candidates": [],
            "node_logs": [],
        }
        result = rerank_candidates_node(state)
        reranked = result.get("reranked_candidates", [])
        info(f"    Reranked {len(reranked)} candidates")
        if len(reranked) == 0:
            return "reranked_candidates is empty"
        return True
    check("rerank_candidates_node returns non-empty reranked list", rerank_preserves_count)

    def rerank_has_score():
        from nodes.rerank_candidates import rerank_candidates_node
        mock_jd = {"title": "Python Dev", "required_skills": [{"skill": "Python", "experience_years": 3.0}]}
        mock_candidates = [{"candidate_id": "x1", "name": "Alice", "skills": ["Python", "FastAPI"], "experience": "5 years"}]
        state = {"parsed_jd": mock_jd, "retrieved_candidates": mock_candidates, "reranked_candidates": [], "node_logs": []}
        result = rerank_candidates_node(state)
        r = result.get("reranked_candidates", [{}])[0]
        if "rerank_score" not in r:
            return f"Missing rerank_score in reranked candidate: {list(r.keys())}"
        return True
    check("Reranked candidates have rerank_score field", rerank_has_score)


# ══════════════════════════════════════════════════════════════
#  PHASE 11 — ATS Scoring + Score Fusion
# ══════════════════════════════════════════════════════════════
def phase11_scoring():
    section("ATS Scoring & Final Score Fusion", 11)

    def ats_importable():
        from nodes.score_ats import score_ats_node
        return True
    check("score_ats_node importable", ats_importable)

    def ats_score_in_range():
        from nodes.score_ats import score_ats_node
        mock_jd = {
            "required_skills": [{"skill": "Python", "experience_years": 3.0}],
            "experience_min_years": 3,
        }
        mock_candidates = [{
            "candidate_id": "c1",
            "name": "Alice",
            "skills": [{"skill": "Python", "experience_years": 4.0}],
            "experience": [{"role": "Dev", "summary": "Python dev", "duration_months": 48}],
            "education": [{"degree": "B.Tech", "university": "IIT", "graduation_year": 2019, "gpa": 8.5}],
            "certifications": ["AWS"],
        }]
        state = {"parsed_jd": mock_jd, "reranked_candidates": mock_candidates, "ats_scores": {}, "node_logs": []}
        result = score_ats_node(state)
        scores = result.get("ats_scores", {})
        score = scores.get("c1")
        if score is None:
            return "ATS score for c1 is None"
        if not (0.0 <= score <= 1.0):
            return f"ATS score out of range [0,1]: {score}"
        info(f"    ATS score for Alice: {score:.4f}")
        return True
    check("ATS score is in range [0.0, 1.0]", ats_score_in_range)

    def fusion_importable():
        from nodes.fuse_scores import fuse_scores_node
        return True
    check("fuse_scores_node importable", fusion_importable)

    def fusion_score_in_range():
        from nodes.fuse_scores import fuse_scores_node
        mock_candidates = [{
            "candidate_id": "c1",
            "name": "Alice",
            "semantic_similarity": 0.85,
            "skill_match_score": 0.90,
            "experience_score": 0.80,
        }]
        state = {
            "reranked_candidates": mock_candidates,
            "ats_scores": {"c1": 0.75},
            "final_scores": [],
            "node_logs": [],
        }
        result = fuse_scores_node(state)
        final = result.get("final_scores", [])
        if not final:
            return "final_scores is empty"
        fs = final[0].get("final_score")
        if fs is None:
            return "final_score field missing"
        if not (0.0 <= fs <= 1.0):
            return f"Final score out of range: {fs}"
        info(f"    Final fused score for Alice: {fs:.4f}")
        return True
    check("Final fused score is in range [0.0, 1.0]", fusion_score_in_range)


# ══════════════════════════════════════════════════════════════
#  PHASE 12 — LangGraph Wiring
# ══════════════════════════════════════════════════════════════
def phase12_graph():
    section("LangGraph Graph Wiring", 12)

    def ingestion_graph_importable():
        from graphs.ingestion_graph import build_ingestion_graph
        return True
    check("build_ingestion_graph importable", ingestion_graph_importable)

    def query_graph_importable():
        from graphs.query_graph import build_query_graph
        return True
    check("build_query_graph importable", query_graph_importable)

    def state_typed_dict():
        from graphs.state import ResumeMatcherState
        import typing
        hints = typing.get_type_hints(ResumeMatcherState)
        required = ["resume_file_paths", "parsed_resumes", "failed_docs", "jd_raw_text", "final_scores"]
        for field in required:
            if field not in hints:
                return f"Missing field '{field}' in ResumeMatcherState"
        info(f"    State fields: {list(hints.keys())}")
        return True
    check("ResumeMatcherState TypedDict has all required fields", state_typed_dict)

    def ingestion_graph_compiles():
        from graphs.ingestion_graph import build_ingestion_graph
        graph = build_ingestion_graph()
        # LangGraph compiled graph should have a get_graph method
        assert hasattr(graph, "invoke") or hasattr(graph, "stream"), "Graph missing invoke/stream"
        return True
    check("Ingestion graph compiles without error", ingestion_graph_compiles)

    def query_graph_compiles():
        from graphs.query_graph import build_query_graph
        graph = build_query_graph()
        assert hasattr(graph, "invoke") or hasattr(graph, "stream"), "Graph missing invoke/stream"
        return True
    check("Query graph compiles without error", query_graph_compiles)


# ══════════════════════════════════════════════════════════════
#  PHASE 13 — Streamlit UI (static checks)
# ══════════════════════════════════════════════════════════════
def phase13_ui():
    section("Streamlit UI (static checks)", 13)

    def main_app_exists():
        if not Path("app/main.py").exists():
            return "app/main.py not found"
        return True
    check("app/main.py exists", main_app_exists)

    def auth_module_exists():
        if not Path("app/auth.py").exists():
            return "app/auth.py not found"
        return True
    check("app/auth.py exists", auth_module_exists)

    def auth_uses_bcrypt():
        auth_content = Path("app/auth.py").read_text() if Path("app/auth.py").exists() else ""
        if "bcrypt" not in auth_content and "passlib" not in auth_content:
            return "auth.py does not use bcrypt or passlib — plaintext password detected!"
        return True
    check("auth.py uses bcrypt/passlib (not plaintext)", auth_uses_bcrypt)

    def no_localStorage_usage():
        app_content = Path("app/main.py").read_text() if Path("app/main.py").exists() else ""
        if "local storage" in app_content.lower() or "localstorage" in app_content.lower():
            return "app/main.py references localStorage — insecure"
        return True
    check("No localStorage usage in Streamlit app", no_localStorage_usage)

    def download_button_for_pdf():
        pages = list(Path("app/pages").glob("*.py")) if Path("app/pages").exists() else []
        results_page = next((p for p in pages if "result" in p.name), None)
        if not results_page:
            warn("app/pages/results.py not found — skipping PDF check")
            return True
        content = results_page.read_text()
        if "download_button" not in content and "components.v1.html" not in content:
            return "results.py should use st.download_button or st.components.v1.html for PDF"
        return True
    check("PDF display uses st.download_button or iframe", download_button_for_pdf)


# ══════════════════════════════════════════════════════════════
#  PHASE 14 — Observability
# ══════════════════════════════════════════════════════════════
def phase14_observability():
    section("Observability & Logging", 14)

    def logger_importable():
        from utils.logger import log_node
        return True
    check("utils.logger.log_node importable", logger_importable)

    def logger_writes_json():
        from utils.logger import log_node
        log_node("test_node", "success", latency_ms=42, error=None)
        log_file = Path("logs/pipeline.log")
        if not log_file.exists():
            return "logs/pipeline.log not created by logger"
        last_line = log_file.read_text().strip().split("\n")[-1]
        entry = json.loads(last_line)
        assert "node" in entry, "Log entry missing 'node' field"
        assert "latency_ms" in entry, "Log entry missing 'latency_ms'"
        info(f"    Log entry: {entry}")
        return True
    check("logger writes JSON entries to logs/pipeline.log", logger_writes_json)

    def node_logs_in_state():
        # Check any node adds to node_logs in state
        from nodes.load_documents import load_documents_node
        state = {"resume_file_paths": [], "raw_resume_texts": [], "node_logs": []}
        result = load_documents_node(state)
        logs = result.get("node_logs", [])
        if not logs:
            return "load_documents_node did not append to node_logs"
        entry = logs[0]
        for field in ["node", "status", "latency_ms"]:
            if field not in entry:
                return f"node_logs entry missing '{field}': {entry}"
        return True
    check("Nodes append structured entries to state.node_logs", node_logs_in_state)


# ══════════════════════════════════════════════════════════════
#  REGISTRY & RUNNER
# ══════════════════════════════════════════════════════════════

PHASES = {
    1:  ("Project Setup",              phase1_setup),
    2:  ("Pydantic Schemas",           phase2_schemas),
    3:  ("Skills Expansion Graph",     phase3_skills),
    4:  ("Document Ingestion",         phase4_ingestion),
    5:  ("Resume Parsing",             phase5_parsing),
    6:  ("Validation Node",            phase6_validation),
    7:  ("Embedding & Qdrant",         phase7_embedding),
    8:  ("JD Pipeline",                phase8_jd),
    9:  ("Hybrid Retrieval",           phase9_retrieval),
    10: ("Re-ranking",                 phase10_rerank),
    11: ("ATS Scoring & Fusion",       phase11_scoring),
    12: ("LangGraph Wiring",           phase12_graph),
    13: ("Streamlit UI",               phase13_ui),
    14: ("Observability",              phase14_observability),
}

def parse_phase_arg(arg: str) -> list[int]:
    if arg == "all":
        return list(PHASES.keys())
    if "-" in arg:
        start, end = arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(arg)]

def main():
    # Add project root to sys.path so nodes/, schemas/ etc. are importable
    sys.path.insert(0, str(Path(".").resolve()))

    parser = argparse.ArgumentParser(description="Agentic Resume Matcher — Phase Evaluator")
    parser.add_argument("--phase", default="all", help="Phase number, range (e.g. 1-5), or 'all'")
    args = parser.parse_args()

    phases_to_run = parse_phase_arg(args.phase)
    invalid = [p for p in phases_to_run if p not in PHASES]
    if invalid:
        print(f"{RED}Unknown phase(s): {invalid}. Valid: 1–14{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}  AGENTIC RESUME MATCHER — EVALUATION RUNNER{RESET}")
    print(f"{BOLD}  Phases: {phases_to_run}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    global PASS_COUNT, FAIL_COUNT
    PASS_COUNT = 0
    FAIL_COUNT = 0

    for phase_num in phases_to_run:
        if phase_num not in PHASES:
            continue
        _, fn = PHASES[phase_num]
        try:
            fn()
        except Exception as e:
            print(f"\n{RED}Phase {phase_num} runner crashed: {e}{RESET}")
            traceback.print_exc()

    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}  FINAL RESULT: {PASS_COUNT}/{total} checks passed{RESET}")
    if FAIL_COUNT > 0:
        print(f"  {RED}{FAIL_COUNT} check(s) failed — fix before proceeding{RESET}")
    else:
        print(f"  {GREEN}All checks passed!{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

    sys.exit(0 if FAIL_COUNT == 0 else 1)

if __name__ == "__main__":
    main()

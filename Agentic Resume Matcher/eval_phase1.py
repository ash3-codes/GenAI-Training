"""
============================================================
  PHASE 1 EVALUATION — Project Setup
============================================================
Run from your project root (resume_matcher/):
    python eval_phase1.py

Checks:
  1.  All required env vars present
  2.  Config loads and weight sums are exactly 1.0
  3.  Qdrant cloud connection successful
  4.  AzureOpenAI chat model reachable (tiny ping)
  5.  AzureOpenAI embedding model reachable + correct dimension
  6.  All required libraries importable
  7.  Logger writes structured JSON to logs/pipeline.log
  8.  settings.py validate_config() returns no errors
  9.  Data directories exist (or get created)
  10. Missing libraries reported clearly (rank-bm25, passlib)
============================================================
"""

import sys
import os
import time
import json
import importlib
from pathlib import Path

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"; DIM = "\033[2m"

pass_count = 0
fail_count = 0
warn_count = 0

def ok(msg):
    global pass_count; pass_count += 1
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")

def fail(msg, detail=""):
    global fail_count; fail_count += 1
    print(f"  {RED}✗ FAIL{RESET}  {msg}")
    if detail:
        print(f"         {DIM}{detail}{RESET}")

def warn(msg, detail=""):
    global warn_count; warn_count += 1
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
    if detail:
        print(f"         {DIM}{detail}{RESET}")

def info(msg):
    print(f"         {DIM}{msg}{RESET}")

def header(title):
    print(f"\n{BOLD}{CYAN}── {title} {'─'*(54-len(title))}{RESET}")

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(".").resolve()))

# Load .env early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print(f"{RED}python-dotenv not installed. Run: pip install python-dotenv{RESET}")
    sys.exit(1)

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 1 — Project Setup Evaluation{RESET}")
print(f"{BOLD}{'='*60}{RESET}")


# ── CHECK 1: Environment Variables ────────────────────────────────────────────
header("1. Environment Variables")

env_vars = {
    "ENDPOINT":                  "Azure OpenAI endpoint URL",
    "AZURE_OPENAI_API_KEY":      "Azure OpenAI API key",
    "API_VERSION":               "Chat model API version",
    "DEPLOYMENT_NAME":           "Chat deployment name",
    "DEPLOYMENT_NAME_EMBEDDING": "Embedding deployment name",
    "API_VERSION_EMBEDDING":     "Embedding API version",
    "QDARNT_ENDPOINT":           "Qdrant endpoint (typo preserved)",
    "QDARNT_API_KEY":            "Qdrant API key (typo preserved)",
}

all_vars_present = True
for var, desc in env_vars.items():
    val = os.environ.get(var)
    if val:
        ok(f"{var} present  ({desc})")
        info(f"Value: {val[:40]}{'...' if len(val) > 40 else ''}")
    else:
        fail(f"{var} MISSING  ({desc})", "Add this to your .env file")
        all_vars_present = False

optional_vars = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]
for var in optional_vars:
    val = os.environ.get(var)
    if val:
        ok(f"{var} present (LangSmith tracing enabled)")
    else:
        warn(f"{var} not set — LangSmith tracing disabled")


# ── CHECK 2: config.yaml ──────────────────────────────────────────────────────
header("2. config.yaml")

try:
    import yaml
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        fail("config.yaml not found", "Create it at your project root")
    else:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        ok("config.yaml found and parsed")
        info(f"Path: {cfg_path.resolve()}")

        # Check required sections
        required_sections = ["azure_openai", "qdrant", "retrieval", "ats_weights", "final_score_weights", "validation"]
        for section in required_sections:
            if section in cfg:
                ok(f"Section '{section}' present")
            else:
                fail(f"Section '{section}' missing from config.yaml")

        # Weight validation
        ats = cfg.get("ats_weights", {})
        ats_sum = round(sum(ats.values()), 6)
        if abs(ats_sum - 1.0) < 0.001:
            ok(f"ats_weights sum = {ats_sum} ✓")
        else:
            fail(f"ats_weights sum = {ats_sum} (must be 1.0)")

        fs = cfg.get("final_score_weights", {})
        fs_sum = round(sum(fs.values()), 6)
        if abs(fs_sum - 1.0) < 0.001:
            ok(f"final_score_weights sum = {fs_sum} ✓")
        else:
            fail(f"final_score_weights sum = {fs_sum} (must be 1.0)")

        info(f"Chat deployment: {cfg.get('azure_openai', {}).get('chat_deployment')}")
        info(f"Embedding deployment: {cfg.get('azure_openai', {}).get('embedding_deployment')}")

except Exception as e:
    fail(f"config.yaml error: {e}")


# ── CHECK 3: settings.py ──────────────────────────────────────────────────────
header("3. config/settings.py")

try:
    from config.settings import validate_config, AZURE_CHAT_DEPLOYMENT, AZURE_EMBEDDING_DEPLOYMENT, QDRANT_URL, QDRANT_RESUME_COLLECTION, QDRANT_JD_COLLECTION
    ok("config.settings imported successfully")
    info(f"Chat deployment:      {AZURE_CHAT_DEPLOYMENT}")
    info(f"Embedding deployment: {AZURE_EMBEDDING_DEPLOYMENT}")
    info(f"Qdrant URL:           {QDRANT_URL}")
    info(f"Resume collection:    {QDRANT_RESUME_COLLECTION}")
    info(f"JD collection:        {QDRANT_JD_COLLECTION}")

    errors = validate_config()
    if not errors:
        ok("validate_config() passed — no errors")
    else:
        for e in errors:
            fail(f"validate_config(): {e}")
except Exception as e:
    fail(f"config.settings import failed: {e}", str(e))


# ── CHECK 4: Qdrant Connection ────────────────────────────────────────────────
header("4. Qdrant Connection")

try:
    from qdrant_client import QdrantClient
    url = os.environ.get("QDARNT_ENDPOINT")
    api_key = os.environ.get("QDARNT_API_KEY")

    if not url:
        fail("Cannot test Qdrant — QDARNT_ENDPOINT not set")
    else:
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
        collections = client.get_collections()
        coll_names = [c.name for c in collections.collections]
        ok(f"Qdrant connected at {url}")
        info(f"Existing collections: {coll_names if coll_names else '(none yet)'}")

        # Check if our collections exist already
        from config.settings import QDRANT_RESUME_COLLECTION, QDRANT_JD_COLLECTION
        for coll in [QDRANT_RESUME_COLLECTION, QDRANT_JD_COLLECTION]:
            if coll in coll_names:
                ok(f"Collection '{coll}' already exists")
            else:
                warn(f"Collection '{coll}' not yet created — will be created in Phase 7")

except Exception as e:
    fail(f"Qdrant connection failed: {type(e).__name__}", str(e))


# ── CHECK 5: AzureOpenAI Chat ─────────────────────────────────────────────────
header("5. AzureOpenAI Chat Model")

try:
    from config.settings import get_chat_llm, AZURE_CHAT_DEPLOYMENT

    # Use factory — it passes azure_endpoint= explicitly, no env var ambiguity
    llm = get_chat_llm(max_tokens=5, temperature=0)
    start = time.time()
    resp = llm.invoke("Reply with the single word: OK")
    latency = (time.time() - start) * 1000
    ok(f"Chat model reachable (latency: {latency:.0f}ms)")
    info(f"Response: {resp.content.strip()}")
    info(f"Deployment: {AZURE_CHAT_DEPLOYMENT}")
except Exception as e:
    fail(f"AzureOpenAI chat failed: {type(e).__name__}", str(e))


# ── CHECK 6: AzureOpenAI Embeddings ──────────────────────────────────────────
header("6. AzureOpenAI Embedding Model")

try:
    from config.settings import get_embedding_model, AZURE_EMBEDDING_DEPLOYMENT, QDRANT_VECTOR_SIZE

    # Use factory — same pattern, explicit azure_endpoint
    emb = get_embedding_model()
    start = time.time()
    vec = emb.embed_query("test embedding ping")
    latency = (time.time() - start) * 1000
    ok(f"Embedding model reachable (latency: {latency:.0f}ms)")
    info(f"Deployment: {AZURE_EMBEDDING_DEPLOYMENT}")

    if len(vec) == QDRANT_VECTOR_SIZE:
        ok(f"Embedding dimension = {len(vec)} (matches config: {QDRANT_VECTOR_SIZE})")
    else:
        fail(f"Dimension mismatch: got {len(vec)}, config expects {QDRANT_VECTOR_SIZE}",
             f"Update qdrant.vector_size in config.yaml to {len(vec)}")

except Exception as e:
    fail(f"AzureOpenAI embeddings failed: {type(e).__name__}", str(e))


# ── CHECK 7: Required Libraries ───────────────────────────────────────────────
header("7. Library Availability")

# (library_import_name, pip_install_name, required_or_optional)
libs = [
    ("langchain",            "langchain",              True),
    ("langchain_openai",     "langchain-openai",       True),
    ("langgraph",            "langgraph",              True),
    ("langsmith",            "langsmith",              True),
    ("qdrant_client",        "qdrant-client",          True),
    ("pydantic",             "pydantic",               True),
    ("streamlit",            "streamlit",              True),
    ("dotenv",               "python-dotenv",          True),
    ("yaml",                 "pyyaml",                 True),
    ("tenacity",             "tenacity",               True),
    ("pypdf",                "pypdf",                  True),
    ("pdfplumber",           "pdfplumber",             True),
    ("docx",                 "python-docx",            True),
    ("sentence_transformers","sentence-transformers",  True),
    ("sklearn",              "scikit-learn",           True),
    ("pythonjsonlogger",     "python-json-logger",     True),
    ("rank_bm25",            "rank-bm25",              True),   # Likely MISSING
    ("passlib",              "passlib[bcrypt]",        True),   # Likely MISSING
]

missing_installs = []
for import_name, pip_name, required in libs:
    try:
        importlib.import_module(import_name)
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        ok(f"{import_name} ({ver})")
    except ImportError:
        if required:
            fail(f"{import_name} NOT installed", f"pip install {pip_name}")
            missing_installs.append(pip_name)
        else:
            warn(f"{import_name} not installed (optional)", f"pip install {pip_name}")

if missing_installs:
    print(f"\n  {YELLOW}Run this to install missing libraries:{RESET}")
    print(f"  {BOLD}pip install {' '.join(missing_installs)}{RESET}")


# ── CHECK 8: Logger ───────────────────────────────────────────────────────────
header("8. Structured Logger")

try:
    from utils.logger import log_node, NodeTimer
    ok("utils.logger imported")

    entry = log_node("eval_phase1_test", "success", latency_ms=12.5, extra={"test": True})
    ok(f"log_node() executed successfully")

    log_file = Path("logs/pipeline.log")
    if log_file.exists():
        last_line = log_file.read_text(encoding="utf-8").strip().split("\n")[-1]
        try:
            parsed = json.loads(last_line)
            ok("Log file contains valid JSON")
            info(f"Sample entry: {last_line[:120]}")
        except json.JSONDecodeError:
            warn("Log file exists but last line is not valid JSON", last_line[:80])
    else:
        fail("logs/pipeline.log not created by logger")

except Exception as e:
    fail(f"Logger test failed: {type(e).__name__}", str(e))


# ── CHECK 9: Folder Structure ─────────────────────────────────────────────────
header("9. Folder Structure")

required_dirs = [
    "config", "utils", "schemas", "nodes", "graphs",
    "retrieval", "skills", "data/resumes", "data/jd",
    "app", "app/pages", "logs",
]

for d in required_dirs:
    p = Path(d)
    if p.exists():
        ok(f"{d}/  exists")
    else:
        p.mkdir(parents=True, exist_ok=True)
        warn(f"{d}/  created (was missing)")

# Check for __init__.py in Python packages
py_packages = ["config", "utils", "schemas", "nodes", "graphs", "retrieval", "skills"]
for pkg in py_packages:
    init = Path(pkg) / "__init__.py"
    if not init.exists():
        init.touch()
        warn(f"{pkg}/__init__.py created")
    else:
        ok(f"{pkg}/__init__.py present")


# ── CHECK 10: Skills Graph ────────────────────────────────────────────────────
header("10. Skills Graph File")

skills_file = Path("skills/skills_graph.json")
if skills_file.exists():
    try:
        graph = json.loads(skills_file.read_text())
        ok(f"skills_graph.json found ({len(graph)} entries)")
        info(f"Sample: {list(graph.items())[:3]}")
    except Exception as e:
        fail(f"skills_graph.json is not valid JSON: {e}")
else:
    warn("skills_graph.json not yet created — will be built in Phase 3")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
total = pass_count + fail_count
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  PHASE 1 RESULT: {pass_count}/{total} checks passed  |  {warn_count} warnings{RESET}")
if fail_count == 0:
    print(f"  {GREEN}{BOLD}All required checks passed! Ready for Phase 2.{RESET}")
elif fail_count <= 2:
    print(f"  {YELLOW}Minor issues — fix the FAILs above before Phase 2.{RESET}")
else:
    print(f"  {RED}Multiple failures — resolve before proceeding.{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

sys.exit(0 if fail_count == 0 else 1)
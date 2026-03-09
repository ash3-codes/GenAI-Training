"""
config/settings.py
------------------
Single source of truth for all configuration.
Reads .env for secrets, config.yaml for tunable parameters.
All other modules import from here — never call os.getenv() elsewhere.

FIX: langchain-openai requires azure_endpoint= passed explicitly OR
     the env var must be named AZURE_OPENAI_ENDPOINT (not ENDPOINT).
     We read from ENDPOINT (your .env name) and inject it as the
     standard AZURE_OPENAI_ENDPOINT so all LangChain constructors work.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Locate project root ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"

with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)


# ── Azure OpenAI — Chat ──────────────────────────────────────────────────────
# Your .env uses "ENDPOINT" — we read it and also inject it under the
# standard name so langchain-openai finds it via env var lookup too.

AZURE_OPENAI_ENDPOINT    = os.getenv("ENDPOINT")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("API_VERSION", _cfg["azure_openai"]["api_version"])
AZURE_CHAT_DEPLOYMENT    = os.getenv("DEPLOYMENT_NAME", _cfg["azure_openai"]["chat_deployment"])

# Inject standard env var name that langchain-openai expects
if AZURE_OPENAI_ENDPOINT:
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

# ── Azure OpenAI — Embeddings ────────────────────────────────────────────────
AZURE_EMBEDDING_DEPLOYMENT        = os.getenv("DEPLOYMENT_NAME_EMBEDDING", _cfg["azure_openai"]["embedding_deployment"])
AZURE_OPENAI_API_VERSION_EMBEDDING = os.getenv("API_VERSION_EMBEDDING", _cfg["azure_openai"]["api_version_embedding"])

# ── Qdrant ───────────────────────────────────────────────────────────────────
# Typos in env var names preserved intentionally from original config.
QDRANT_URL     = os.getenv("QDARNT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDARNT_API_KEY")

QDRANT_RESUME_COLLECTION = _cfg["qdrant"]["resume_collection"]   # "resumes_index"
QDRANT_JD_COLLECTION     = _cfg["qdrant"]["jd_collection"]       # "jd_index"
QDRANT_VECTOR_SIZE       = _cfg["qdrant"]["vector_size"]         # 1536

# ── LLM Behaviour ────────────────────────────────────────────────────────────
TEMPERATURE  = _cfg["azure_openai"]["temperature"]
MAX_TOKENS   = _cfg["azure_openai"]["max_tokens"]

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_VECTOR = _cfg["retrieval"]["top_k_vector"]
TOP_K_BM25   = _cfg["retrieval"]["top_k_bm25"]
TOP_K_FINAL  = _cfg["retrieval"]["top_k_final"]
RRF_K        = _cfg["retrieval"]["rrf_k"]

# ── Reranking ─────────────────────────────────────────────────────────────────
CROSS_ENCODER_MODEL = _cfg["reranking"]["cross_encoder_model"]
TOP_K_RERANK        = _cfg["reranking"]["top_k_rerank"]

# ── ATS Weights ───────────────────────────────────────────────────────────────
ATS_WEIGHTS         = _cfg["ats_weights"]
FINAL_SCORE_WEIGHTS = _cfg["final_score_weights"]

# ── Validation ────────────────────────────────────────────────────────────────
MAX_RETRIES           = _cfg["validation"]["max_retries"]
RETRY_BACKOFF_SECONDS = _cfg["validation"]["retry_backoff_seconds"]

# ── Paths ─────────────────────────────────────────────────────────────────────
RESUMES_DIR = _ROOT / _cfg["data"]["resumes_dir"]
JD_DIR      = _ROOT / _cfg["data"]["jd_dir"]
LOG_FILE    = _ROOT / _cfg["logging"]["log_file"]

# ── Startup Validation ────────────────────────────────────────────────────────
def validate_config() -> list[str]:
    """
    Call once at app startup. Returns list of error messages.
    Empty list = all good.
    """
    errors = []

    # Required env vars
    required = {
        "ENDPOINT":                  AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY":      AZURE_OPENAI_API_KEY,
        "DEPLOYMENT_NAME":           AZURE_CHAT_DEPLOYMENT,
        "DEPLOYMENT_NAME_EMBEDDING": AZURE_EMBEDDING_DEPLOYMENT,
        "QDARNT_ENDPOINT":           QDRANT_URL,
        "QDARNT_API_KEY":            QDRANT_API_KEY,
    }
    for var, val in required.items():
        if not val:
            errors.append(f"Missing env var: {var}")

    # Weight sums
    ats_sum = round(sum(ATS_WEIGHTS.values()), 6)
    if abs(ats_sum - 1.0) > 0.001:
        errors.append(f"ats_weights must sum to 1.0, got {ats_sum}")

    fs_sum = round(sum(FINAL_SCORE_WEIGHTS.values()), 6)
    if abs(fs_sum - 1.0) > 0.001:
        errors.append(f"final_score_weights must sum to 1.0, got {fs_sum}")

    return errors


# ── LangChain / Qdrant factory helpers ───────────────────────────────────────
# Use these everywhere instead of constructing clients directly.
# They always pass azure_endpoint= explicitly — no env var ambiguity.

def get_chat_llm(max_tokens: int | None = None, temperature: float | None = None):
    """Returns a configured AzureChatOpenAI instance."""
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_CHAT_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=temperature if temperature is not None else TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS,
    )


def get_embedding_model():
    """Returns a configured AzureOpenAIEmbeddings instance."""
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION_EMBEDDING,
        api_key=AZURE_OPENAI_API_KEY,
    )


# Module-level singleton — all code shares ONE connection to Qdrant Cloud.
# Creating a new QdrantClient per call was causing the reset+reingest bug:
# the reset used one client instance, embed_and_store used another, and
# in-flight connection state diverged on Qdrant Cloud (stale view after delete).
_qdrant_client = None

def get_qdrant_client():
    """Returns the cached Qdrant singleton client.
    Safe to call repeatedly — always returns the same instance.
    Call reset_qdrant_client() after a collection delete to force reconnect.
    """
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    return _qdrant_client


def reset_qdrant_client():
    """Force a fresh client on the next get_qdrant_client() call.
    Must be called after deleting/recreating Qdrant collections so that
    the new connection sees the clean state rather than any cached view.
    """
    global _qdrant_client
    _qdrant_client = None


if __name__ == "__main__":
    # Quick sanity check — run: python config/settings.py
    errors = validate_config()
    if errors:
        print("CONFIG ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("✓ Config valid")
        print(f"  Chat deployment:      {AZURE_CHAT_DEPLOYMENT}")
        print(f"  Embedding deployment: {AZURE_EMBEDDING_DEPLOYMENT}")
        print(f"  Qdrant URL:           {QDRANT_URL}")
        print(f"  Resume collection:    {QDRANT_RESUME_COLLECTION}")
        print(f"  JD collection:        {QDRANT_JD_COLLECTION}")
        print(f"  ATS weights sum:      {sum(ATS_WEIGHTS.values())}")
        print(f"  Final score weights:  {sum(FINAL_SCORE_WEIGHTS.values())}")
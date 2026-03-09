"""
utils/config_loader.py
----------------------
Config loader utility -- loads and validates config.yaml with schema checking.

Provides get_config() as a cached singleton so yaml is only read once.
Also exposes typed accessors for each config section.

Usage:
    from utils.config_loader import get_config, get_retrieval_config

    cfg = get_config()
    top_k = cfg["retrieval"]["top_k_final"]
"""

import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Expected structure with types and defaults
_SCHEMA: dict[str, dict[str, Any]] = {
    "azure_openai": {
        "chat_deployment":      (str,   "gpt-4o"),
        "embedding_deployment": (str,   "text-embedding-ada-002"),
        "api_version":          (str,   "2024-02-01"),
        "api_version_embedding":(str,   "2024-02-01"),
        "temperature":          (float, 0.0),
        "max_tokens":           (int,   2000),
    },
    "qdrant": {
        "resume_collection":    (str,   "resumes_index"),
        "jd_collection":        (str,   "jd_index"),
        "vector_size":          (int,   1536),
    },
    "retrieval": {
        "top_k_vector":         (int,   20),
        "top_k_bm25":           (int,   20),
        "top_k_final":          (int,   5),
        "rrf_k":                (int,   60),
    },
    "reranking": {
        "cross_encoder_model":  (str,   "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "top_k_rerank":         (int,   10),
    },
    "ats_weights": {
        "skills":               (float, 0.40),
        "experience":           (float, 0.30),
        "projects":             (float, 0.15),
        "education":            (float, 0.10),
        "certifications":       (float, 0.05),
    },
    "final_score_weights": {
        "semantic_similarity":  (float, 0.40),
        "skill_match":          (float, 0.30),
        "experience_score":     (float, 0.20),
        "ats_score":            (float, 0.10),
    },
    "validation": {
        "max_retries":          (int,   3),
        "retry_backoff_seconds":(float, 2.0),
    },
    "data": {
        "resumes_dir":          (str,   "data/resumes"),
        "jd_dir":               (str,   "data/jd"),
    },
    "logging": {
        "log_file":             (str,   "logs/pipeline.log"),
    },
}


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load config.yaml and return as dict.
    Cached after first call -- safe to call repeatedly.
    Raises FileNotFoundError if config.yaml is missing.
    Raises ValueError if required keys are missing or wrong type.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Copy config.yaml.example to config.yaml and fill in values."
        )

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    errors = _validate(cfg)
    if errors:
        raise ValueError(
            "config.yaml validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return cfg


def _validate(cfg: dict) -> list[str]:
    """Validate config against schema. Returns list of error strings."""
    errors = []
    for section, fields in _SCHEMA.items():
        if section not in cfg:
            # Sections with defaults are not required (they can be absent)
            continue
        for key, (expected_type, _default) in fields.items():
            if key not in cfg[section]:
                continue
            val = cfg[section][key]
            # Allow int where float expected
            if expected_type is float and isinstance(val, int):
                continue
            if not isinstance(val, expected_type):
                errors.append(
                    f"{section}.{key}: expected {expected_type.__name__}, "
                    f"got {type(val).__name__} ({val!r})"
                )

    # Weight sum checks
    for weight_key in ("ats_weights", "final_score_weights"):
        if weight_key in cfg:
            total = sum(cfg[weight_key].values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"{weight_key} values must sum to 1.0, got {total:.4f}"
                )

    return errors


def get_section(section: str) -> dict:
    """Return a specific config section as a dict."""
    cfg = get_config()
    if section not in cfg:
        # Return defaults for known sections
        defaults = {}
        for key, (_, default) in _SCHEMA.get(section, {}).items():
            defaults[key] = default
        return defaults
    return cfg[section]


# ── Typed convenience accessors ───────────────────────────────────────────────

def get_retrieval_config() -> dict:
    return get_section("retrieval")

def get_ats_weights() -> dict:
    return get_section("ats_weights")

def get_final_score_weights() -> dict:
    return get_section("final_score_weights")

def get_validation_config() -> dict:
    return get_section("validation")

def get_azure_config() -> dict:
    return get_section("azure_openai")

def get_qdrant_config() -> dict:
    return get_section("qdrant")


# ── CLI: validate config ──────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        cfg = get_config()
        print("config.yaml is valid")
        for section in _SCHEMA:
            if section in cfg:
                print(f"  {section}: {len(cfg[section])} keys")
    except Exception as e:
        print(f"ERROR: {e}")
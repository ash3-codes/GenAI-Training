"""
nodes/rerank_candidates.py
--------------------------
Re-ranking Node — LangGraph node.

Uses a local CrossEncoder (sentence-transformers) to score each
(JD, candidate_summary) pair. CrossEncoders are more accurate than
bi-encoders for pairwise ranking because they see both texts jointly.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Downloaded once from HuggingFace, cached locally
  - Runs on CPU — fast enough for top-20 candidates
  - No API cost

The model is loaded lazily (on first call) to avoid startup delay.
"""

from __future__ import annotations
from typing import Any

from utils.logger import NodeTimer

# Module-level cache — loaded once per process
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        from config.settings import CROSS_ENCODER_MODEL
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _build_candidate_summary(candidate: dict) -> str:
    """
    Build a concise text summary of a candidate for the cross-encoder.
    Uses payload fields available after retrieval.
    """
    payload = candidate.get("payload", candidate)  # fallback to candidate dict itself

    name    = payload.get("name", "Unknown")
    skills  = payload.get("skills", [])
    exp_mo  = payload.get("total_experience_months", 0)
    exp_yr  = round(exp_mo / 12, 1) if exp_mo else 0
    degrees = payload.get("education_degrees", [])
    certs   = payload.get("certifications", [])

    parts = [
        f"Candidate: {name}.",
        f"Experience: {exp_yr} years.",
        f"Skills: {', '.join(skills[:15])}." if skills else "",
        f"Education: {', '.join(degrees)}." if degrees else "",
        f"Certifications: {', '.join(certs)}." if certs else "",
    ]
    return " ".join(p for p in parts if p)


def _build_jd_query(parsed_jd: dict) -> str:
    """Build a concise JD query string for the cross-encoder."""
    title     = parsed_jd.get("title", "")
    req       = parsed_jd.get("required_skills", [])
    skills    = [s.get("skill", "") if isinstance(s, dict) else s for s in req]
    min_exp   = parsed_jd.get("experience_min_years", 0)
    domain    = parsed_jd.get("domain", "")

    parts = [
        f"Job: {title}.",
        f"Required Skills: {', '.join(skills)}." if skills else "",
        f"Minimum Experience: {min_exp} years." if min_exp else "",
        f"Domain: {domain}." if domain else "",
    ]
    return " ".join(p for p in parts if p)


def rerank_candidates(
    candidates: list[dict[str, Any]],
    parsed_jd: dict[str, Any],
    top_k: int | None = None,
    cross_encoder=None,
) -> list[dict[str, Any]]:
    """
    Re-rank candidates using CrossEncoder.

    Args:
        candidates:   List of candidate dicts from hybrid_retrieve_node
        parsed_jd:    Parsed JD dict for building the query
        top_k:        Return only top_k after reranking (None = return all)
        cross_encoder: Injected for testability

    Returns:
        Re-ranked list with "rerank_score" field added to each candidate.
        Sorted by rerank_score descending.
    """
    if not candidates:
        return []

    if cross_encoder is None:
        cross_encoder = _get_cross_encoder()

    jd_query = _build_jd_query(parsed_jd)

    # Build (query, document) pairs for the cross-encoder
    pairs = [
        (jd_query, _build_candidate_summary(c))
        for c in candidates
    ]

    scores = cross_encoder.predict(pairs)

    # Attach scores and sort
    reranked = []
    for candidate, score in zip(candidates, scores):
        enriched = dict(candidate)
        enriched["rerank_score"] = round(float(score), 6)
        reranked.append(enriched)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    if top_k is not None:
        reranked = reranked[:top_k]

    # Update rank numbers
    for i, c in enumerate(reranked):
        c["rank"] = i + 1

    return reranked


def rerank_candidates_node(state: dict) -> dict:
    """
    LangGraph node: Re-rank retrieved candidates using CrossEncoder.

    Reads:  state["retrieved_candidates"]
            state["parsed_jd"]
    Writes: state["reranked_candidates"]
    """
    candidates: list[dict] = state.get("retrieved_candidates", [])
    parsed_jd:  dict | None = state.get("parsed_jd")

    with NodeTimer("rerank_candidates_node", state) as timer:
        if not candidates or not parsed_jd:
            timer.extra = {"error": "no_candidates_or_jd"}
            return {"reranked_candidates": []}

        from config.settings import TOP_K_RERANK, TOP_K_FINAL

        reranked = rerank_candidates(
            candidates=candidates,
            parsed_jd=parsed_jd,
            top_k=TOP_K_FINAL,
        )

        timer.extra = {
            "input_count":  len(candidates),
            "output_count": len(reranked),
            "top_score":    reranked[0]["rerank_score"] if reranked else None,
            "bottom_score": reranked[-1]["rerank_score"] if reranked else None,
        }

    return {"reranked_candidates": reranked}
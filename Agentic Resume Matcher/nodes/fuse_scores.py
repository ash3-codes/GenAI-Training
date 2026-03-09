"""
nodes/fuse_scores.py
--------------------
Final Score Fusion Node — LangGraph node.

Combines semantic similarity, skill match, experience, and ATS scores
into a single final_score for ranking in the UI.

Weights (from config.yaml):
  semantic_similarity: 0.40
  skill_match:         0.30
  experience_score:    0.20
  ats_score:           0.10

All component scores are normalised to [0.0, 1.0] before weighting.
Missing scores default to 0.0 (with warning logged).
"""

from typing import Any
from utils.logger import NodeTimer, log_info


def _normalise_rerank_score(score: float) -> float:
    """
    CrossEncoder scores are typically in [-10, 10].
    Sigmoid-normalise to [0, 1].
    """
    import math
    try:
        return round(1.0 / (1.0 + math.exp(-score)), 4)
    except OverflowError:
        return 0.0 if score < 0 else 1.0


def fuse_scores(
    candidate: dict[str, Any],
    ats_score: float,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute the final fused score for a single candidate.

    Component sources:
      semantic_similarity → candidate["semantic_score"]   (vector search cosine, [0,1])
      skill_match         → candidate["rrf_score"]        (proxy — already normalised by RRF)
      experience_score    → normalised rerank_score       (CrossEncoder sigmoid-normalised)
      ats_score           → from score_ats_node

    Returns float in [0.0, 1.0].
    """
    if weights is None:
        from config.settings import FINAL_SCORE_WEIGHTS
        weights = FINAL_SCORE_WEIGHTS

    # Validate weights sum
    w_sum = sum(weights.values())
    assert abs(w_sum - 1.0) < 0.01, f"Weights must sum to 1.0, got {w_sum}"

    # Extract component scores
    semantic  = float(candidate.get("semantic_score", 0.0) or 0.0)
    rrf       = float(candidate.get("rrf_score", 0.0) or 0.0)
    rerank    = _normalise_rerank_score(float(candidate.get("rerank_score", 0.0) or 0.0))
    ats       = float(ats_score or 0.0)

    # Normalise RRF score to [0,1] — RRF max for a single list of 20 is 1/(60+1) ≈ 0.016
    # Scale by 20x so top RRF candidate ≈ 0.33; apply min/max cap
    rrf_normalised = min(rrf * 20, 1.0)

    final = (
        weights.get("semantic_similarity", 0.40) * min(semantic, 1.0) +
        weights.get("skill_match",         0.30) * rrf_normalised     +
        weights.get("experience_score",    0.20) * rerank             +
        weights.get("ats_score",           0.10) * min(ats, 1.0)
    )

    return round(max(0.0, min(1.0, final)), 4)


def fuse_scores_node(state: dict) -> dict:
    """
    LangGraph node: Compute final scores for all reranked candidates.

    Reads:  state["reranked_candidates"]
            state["ats_scores"]
    Writes: state["final_scores"] — list of candidate dicts with final_score,
                                    sorted by final_score descending
    """
    candidates: list[dict] = state.get("reranked_candidates", [])
    ats_scores: dict        = state.get("ats_scores", {})

    with NodeTimer("fuse_scores_node", state) as timer:
        if not candidates:
            timer.extra = {"error": "no_candidates"}
            return {"final_scores": []}

        from config.settings import FINAL_SCORE_WEIGHTS

        final_scores = []
        for c in candidates:
            cid        = c.get("candidate_id")
            ats        = ats_scores.get(cid, 0.0)
            final      = fuse_scores(c, ats_score=ats, weights=FINAL_SCORE_WEIGHTS)

            enriched = dict(c)
            enriched["ats_score"]        = ats
            enriched["final_score"]      = final
            enriched["rerank_score_raw"] = c.get("rerank_score", 0.0)
            enriched["rerank_score"]     = _normalise_rerank_score(c.get("rerank_score", 0.0))
            final_scores.append(enriched)

        # Sort by final_score descending
        final_scores.sort(key=lambda x: x["final_score"], reverse=True)
        for i, c in enumerate(final_scores):
            c["final_rank"] = i + 1

        score_vals = [c["final_score"] for c in final_scores]
        timer.extra = {
            "candidates_scored": len(final_scores),
            "top_score":    max(score_vals) if score_vals else 0,
            "bottom_score": min(score_vals) if score_vals else 0,
        }

    return {"final_scores": final_scores}
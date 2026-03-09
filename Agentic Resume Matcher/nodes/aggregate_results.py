"""
nodes/aggregate_results.py
--------------------------
Result Aggregator Node -- LangGraph node.

Final step of the query pipeline. Takes final_scores (already fused + ranked)
and assembles the clean output dict the Streamlit UI renders.

Responsibilities:
  - Trim to TOP_K_FINAL candidates
  - Ensure every candidate dict has all required UI fields (with defaults)
  - Build a summary dict: top candidate, avg score, skill coverage
  - Write results_summary to state for dashboard display
"""

from utils.logger import NodeTimer


_REQUIRED_FIELDS = [
    "candidate_id", "name", "final_score", "final_rank",
    "ats_score", "rerank_score", "semantic_score", "bm25_score",
    "rrf_score", "sources", "payload",
]

_DEFAULTS = {
    "candidate_id":   "unknown",
    "name":           "Unknown Candidate",
    "final_score":    0.0,
    "final_rank":     0,
    "ats_score":      0.0,
    "rerank_score":   0.0,
    "semantic_score": 0.0,
    "bm25_score":     0.0,
    "rrf_score":      0.0,
    "sources":        [],
    "payload":        {},
}


def _normalise_candidate(c: dict) -> dict:
    """Ensure all required fields exist with sensible defaults."""
    result = dict(c)
    for field, default in _DEFAULTS.items():
        if field not in result or result[field] is None:
            result[field] = default
    # Ensure numeric fields are actual floats
    for f in ("final_score", "ats_score", "rerank_score",
              "semantic_score", "bm25_score", "rrf_score"):
        try:
            result[f] = float(result[f])
        except (TypeError, ValueError):
            result[f] = 0.0
    return result


def aggregate_results_node(state: dict) -> dict:
    """
    LangGraph node: Aggregate and finalise ranked candidates for UI display.

    Reads:  state["final_scores"]  -- list of scored candidate dicts
            state["parsed_jd"]     -- for summary stats
    Writes: state["final_scores"]  -- normalised, trimmed list
            state["results_summary"] -- dict for dashboard display
    """
    raw_final  = state.get("final_scores") or []
    parsed_jd  = state.get("parsed_jd") or {}

    with NodeTimer("aggregate_results_node", state) as timer:
        from config.settings import TOP_K_FINAL

        # Normalise all candidates
        candidates = [_normalise_candidate(c) for c in raw_final]

        # Sort by final_score descending (should already be sorted, but guarantee it)
        candidates.sort(key=lambda c: c["final_score"], reverse=True)

        # Trim to TOP_K_FINAL
        candidates = candidates[:TOP_K_FINAL]

        # Re-assign final_rank after trim
        for i, c in enumerate(candidates):
            c["final_rank"] = i + 1

        # Build summary for dashboard
        req_skills = {(s.get("skill") or "").lower()
                      for s in (parsed_jd.get("required_skills") or [])}

        if candidates:
            top    = candidates[0]
            avg_fs = round(sum(c["final_score"] for c in candidates) / len(candidates), 3)

            # Skill coverage: what fraction of JD required skills appear in top candidate
            top_skills = {s.lower() for s in (top.get("payload") or {}).get("skills") or []}
            skill_cov  = round(len(req_skills & top_skills) / max(len(req_skills), 1), 2)

            summary = {
                "total_candidates": len(candidates),
                "top_candidate":    top.get("name"),
                "top_score":        top.get("final_score"),
                "avg_score":        avg_fs,
                "skill_coverage":   skill_cov,
                "jd_title":         parsed_jd.get("title") or "Untitled",
            }
        else:
            summary = {
                "total_candidates": 0,
                "top_candidate":    None,
                "top_score":        0.0,
                "avg_score":        0.0,
                "skill_coverage":   0.0,
                "jd_title":         parsed_jd.get("title") or "Untitled",
            }

        timer.extra = {
            "candidates_in":  len(raw_final),
            "candidates_out": len(candidates),
            "top_score":      summary["top_score"],
        }

    return {
        "final_scores":    candidates,
        "results_summary": summary,
    }
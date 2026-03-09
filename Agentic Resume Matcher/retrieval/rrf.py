"""
retrieval/rrf.py
----------------
Reciprocal Rank Fusion (RRF) — rank-based score fusion.

RRF formula: score(d) = Σ 1 / (k + rank(d))
  where k=60 (literature default) and rank is 1-indexed.

Why RRF:
  - Rank-based, so vector cosine scores and BM25 raw scores
    don't need to be on the same scale
  - Robust: a candidate ranked #1 in one list and #5 in another
    beats one ranked #3 in both
  - Simple, no hyperparameters beyond k
"""

from typing import Any


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Merge multiple ranked result lists using RRF.

    Args:
        result_lists: Each list is a ranked list of candidate dicts,
                      each dict must have a "candidate_id" key.
                      Dicts from different lists for the same candidate
                      are merged (last one wins for payload/name).
        k:            RRF constant (default 60)
        top_k:        Truncate output to top_k results (None = return all)

    Returns:
        Merged list of candidate dicts sorted by RRF score descending.
        Each dict has all original fields plus:
          "rrf_score":   float — the fused score
          "sources":     list[str] — which retrievers found this candidate
    """
    rrf_scores: dict[str, float] = {}
    candidate_data: dict[str, dict] = {}
    candidate_sources: dict[str, set] = {}

    for result_list in result_lists:
        for rank_0based, candidate in enumerate(result_list):
            cid = candidate.get("candidate_id")
            if not cid:
                continue

            rank_1based = rank_0based + 1
            rrf_scores[cid]      = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank_1based)
            candidate_data[cid]  = candidate   # last list's data wins (all have same payload)
            candidate_sources.setdefault(cid, set()).add(candidate.get("source", "unknown"))

    # Build output sorted by RRF score descending
    sorted_cids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    if top_k is not None:
        sorted_cids = sorted_cids[:top_k]

    results = []
    for rank_0based, cid in enumerate(sorted_cids):
        data = dict(candidate_data[cid])   # copy
        data["rrf_score"] = round(rrf_scores[cid], 8)
        data["sources"]   = sorted(candidate_sources[cid])
        data["rank"]      = rank_0based + 1
        results.append(data)

    return results
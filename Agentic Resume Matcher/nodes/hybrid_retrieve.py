"""
nodes/hybrid_retrieve.py
------------------------
Hybrid Retrieval Node — LangGraph node.

Combines:
  1. Vector search  (semantic similarity via Qdrant)
  2. BM25 search    (keyword matching over payloads)
  3. RRF fusion     (rank-based merging)

Qdrant cloud can time out under load — both calls are retried up to 3x
with 2s backoff before raising. The client also has a 30s timeout set
in get_qdrant_client().
"""

import time
from utils.logger import NodeTimer


def _with_retry(fn, retries=3, backoff=2.0):
    """Call fn(), retrying on any exception up to `retries` times."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_err


def hybrid_retrieve_node(state: dict) -> dict:
    """
    LangGraph node: Retrieve top candidates using hybrid search.

    Reads:  state["jd_embedding"]  — vector from embed_jd_node
            state["parsed_jd"]     — for BM25 query terms and metadata filter
    Writes: state["retrieved_candidates"] — top N candidates with payloads
    """
    jd_embedding: list[float] | None = state.get("jd_embedding")
    parsed_jd:    dict | None        = state.get("parsed_jd")

    with NodeTimer("hybrid_retrieve_node", state) as timer:
        if not jd_embedding or not parsed_jd:
            timer.extra = {"error": "missing_jd_embedding_or_parsed_jd"}
            return {"retrieved_candidates": []}

        from config.settings import TOP_K_VECTOR, TOP_K_BM25, TOP_K_FINAL, RRF_K
        from retrieval.vector_search import vector_search
        from retrieval.bm25_search   import bm25_search
        from retrieval.rrf           import reciprocal_rank_fusion
        from schemas.jd_schema       import JobDescriptionSchema
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)
            clean_jd = {k: v for k, v in parsed_jd.items() if k != "parsed"}
            jd_obj = JobDescriptionSchema(**clean_jd)

        # ── 1. Vector search (with retry) ─────────────────────────────────────
        vector_results = _with_retry(lambda: vector_search(
            query_vector=jd_embedding,
            top_k=TOP_K_VECTOR,
        ))

        # ── 2. BM25 search (with retry) ───────────────────────────────────────
        skill_names  = jd_obj.get_required_skill_names()
        bm25_results = _with_retry(lambda: bm25_search(
            jd_skill_names=skill_names,
            top_k=TOP_K_BM25,
        ))

        # ── 3. RRF fusion ─────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            result_lists=[vector_results, bm25_results],
            k=RRF_K,
            top_k=TOP_K_FINAL * 4,
        )

        # ── 4. Enrich with individual scores ─────────────────────────────────
        vector_score_map = {r["candidate_id"]: r["score"] for r in vector_results}
        bm25_score_map   = {r["candidate_id"]: r["score"] for r in bm25_results}

        candidates = []
        for c in fused:
            cid = c["candidate_id"]
            enriched = {
                **c,
                "semantic_score": vector_score_map.get(cid, 0.0),
                "bm25_score":     bm25_score_map.get(cid, 0.0),
            }
            candidates.append(enriched)

        timer.extra = {
            "vector_results":  len(vector_results),
            "bm25_results":    len(bm25_results),
            "fused_results":   len(candidates),
            "jd_title":        jd_obj.title,
        }

    return {"retrieved_candidates": candidates}
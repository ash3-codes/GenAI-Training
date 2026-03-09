"""
retrieval/vector_search.py
--------------------------
Vector search against Qdrant resumes_index using cosine similarity.

Returns ranked candidate IDs + scores + payloads.
Supports optional metadata pre-filtering (skills, experience).
"""

from typing import Any
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range, QueryRequest


def vector_search(
    query_vector: list[float],
    top_k: int = 20,
    filter_skills: list[str] | None = None,
    filter_min_experience_months: int | None = None,
    qdrant_client=None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search resumes_index by vector similarity.

    Args:
        query_vector:                 JD embedding vector (1536 dims)
        top_k:                        Number of results to return
        filter_skills:                If set, only return candidates with ANY of these skills
        filter_min_experience_months: If set, only return candidates with >= this experience
        qdrant_client:                Injected for testability
        collection_name:              Override collection name

    Returns:
        List of dicts: {candidate_id, name, score, payload, rank}
        Sorted by score descending.

    Note: Uses query_points() — qdrant-client >= 1.7 (replaces deprecated search()).
    """
    if qdrant_client is None:
        from config.settings import get_qdrant_client
        qdrant_client = get_qdrant_client()
    if collection_name is None:
        from config.settings import QDRANT_RESUME_COLLECTION
        collection_name = QDRANT_RESUME_COLLECTION

    # Build optional filter
    must_conditions = []

    if filter_skills:
        must_conditions.append(
            FieldCondition(
                key="skills",
                match=MatchAny(any=filter_skills),
            )
        )

    if filter_min_experience_months is not None:
        must_conditions.append(
            FieldCondition(
                key="total_experience_months",
                range=Range(gte=filter_min_experience_months),
            )
        )

    search_filter = Filter(must=must_conditions) if must_conditions else None

    # query_points() replaces the deprecated search() in qdrant-client >= 1.7
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    # query_points returns a QueryResponse with a .points attribute
    points = response.points if hasattr(response, "points") else response

    return [
        {
            "candidate_id": p.payload.get("candidate_id"),
            "name":         p.payload.get("name"),
            "score":        round(float(p.score), 6),
            "payload":      p.payload,
            "rank":         i + 1,
            "source":       "vector",
        }
        for i, p in enumerate(points)
    ]
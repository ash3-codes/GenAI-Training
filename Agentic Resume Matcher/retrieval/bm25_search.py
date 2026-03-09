"""
retrieval/bm25_search.py
------------------------
BM25 keyword search over resume payloads fetched from Qdrant.

Strategy:
  1. Scroll all resume payloads from Qdrant (no vector needed)
  2. Build BM25 corpus from skill names + experience summaries
  3. Tokenise JD required skill names as the query
  4. Return ranked candidate IDs

Uses BM25Plus (not BM25Okapi) because BM25Okapi IDF = log((N-df+0.5)/(df+0.5))
which equals 0 when df = N/2 — this silently zeros out all scores when a skill
appears in exactly half the corpus (common with small resume sets).
BM25Plus IDF = log((N+1)/df) is always > 0, making it robust for any corpus size.

Note: scrolling all payloads is fine for up to ~10k resumes.
For larger corpora, add a pre-filter or use Qdrant's sparse vectors.
"""

from typing import Any
from rank_bm25 import BM25Plus


def _build_corpus_text(payload: dict) -> str:
    """
    Build a single text string from a resume payload for BM25 indexing.
    Combines skills and all text fields to maximise recall.
    """
    parts = []

    # Skill names — repeat proportional to experience (rough TF boost)
    for skill in payload.get("skills", []):
        years = payload.get("skill_years", {}).get(skill, 1)
        repeat = max(1, int(years))
        parts.extend([skill.lower()] * repeat)

    # Education
    for degree in payload.get("education_degrees", []):
        parts.append(degree.lower())

    # Certifications
    for cert in payload.get("certifications", []):
        parts.extend(cert.lower().split())

    # Name (for reference, low weight)
    name = payload.get("name", "")
    if name:
        parts.extend(name.lower().split())

    return " ".join(parts)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser."""
    import re
    return re.findall(r"[a-z0-9#+.\-]+", text.lower())


def bm25_search(
    jd_skill_names: list[str],
    top_k: int = 20,
    qdrant_client=None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    BM25 search over resume payloads.

    Args:
        jd_skill_names:  Required skill names from the JD (used as BM25 query)
        top_k:           Number of top results to return
        qdrant_client:   Injected for testability
        collection_name: Override collection name

    Returns:
        List of dicts: {candidate_id, name, score, payload, rank, source}
        Sorted by BM25 score descending.
    """
    if qdrant_client is None:
        from config.settings import get_qdrant_client
        qdrant_client = get_qdrant_client()
    if collection_name is None:
        from config.settings import QDRANT_RESUME_COLLECTION
        collection_name = QDRANT_RESUME_COLLECTION

    # Scroll all payloads (no vector needed).
    # Confirmed via debug: qdrant-client 1.17 sync scroll() returns a plain
    # tuple: (List[Record], Optional[PointId]).
    all_payloads: list[dict] = []
    offset = None

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in points:
            if record.payload:
                all_payloads.append(record.payload)

        if not next_offset:
            break
        offset = next_offset

    if not all_payloads:
        return []

    # Build BM25 index
    corpus_texts = [_build_corpus_text(p) for p in all_payloads]
    tokenised_corpus = [_tokenise(text) for text in corpus_texts]
    bm25 = BM25Plus(tokenised_corpus)

    # Build query from JD skill names
    query_text = " ".join(jd_skill_names)
    query_tokens = _tokenise(query_text)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    # Rank and return top_k
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices):
        if scores[idx] <= 0:
            break   # BM25 score 0 = no keyword match at all
        payload = all_payloads[idx]
        results.append({
            "candidate_id": payload.get("candidate_id"),
            "name":         payload.get("name"),
            "score":        round(float(scores[idx]), 6),
            "payload":      payload,
            "rank":         rank + 1,
            "source":       "bm25",
        })

    return results
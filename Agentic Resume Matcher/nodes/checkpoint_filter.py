"""
nodes/checkpoint_filter.py
--------------------------
Checkpoint Filter Node — LangGraph node.

Sits between load_documents and parse_resume.
Queries Qdrant for all already-indexed file_names and removes matching
documents from raw_resume_texts BEFORE the LLM parse step.

This is the biggest speed win for large re-ingestion runs:
  - Current: 800 files × 12s LLM parse = 160 min, then dedup at embed step
  - With checkpoint: only NEW files reach parse_resume, existing ones skip LLM entirely

State reads:  raw_resume_texts   — list of loaded document dicts
State writes: raw_resume_texts   — filtered list (new files only)
              skipped_checkpoint — count of files skipped
              failed_docs        — unchanged (nothing fails here)
"""

from utils.logger import NodeTimer


def checkpoint_filter_node(state: dict) -> dict:
    """
    LangGraph node: Remove already-indexed resumes before LLM parsing.

    Queries Qdrant resumes_index for all stored file_names.
    Any loaded document whose file_name is already in Qdrant is dropped
    from raw_resume_texts so parse_resume_node never sees it.

    This node is a no-op if Qdrant is unreachable — it logs a warning
    and passes all documents through rather than crashing.
    """
    raw_docs: list[dict] = state.get("raw_resume_texts", [])

    with NodeTimer("checkpoint_filter_node", state) as timer:
        # ── Fetch all already-indexed file_names from Qdrant ─────────────────
        indexed: set[str] = set()
        try:
            from config.settings import get_qdrant_client, QDRANT_RESUME_COLLECTION

            client = get_qdrant_client()
            offset = None
            while True:
                result = client.scroll(
                    collection_name=QDRANT_RESUME_COLLECTION,
                    scroll_filter=None,
                    limit=250,           # large page — fewer round trips
                    offset=offset,
                    with_payload=["file_name"],
                    with_vectors=False,
                )
                points, next_offset = result
                for pt in points:
                    fn = (pt.payload or {}).get("file_name")
                    if fn:
                        indexed.add(fn)
                if not next_offset:
                    break
                offset = next_offset

        except Exception as exc:
            # Qdrant unavailable — pass everything through, dedup will catch dupes
            timer.extra = {
                "qdrant_error":    str(exc)[:120],
                "indexed_count":   0,
                "skipped_count":   0,
                "passing_count":   len(raw_docs),
            }
            return {
                "raw_resume_texts":   raw_docs,
                "skipped_checkpoint": 0,
            }

        # ── Filter ────────────────────────────────────────────────────────────
        passing  = []
        skipped  = []
        for doc in raw_docs:
            if doc.get("file_name") in indexed:
                skipped.append(doc["file_name"])
            else:
                passing.append(doc)

        timer.extra = {
            "indexed_in_qdrant": len(indexed),
            "docs_in":           len(raw_docs),
            "skipped_count":     len(skipped),
            "passing_count":     len(passing),
        }

    return {
        "raw_resume_texts":   passing,
        "skipped_checkpoint": len(skipped),
    }
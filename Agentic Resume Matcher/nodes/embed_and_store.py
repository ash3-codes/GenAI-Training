"""
nodes/embed_and_store.py
------------------------
Embedding & Qdrant Storage Node — LangGraph node.

Takes skill-expanded resume dicts, builds embedding text, generates
vectors via AzureOpenAI, and upserts into Qdrant resumes_index collection.

Design decisions:
  - Collection is created if it doesn't exist (idempotent)
  - Uses UPSERT not INSERT — re-running ingestion won't create duplicates
  - Point ID in Qdrant = candidate_id (UUID) so we can retrieve by ID later
  - Batch size 10 — balances API rate limits vs. round trips
  - Pydantic serializer warning suppressed at call site (LangChain internal)
"""

import warnings
import uuid
from typing import Any

from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)

from utils.logger import NodeTimer
from utils.embed_template import build_resume_embedding_text


# ── Collection initialisation ─────────────────────────────────────────────────

def ensure_resume_collection(client, collection_name: str, vector_size: int) -> None:
    """
    Create the resumes_index collection if it doesn't exist.
    Safe to call on every startup — no-op if already present.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def ensure_jd_collection(client, collection_name: str, vector_size: int) -> None:
    """Create jd_index collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


# ── Core embed+store function ─────────────────────────────────────────────────

def embed_and_store_resumes(
    resume_dicts: list[dict[str, Any]],
    embedding_model=None,
    qdrant_client=None,
    collection_name: str | None = None,
    vector_size: int | None = None,
    batch_size: int = 50,    # increased from 10 — fewer API round trips
) -> tuple[int, list[dict]]:
    """
    Embed resume dicts and upsert into Qdrant.

    Args:
        resume_dicts:     List of validated ResumeSchema dicts (post skill-expansion)
        embedding_model:  AzureOpenAIEmbeddings instance (injected for testability)
        qdrant_client:    QdrantClient instance
        collection_name:  Override collection name (uses config default if None)
        vector_size:      Override vector size
        batch_size:       How many resumes to embed per API call

    Returns:
        (stored_count, failed_list)
        failed_list contains dicts with file_name, error, reason
    """
    from schemas.resume_schema import ResumeSchema

    if embedding_model is None:
        from config.settings import get_embedding_model
        embedding_model = get_embedding_model()
    if qdrant_client is None:
        from config.settings import get_qdrant_client
        qdrant_client = get_qdrant_client()
    if collection_name is None:
        from config.settings import QDRANT_RESUME_COLLECTION
        collection_name = QDRANT_RESUME_COLLECTION
    if vector_size is None:
        from config.settings import QDRANT_VECTOR_SIZE
        vector_size = QDRANT_VECTOR_SIZE

    # Ensure collection exists
    ensure_resume_collection(qdrant_client, collection_name, vector_size)

    # ── Deduplication by file_name ────────────────────────────────────────────
    # Scroll existing payloads to build a set of already-ingested file_names.
    # This prevents re-embedding the same file if ingestion is re-run.
    existing_file_names: set[str] = set()
    try:
        offset = None
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=100,
                offset=offset,
                with_payload=["file_name"],
                with_vectors=False,
            )
            points, next_offset = scroll_result
            for pt in points:
                fn = (pt.payload or {}).get("file_name")
                if fn:
                    existing_file_names.add(fn)
            if not next_offset:
                break
            offset = next_offset
    except Exception:
        pass  # If scroll fails, proceed without dedup (better than crashing)

    # Filter out already-ingested resumes
    skipped_dupes = 0
    deduped_dicts = []
    for rd in resume_dicts:
        fname = rd.get("file_name", "")
        if fname and fname in existing_file_names:
            skipped_dupes += 1
        else:
            deduped_dicts.append(rd)
    resume_dicts = deduped_dicts

    stored  = 0
    failed  = []

    # Process in batches
    for batch_start in range(0, len(resume_dicts), batch_size):
        batch = resume_dicts[batch_start : batch_start + batch_size]

        # Build resume objects and embedding texts
        resume_objs  = []
        embed_texts  = []
        valid_batch  = []

        for rd in batch:
            try:
                # Suppress Pydantic serializer warning from LangChain internals
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*PydanticSerializationUnexpectedValue.*",
                        category=UserWarning,
                    )
                    # Strip any LangChain-injected 'parsed' field before validation
                    clean_rd = {k: v for k, v in rd.items() if k != "parsed"}
                    resume_obj = ResumeSchema(**clean_rd)

                embed_text = build_resume_embedding_text(resume_obj)
                resume_objs.append(resume_obj)
                embed_texts.append(embed_text)
                valid_batch.append(rd)
            except Exception as e:
                failed.append({
                    "file_name": rd.get("file_name", "unknown"),
                    "error":     type(e).__name__,
                    "reason":    f"Pre-embedding validation failed: {str(e)[:200]}",
                    "stage":     "embed_and_store_node",
                })

        if not embed_texts:
            continue

        # Generate embeddings for the batch
        try:
            vectors = embedding_model.embed_documents(embed_texts)
        except Exception as e:
            for rd in valid_batch:
                failed.append({
                    "file_name": rd.get("file_name", "unknown"),
                    "error":     type(e).__name__,
                    "reason":    f"Embedding API call failed: {str(e)[:200]}",
                    "stage":     "embed_and_store_node",
                })
            continue

        # Build Qdrant points
        points = []
        for resume_obj, vector in zip(resume_objs, vectors):
            # Use candidate_id as Qdrant point ID (UUID → Qdrant UUID format)
            point_id = str(uuid.UUID(resume_obj.candidate_id))
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=resume_obj.to_qdrant_payload(),
            ))

        # Upsert (idempotent — safe to re-run)
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            stored += len(points)
        except Exception as e:
            for rd in valid_batch:
                failed.append({
                    "file_name": rd.get("file_name", "unknown"),
                    "error":     type(e).__name__,
                    "reason":    f"Qdrant upsert failed: {str(e)[:200]}",
                    "stage":     "embed_and_store_node",
                })

    return stored, failed, skipped_dupes


# ── LangGraph node ────────────────────────────────────────────────────────────

def embed_and_store_node(state: dict) -> dict:
    """
    LangGraph node: Embed expanded resumes and store in Qdrant.

    Reads:  state["expanded_resumes"] — skill-expanded, validated resume dicts
    Writes: (no state change — Qdrant is the output)
            state["failed_docs"]      — appends embedding/storage failures

    Note: This node doesn't modify any state keys used downstream.
    The ingestion pipeline ends here. The query pipeline starts fresh
    from Qdrant at retrieve time.
    """
    expanded: list[dict] = state.get("expanded_resumes", [])
    existing_failed: list = state.get("failed_docs", [])

    with NodeTimer("embed_and_store_node", state) as timer:
        stored_count, new_failures, skipped_dupes = embed_and_store_resumes(expanded)
        all_failed = list(existing_failed) + new_failures

        timer.extra = {
            "resumes_to_embed": len(expanded),
            "skipped_dupes":    skipped_dupes,
            "stored_count":     stored_count,
            "embed_failures":   len(new_failures),
        }

    return {"failed_docs": all_failed}
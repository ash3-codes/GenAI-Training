"""
graphs/state.py
---------------
Shared LangGraph state TypedDict.

All nodes read from and write to this single state object.
LangGraph merges node outputs into state automatically.

Design:
  - Ingestion pipeline populates: resume_file_paths → ... → expanded_resumes (stored in Qdrant)
  - Query pipeline populates:     jd_raw_text → ... → final_scores
  - node_logs and failed_docs accumulate across all nodes
  - All fields are Optional so either pipeline can run independently
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Ingestion inputs ──────────────────────────────────────
    resume_file_paths:  list[str]          # Paths to resume files on disk

    # ── Ingestion intermediates ───────────────────────────────
    raw_resume_texts:    list[dict]        # {file_name, text, upload_time, ...}
    skipped_checkpoint:  int               # Files skipped by checkpoint_filter (already in Qdrant)
    parsed_resumes:      list[dict]        # ResumeSchema dicts (pre/post validation)
    expanded_resumes:    list[dict]        # ResumeSchema dicts with expanded skills

    # ── Query inputs ──────────────────────────────────────────
    jd_raw_text:        str                # Raw JD text (from UI or file)
    jd_file_name:       Optional[str]      # Optional: source filename
    jd_upload_time:     Optional[str]      # Optional: ISO datetime

    # ── Query intermediates ───────────────────────────────────
    parsed_jd:          Optional[dict]     # JobDescriptionSchema dict
    jd_embedding:       Optional[list]     # list[float] — 1536 dims

    # ── Retrieval & ranking ───────────────────────────────────
    retrieved_candidates: list[dict]       # After hybrid retrieval + RRF
    reranked_candidates:  list[dict]       # After CrossEncoder reranking
    ats_scores:           dict[str, float] # {candidate_id: ats_score}
    final_scores:         list[dict]       # Final ranked candidates (UI output)

    # ── Cross-cutting ─────────────────────────────────────────
    failed_docs:        list[dict]         # {file_name, error, reason, stage}
    node_logs:          list[dict]         # Execution log entries per node
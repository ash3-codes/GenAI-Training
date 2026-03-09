"""
nodes/validate_schema.py
------------------------
Validation Node — LangGraph node.

Validates parsed resume dicts against ResumeSchema using Pydantic.
For documents that fail validation, attempts LLM re-parse up to MAX_RETRIES
times with exponential backoff before sending to failed_docs.

Retry logic:
  - Attempt 1: validate as-is
  - Attempt 2: ask LLM to fix the specific validation error
  - Attempt 3: ask LLM to re-parse from scratch (fallback)
  - After 3 failures: move to failed_docs with detailed reason

Error classification:
  - "schema_error":  Pydantic validation failure (fixable by LLM retry)
  - "missing_name":  name field is empty/missing (unrecoverable)
  - "empty_doc":     all fields empty (unrecoverable, don't retry)
"""

import time
from typing import Any

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.logger import NodeTimer
import json
from schemas.resume_schema import ResumeSchema

# ── Error classification ──────────────────────────────────────────────────────

def classify_error(error: ValidationError) -> str:
    """Classify a Pydantic ValidationError to decide whether to retry."""
    errors = error.errors()
    error_types = {e.get("type", "") for e in errors}
    error_locs  = {str(e.get("loc", "")) for e in errors}

    # Unrecoverable: name is missing or empty
    if any("name" in loc for loc in error_locs):
        return "missing_name"

    # Likely fixable by LLM: type coercion issues
    if error_types & {"string_type", "int_type", "float_type", "list_type", "value_error"}:
        return "schema_error"

    return "schema_error"


# ── Repair prompt ─────────────────────────────────────────────────────────────

REPAIR_PROMPT = """The following JSON failed Pydantic schema validation.
Fix ONLY the fields that caused errors. Keep all other fields unchanged.
Return valid JSON only — no markdown, no explanation.

VALIDATION ERRORS:
{errors}

CURRENT JSON:
{current_json}

SCHEMA REQUIREMENTS:
- skills: must be a list of {{"skill": str, "experience_years": float}}
- education: must be a list of {{"degree": str, "university": str}}
- experience: must be a list of {{"role": str, "summary": str}}
- certifications: must be a list of strings
- experience_years and duration_months must be numbers (not strings)
- name: must be a non-empty string
"""


# ── Core validation with retry ────────────────────────────────────────────────

def validate_and_repair(
    resume_dict: dict[str, Any],
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    llm=None,
) -> tuple[dict | None, str | None]:
    """
    Validate a resume dict and attempt LLM repair if invalid.

    Returns:
        (valid_dict, None)      on success
        (None, error_reason)    if all attempts fail
    """


    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            validated = ResumeSchema(**resume_dict)
            return validated.model_dump(), None  # Success

        except ValidationError as e:
            last_error = e
            error_type = classify_error(e)

            # Unrecoverable errors — don't waste LLM calls
            if error_type == "missing_name":
                return None, f"Unrecoverable: candidate name is missing or empty. Errors: {e.error_count()} field(s)"

            # Last attempt — give up
            if attempt == max_retries:
                break

            # Retry with LLM repair
            if llm is None:
                from config.settings import get_chat_llm
                llm = get_chat_llm(max_tokens=2000)

            error_summary = "; ".join(
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )

            prompt = REPAIR_PROMPT.format(
                errors=error_summary,
                current_json=json.dumps(resume_dict, indent=2)[:3000],
            )

            try:
                response = llm.invoke(prompt)
                repaired_text = response.content.strip()
                # Strip markdown code fences if present
                if repaired_text.startswith("```"):
                    lines = repaired_text.split("\n")
                    repaired_text = "\n".join(
                        l for l in lines if not l.startswith("```")
                    )
                resume_dict = json.loads(repaired_text)
            except Exception as repair_err:
                # Repair itself failed — backoff and try again as-is
                pass

            # Exponential backoff: 2s, 4s, 8s
            sleep_time = backoff_seconds * (2 ** (attempt - 1))
            time.sleep(sleep_time)

    # All attempts exhausted
    error_msgs = "; ".join(
        f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}"
        for e in last_error.errors()
    ) if last_error else "Unknown validation error"

    return None, f"Failed after {max_retries} attempts. Last errors: {error_msgs}"


# ── LangGraph node ────────────────────────────────────────────────────────────

def validate_schema_node(state: dict) -> dict:
    """
    LangGraph node: Validate all parsed resumes against ResumeSchema.

    Reads:  state["parsed_resumes"]   — list of dicts from parse_resume_node
    Writes: state["parsed_resumes"]   — only the valid, Pydantic-validated dicts
            state["failed_docs"]      — appends validation failures

    After this node, every dict in parsed_resumes is guaranteed to be
    a valid ResumeSchema (all types correct, required fields present).
    """
    parsed_resumes: list[dict] = state.get("parsed_resumes", [])
    existing_failed: list      = state.get("failed_docs", [])

    with NodeTimer("validate_schema_node", state) as timer:
        from config.settings import MAX_RETRIES, RETRY_BACKOFF_SECONDS
        llm = None  # Lazy-initialised on first repair attempt

        valid_resumes = []
        failed        = list(existing_failed)
        repaired      = 0
        rejected      = 0

        for resume_dict in parsed_resumes:
            file_name = resume_dict.get("file_name", "unknown")

            valid_dict, error_reason = validate_and_repair(
                resume_dict=resume_dict,
                max_retries=MAX_RETRIES,
                backoff_seconds=RETRY_BACKOFF_SECONDS,
                llm=llm,
            )

            if valid_dict is not None:
                valid_resumes.append(valid_dict)
                # Track if it needed repair (original had errors but was fixed)
            else:
                rejected += 1
                failed.append({
                    "file_name": file_name,
                    "error":     "ValidationError",
                    "reason":    error_reason,
                    "stage":     "validate_schema_node",
                })

        timer.extra = {
            "validated_count": len(valid_resumes),
            "rejected_count":  rejected,
        }

    return {
        "parsed_resumes": valid_resumes,
        "failed_docs":    failed,
    }
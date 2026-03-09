"""
nodes/parse_resume.py
---------------------
Resume Parsing Node — LangGraph node.

Takes raw text documents from load_documents_node and calls AzureOpenAI
to extract structured data matching ResumeSchema.

Design decisions:
  - Uses langchain structured output (.with_structured_output) for clean JSON
  - candidate_id is injected AFTER parsing (never trusted from LLM)
  - file_name and upload_time are copied from the source document dict
  - Parsing failures go to failed_docs, not exceptions — pipeline continues
  - Each resume is parsed independently so one failure doesn't block others
"""

import uuid
import time
import warnings
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import NodeTimer

# Max concurrent LLM parse calls.
# Azure OpenAI typically allows 60 RPM on gpt-4o-mini / 30 RPM on gpt-4o.
# 10 workers = safe for most tiers; reduce to 5 if you hit 429 rate-limit errors.
_PARSE_WORKERS = 10

# Silence the LangChain/Pydantic "parsed: None" serializer warning globally.
# Root cause: llm.with_structured_output() wraps results in an internal
# RunnableWithStructuredOutput whose state dict has a field `parsed: None`
# (a LangChain bookkeeping slot). When Pydantic serialises the pipeline state
# it sees a ResumeSchema object in that None-typed slot and warns. The actual
# parsed data is correct — this is purely a serialisation warning from
# LangChain internals and carries no functional impact.
warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)


# ── Prompt ────────────────────────────────────────────────────────────────────

PARSE_RESUME_PROMPT = """You are an expert resume parser. Extract structured information from the resume text below.

RULES:
- Extract ALL skills mentioned anywhere in the resume (skills section, experience bullets, projects)
- For experience_years: calculate from dates if given (e.g. "2019-2022" = 3 years). If no dates and no explicit years mentioned, set experience_years to 0.0 — do NOT guess or default to 5. Only set a non-zero value if you can derive it from the resume text.
- For duration_months: calculate from start/end dates if provided; otherwise estimate
- summary: write 1-2 concise sentences, remove filler words (responsible for, worked on, etc.)
- technologies: list specific tools/frameworks mentioned in that role
- If a field is not present in the resume, use null (not empty string)
- certifications: only include formal certifications, not skills or education degrees
- Do NOT invent information — only extract what is present

CRITICAL — name field:
- Extract the candidate name ONLY if it is explicitly written in the resume text
- If you cannot find a clear candidate name, set name to the string "UNKNOWN"
- NEVER use placeholder names like "John Doe", "John Smith", "Jane Doe", or "Candidate"
- The resume filename is NOT the candidate name

RESUME TEXT:
{resume_text}
"""


# ── Parser function (pure, testable independently) ───────────────────────────

def parse_resume_text(
    resume_text: str,
    file_name: str,
    upload_time: str,
    llm=None,
) -> dict[str, Any]:
    """
    Parse raw resume text into a structured dict matching ResumeSchema.

    Args:
        resume_text:  Raw text extracted from resume file
        file_name:    Original filename (set as metadata, not from LLM)
        upload_time:  ISO datetime string (set as metadata, not from LLM)
        llm:          AzureChatOpenAI instance (injected for testability)

    Returns:
        Dict ready to be validated by ResumeSchema.
        candidate_id is always freshly generated here.

    Raises:
        Exception if LLM call fails (caller handles retry).
    """
    if llm is None:
        from config.settings import get_chat_llm
        llm = get_chat_llm()

    from schemas.resume_schema import ResumeSchema

    # Use structured output — LangChain will enforce the Pydantic schema
    structured_llm = llm.with_structured_output(ResumeSchema)

    prompt = PARSE_RESUME_PROMPT.format(resume_text=resume_text[:6000])  # token guard

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*PydanticSerializationUnexpectedValue.*")
        result: ResumeSchema = structured_llm.invoke(prompt)

    # Override metadata fields — never trust LLM for these
    result_dict = result.model_dump()
    result_dict["candidate_id"] = str(uuid.uuid4())
    result_dict["file_name"]    = file_name
    result_dict["upload_time"]  = upload_time

    # If LLM could not find a name it returns "UNKNOWN" per our prompt instruction.
    # Also catch common placeholder hallucinations as a safety net.
    _BAD_NAMES = {"unknown", "john doe", "john smith", "jane doe", "candidate", "n/a", ""}
    extracted_name = (result_dict.get("name") or "").strip()
    if extracted_name.lower() in _BAD_NAMES:
        # Try to derive name from filename (e.g. "Satya Mishra_Resume.pdf" → "Satya Mishra")
        stem = file_name.rsplit(".", 1)[0]                        # drop extension
        stem = stem.replace("_", " ").replace("-", " ")
        # Strip common suffixes: Resume, CV, QA, developer, etc.
        import re
        stem = re.sub(r"\b(resume|cv|qa|developer|engineer|consultant|profile)\b",
                      "", stem, flags=re.IGNORECASE).strip()
        # If stem looks like a real name (2+ words, no digits), use it
        parts = [p for p in stem.split() if p and not any(c.isdigit() for c in p)]
        result_dict["name"] = " ".join(parts[:3]) if len(parts) >= 2 else extracted_name or "UNKNOWN"

    return result_dict


# ── LangGraph node ────────────────────────────────────────────────────────────

def parse_resume_node(state: dict) -> dict:
    """
    LangGraph node: Parse all loaded resume documents into structured dicts.

    Reads:  state["raw_resume_texts"]  — list of document dicts from load_documents_node
    Writes: state["parsed_resumes"]    — list of ResumeSchema-valid dicts (partial, pre-validation)
            state["failed_docs"]       — appends parse failures

    Note: Output goes to parsed_resumes (not expanded_resumes). The validation
    node runs next and moves only valid docs forward.
    """
    raw_docs: list[dict]  = state.get("raw_resume_texts", [])
    existing_failed: list = state.get("failed_docs", [])

    with NodeTimer("parse_resume_node", state) as timer:
        from config.settings import get_chat_llm
        llm = get_chat_llm()   # single shared instance for all parses

        parsed    = []
        failed    = list(existing_failed)
        parse_errors = 0

        def _parse_one(doc: dict):
            """Parse a single document — runs in a thread."""
            file_name   = doc.get("file_name", "unknown")
            upload_time = doc.get("upload_time", "")
            text        = doc.get("text", "")
            try:
                result = parse_resume_text(
                    resume_text=text,
                    file_name=file_name,
                    upload_time=upload_time,
                    llm=llm,     # AzureChatOpenAI is thread-safe for concurrent calls
                )
                return ("ok", result)
            except Exception as e:
                return ("err", {
                    "file_name": file_name,
                    "error":     type(e).__name__,
                    "reason":    f"LLM parsing failed: {str(e)[:200]}",
                    "stage":     "parse_resume_node",
                })

        # Submit all documents concurrently — parse_resume_text is I/O bound
        # (waiting on Azure OpenAI API), so threading gives near-linear speedup
        # up to the API rate limit.
        with ThreadPoolExecutor(max_workers=_PARSE_WORKERS) as executor:
            futures = {executor.submit(_parse_one, doc): doc for doc in raw_docs}
            for future in as_completed(futures):
                status, payload = future.result()
                if status == "ok":
                    parsed.append(payload)
                else:
                    parse_errors += 1
                    failed.append(payload)

        timer.extra = {
            "docs_attempted": len(raw_docs),
            "docs_parsed":    len(parsed),
            "parse_errors":   parse_errors,
        }

    return {
        "parsed_resumes": parsed,
        "failed_docs":    failed,
    }
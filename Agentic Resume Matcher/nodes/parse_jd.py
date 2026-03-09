"""
nodes/parse_jd.py
-----------------
Job Description Parsing Node — LangGraph node.

Takes raw JD text and calls AzureOpenAI to extract structured data
matching JobDescriptionSchema. Much simpler than resume parsing because:
  - JDs are typically well-structured and shorter
  - No retry needed — JD is entered by HR directly
  - jd_id is always freshly generated (never from LLM)
"""

import uuid
from typing import Any

import warnings
from utils.logger import NodeTimer

warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)


PARSE_JD_PROMPT = """You are an expert at parsing job descriptions. Extract structured information from the JD below.

RULES:
- required_skills: skills explicitly listed as required/mandatory with their minimum years
  - If years not specified for a skill, set min_years to 0
- nice_to_have_skills: skills listed as preferred/nice-to-have/bonus (names only, no years)
- experience_min_years: minimum total years of experience required (integer)
- domain: the industry or technical domain (e.g. "FinTech", "Healthcare", "Backend", "Data Engineering")
- education_requirements: exact text from JD about education (null if not mentioned)
- Extract ONLY what is present — do not invent requirements

JOB DESCRIPTION:
{jd_text}
"""


def parse_jd_text(
    jd_text: str,
    file_name: str | None = None,
    upload_time: str | None = None,
    llm=None,
) -> dict[str, Any]:
    """
    Parse raw JD text into a structured dict matching JobDescriptionSchema.

    Args:
        jd_text:     Raw JD text
        file_name:   Source filename (metadata, not from LLM)
        upload_time: ISO datetime string
        llm:         AzureChatOpenAI instance (injected for testability)

    Returns:
        Dict ready to be used as JobDescriptionSchema.
    """
    from datetime import datetime, timezone

    if llm is None:
        from config.settings import get_chat_llm
        llm = get_chat_llm()

    from schemas.jd_schema import JobDescriptionSchema

    structured_llm = llm.with_structured_output(JobDescriptionSchema)
    prompt = PARSE_JD_PROMPT.format(jd_text=jd_text[:4000])  # token guard

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*PydanticSerializationUnexpectedValue.*")
        result = structured_llm.invoke(prompt)

    result_dict = result.model_dump()
    # Override metadata — never trust LLM for these
    result_dict["jd_id"]       = str(uuid.uuid4())
    result_dict["raw_text"]    = jd_text
    result_dict["file_name"]   = file_name
    result_dict["upload_time"] = upload_time or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    return result_dict


def parse_jd_node(state: dict) -> dict:
    """
    LangGraph node: Parse JD raw text into a structured JobDescriptionSchema dict.

    Reads:  state["jd_raw_text"]  — raw JD text string
    Writes: state["parsed_jd"]    — JobDescriptionSchema-valid dict
    """
    jd_text: str  = state.get("jd_raw_text", "")
    jd_file: str  = state.get("jd_file_name")
    jd_time: str  = state.get("jd_upload_time")

    with NodeTimer("parse_jd_node", state) as timer:
        if not jd_text.strip():
            timer.extra = {"error": "empty_jd_text"}
            return {"parsed_jd": None}

        result_dict = parse_jd_text(
            jd_text=jd_text,
            file_name=jd_file,
            upload_time=jd_time,
        )

        timer.extra = {
            "jd_title":          result_dict.get("title"),
            "required_skills":   len(result_dict.get("required_skills", [])),
            "nice_to_have":      len(result_dict.get("nice_to_have_skills", [])),
            "experience_min_yrs": result_dict.get("experience_min_years"),
        }

    return {"parsed_jd": result_dict}
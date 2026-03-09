"""
schemas/jd_schema.py
--------------------
Pydantic v2 schema for a parsed Job Description.
This was entirely missing from the original design — defined here.

Used by:
  - nodes/parse_jd.py         (LLM output validated against this)
  - nodes/hybrid_retrieve.py  (metadata filtering: skills, experience)
  - nodes/score_ats.py        (required_skills drives scoring)
  - nodes/rerank_candidates.py (full JD context fed to cross-encoder)
"""

import uuid
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RequiredSkillEntry(BaseModel):
    """A skill required by the JD with minimum years."""
    skill: str = Field(..., min_length=1)
    min_years: float = Field(
        default=0.0, ge=0.0, le=50.0,
        description="Minimum years of experience required for this skill",
    )

    @field_validator("skill")
    @classmethod
    def strip_skill(cls, v: str) -> str:
        return v.strip()

    @field_validator("min_years", mode="before")
    @classmethod
    def coerce_to_float(cls, v) -> float:
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


class JobDescriptionSchema(BaseModel):
    """
    Complete structured representation of a parsed Job Description.

    jd_id is a UUID4 generated at ingestion — not by the LLM.
    required_skills drives retrieval filtering and ATS scoring.
    """
    jd_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID4, generated at ingestion",
    )
    title: str = Field(..., min_length=1, description="Job title, e.g. 'Senior Python Engineer'")
    required_skills: list[RequiredSkillEntry] = Field(
        default_factory=list,
        description="Skills that are mandatory for the role",
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description="Preferred but not mandatory skills",
    )
    experience_min_years: int = Field(
        default=0, ge=0, le=50,
        description="Minimum total years of experience required",
    )
    education_requirements: Optional[str] = Field(
        default=None,
        description="e.g. 'B.Tech or equivalent', 'Master's preferred'",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Industry/domain, e.g. 'FinTech', 'Healthcare', 'Backend'",
    )
    raw_text: str = Field(
        ..., min_length=1,
        description="Original JD text — preserved for cross-encoder reranking",
    )

    # Ingestion metadata
    file_name: Optional[str] = Field(default=None)
    upload_time: Optional[str] = Field(default=None, description="ISO 8601 datetime")

    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, v) -> str:
        return str(v).strip() if v else v

    @field_validator("nice_to_have_skills", mode="before")
    @classmethod
    def ensure_strings(cls, v) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(s).strip() for s in v if s]

    @field_validator("experience_min_years", mode="before")
    @classmethod
    def coerce_experience(cls, v) -> int:
        if v is None:
            return 0
        try:
            return int(float(str(v)))
        except (ValueError, TypeError):
            return 0

    def get_required_skill_names(self) -> list[str]:
        """Returns just the skill names (without years) for quick lookup."""
        return [s.skill for s in self.required_skills]

    def get_required_skill_map(self) -> dict[str, float]:
        """Returns {skill_name: min_years} for ATS scoring."""
        return {s.skill: s.min_years for s in self.required_skills}

    def to_qdrant_payload(self) -> dict:
        """Flat payload stored in jd_index collection."""
        return {
            "jd_id": self.jd_id,
            "title": self.title,
            "required_skills": self.get_required_skill_names(),
            "experience_min_years": self.experience_min_years,
            "domain": self.domain,
            "upload_time": self.upload_time,
            "file_name": self.file_name,
        }

    def to_embedding_text(self) -> str:
        """
        Text used to embed the JD into the vector DB.
        Same structure as resume embedding text for comparable vector space.
        """
        skills_str = ", ".join(
            [f"{s.skill} ({s.min_years}y min)" for s in self.required_skills]
        )
        nice_str = ", ".join(self.nice_to_have_skills)
        parts = [
            f"Job Title: {self.title}.",
            f"Required Skills: {skills_str}." if skills_str else "",
            f"Nice to Have: {nice_str}." if nice_str else "",
            f"Minimum Experience: {self.experience_min_years} years." if self.experience_min_years else "",
            f"Domain: {self.domain}." if self.domain else "",
            f"Education: {self.education_requirements}." if self.education_requirements else "",
        ]
        return " ".join(p for p in parts if p)
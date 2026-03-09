"""
schemas/resume_schema.py
------------------------
Pydantic v2 schema for a parsed resume.
Every field that is genuinely optional in a resume is Optional with None default.
Fields that must always be present (name, candidate_id) are required.

Used by:
  - nodes/parse_resume.py   (LLM output validated against this)
  - nodes/validate_schema.py (re-validates after parse)
  - nodes/expand_skills.py  (reads/writes skills list)
  - nodes/embed_and_store.py (reads all fields for embedding text)
  - utils/embed_template.py  (builds embedding string)
"""

import uuid
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class SkillEntry(BaseModel):
    """A single skill with experience duration."""
    skill: str = Field(..., min_length=1, description="Skill name, e.g. 'Python'")
    experience_years: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Years of experience with this skill",
    )

    @field_validator("skill")
    @classmethod
    def strip_skill_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("experience_years", mode="before")
    @classmethod
    def coerce_to_float(cls, v) -> float:
        """Accept int, str-encoded numbers ('3', '3.5'), None → 0.0."""
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


class EducationEntry(BaseModel):
    """A single education record."""
    degree: str = Field(..., min_length=1, description="e.g. 'B.Tech', 'M.S.', 'PhD'")
    university: str = Field(..., min_length=1)
    graduation_year: Optional[int] = Field(
        default=None, ge=1950, le=2030,
        description="Year of graduation",
    )
    gpa: Optional[float] = Field(
        default=None, ge=0.0, le=10.0,
        description="GPA on any scale; normalised separately if needed",
    )

    @field_validator("degree", "university", mode="before")
    @classmethod
    def strip_strings(cls, v) -> str:
        return str(v).strip() if v else v


class ExperienceEntry(BaseModel):
    """A single work experience / project entry."""
    role: str = Field(..., min_length=1, description="Job title or role")
    summary: str = Field(
        ..., min_length=1,
        description="Concise summary, stopwords removed by LLM during parsing",
    )
    project_title: Optional[str] = Field(default=None)
    client: Optional[str] = Field(default=None)
    duration_months: Optional[int] = Field(
        default=None, ge=0, le=600,
        description="Duration in months; derived from start/end dates if not explicit",
    )
    technologies: list[str] = Field(
        default_factory=list,
        description="Tech stack mentioned in this role (extracted by LLM)",
    )

    @field_validator("role", "summary", mode="before")
    @classmethod
    def strip_strings(cls, v) -> str:
        return str(v).strip() if v else v

    @field_validator("duration_months", mode="before")
    @classmethod
    def coerce_duration(cls, v) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(float(str(v)))
        except (ValueError, TypeError):
            return None


class ResumeSchema(BaseModel):
    """
    Complete structured representation of a parsed resume.

    candidate_id is always a UUID4 string, generated at ingestion time
    (not by the LLM). If the LLM omits it, it is auto-generated.
    """
    candidate_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID4, generated at ingestion — never from LLM",
    )
    name: str = Field(..., min_length=1, description="Full name of the candidate")
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    linkedin: Optional[str] = Field(default=None)
    github: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)

    skills: list[SkillEntry] = Field(
        default_factory=list,
        description="All skills after skill expansion (populated by expand_skills_node)",
    )
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    # Ingestion metadata — set by load_documents_node, not by LLM
    file_name: str = Field(..., min_length=1)
    upload_time: str = Field(..., description="ISO 8601 datetime string")

    # Derived fields — computed by model_validator after all fields set
    total_experience_months: int = Field(
        default=0,
        description="Sum of all experience entry durations in months",
    )

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, v) -> str:
        return str(v).strip() if v else v

    @field_validator("certifications", mode="before")
    @classmethod
    def ensure_cert_strings(cls, v) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(c).strip() for c in v if c]

    @model_validator(mode="after")
    def compute_total_experience(self) -> "ResumeSchema":
        """Auto-compute total_experience_months from experience entries."""
        total = sum(
            e.duration_months for e in self.experience if e.duration_months is not None
        )
        self.total_experience_months = total
        return self

    def to_embedding_dict(self) -> dict:
        """Returns a plain dict suitable for building the embedding text."""
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "skills": [{"skill": s.skill, "experience_years": s.experience_years} for s in self.skills],
            "experience": [{"role": e.role, "summary": e.summary} for e in self.experience],
            "education": [{"degree": e.degree, "university": e.university} for e in self.education],
            "certifications": self.certifications,
            "total_experience_months": self.total_experience_months,
        }

    def to_qdrant_payload(self) -> dict:
        """
        Returns the metadata payload stored alongside the vector in Qdrant.
        Keep this flat and filterable.

        Fix notes:
          - experience is now stored so the UI can show role/summary/tech stack
          - name is guarded: if LLM defaulted to "John Doe" (happens when
            python-docx fails to extract the name from a header/table), we
            store None so the UI can flag it rather than show wrong data
        """
        # Guard against LLM hallucinated placeholder names.
        # When the model cannot find a name it defaults to "John Doe" / "John Smith"
        # because the schema requires a non-empty string. Store None instead so
        # the UI knows the name was not reliably extracted.
        _PLACEHOLDER_NAMES = {"john doe", "john smith", "jane doe", "candidate"}
        safe_name = self.name if self.name.lower() not in _PLACEHOLDER_NAMES else None

        return {
            "candidate_id": self.candidate_id,
            "name": safe_name,
            "email": self.email,
            "file_name": self.file_name,
            "upload_time": self.upload_time,
            "location": self.location,
            # Skills
            "skills": [s.skill for s in self.skills],
            "skill_years": {s.skill: s.experience_years for s in self.skills},
            "total_experience_months": self.total_experience_months,
            # Experience — role, summary and tech stack for each entry.
            # These are embedded in the vector via build_resume_embedding_text
            # and now also stored as payload so the UI can display them.
            "experience": [
                {
                    "role":           e.role,
                    "client":         e.client,
                    "project_title":  e.project_title,
                    "summary":        e.summary,
                    "technologies":   e.technologies or [],
                    "duration_months": e.duration_months,
                }
                for e in self.experience
            ],
            # Education
            "education_degrees": [e.degree for e in self.education],
            "education": [
                {
                    "degree":          e.degree,
                    "university":      e.university,
                    "graduation_year": e.graduation_year,
                }
                for e in self.education
            ],
            "certifications": self.certifications,
        }
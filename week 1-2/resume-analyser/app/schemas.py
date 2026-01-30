from pydantic import BaseModel, Field
from typing import List, Optional

class JDExtract(BaseModel):
    job_title: Optional[str] = None
    seniority_level: Optional[str] = None  # intern/junior/mid/senior
    years_required: Optional[str] = None

    must_have_skills: List[str] = Field(default_factory=list)
    good_to_have_skills: List[str] = Field(default_factory=list)

    responsibilities: List[str] = Field(default_factory=list)

class JDMatchReport(BaseModel):
    jd_fit_score: int = Field(ge=0, le=100)

    matched_skills: List[str] = Field(default_factory=list)
    missing_must_have_skills: List[str] = Field(default_factory=list)
    missing_good_to_have_skills: List[str] = Field(default_factory=list)

    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)

    recommendations: List[str] = Field(default_factory=list)

class EducationItem(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None

class ExperienceItem(BaseModel):
    role: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)

class ProjectItem(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)

class SkillSet(BaseModel):
    programming_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    cloud: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    ml_dl: List[str] = Field(default_factory=list)
    other: List[str] = Field(default_factory=list)

class ResumeExtract(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None

    summary: Optional[str] = None

    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)

    skillset: SkillSet = Field(default_factory=SkillSet)

    certifications: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)

"""
utils/embed_template.py
-----------------------
Builds the text string that gets embedded as a vector for each resume.

Design rules:
  1. Consistent structure — same template every time, no random variation
  2. Information-dense — all semantically meaningful fields included
  3. JD-aligned — structure mirrors JD embedding text so vectors are comparable
  4. No stopwords in experience summaries (LLM removes them during parse)
"""

from schemas.resume_schema import ResumeSchema


def build_resume_embedding_text(resume: ResumeSchema) -> str:
    """
    Builds the embedding input string for a resume.

    Example output:
        Candidate: Jane Doe. Skills: Python (4.0y), FastAPI (2.0y), PostgreSQL (3.0y).
        Experience: Backend Engineer: Built REST APIs payment processing |
                    ML Engineer: Trained classification models customer churn.
        Education: B.Tech Computer Science IIT Delhi.
        Certifications: AWS Certified Developer.

    Args:
        resume: A validated ResumeSchema instance (after skill expansion).

    Returns:
        A single string ready to pass to the embedding model.
    """
    # ── Skills ───────────────────────────────────────────────────────────────
    if resume.skills:
        skills_str = ", ".join(
            f"{s.skill} ({s.experience_years}y)" for s in resume.skills
        )
    else:
        skills_str = "Not specified"

    # ── Experience ───────────────────────────────────────────────────────────
    if resume.experience:
        exp_parts = []
        for e in resume.experience:
            part = f"{e.role}: {e.summary}"
            if e.technologies:
                part += f" [{', '.join(e.technologies)}]"
            exp_parts.append(part)
        exp_str = " | ".join(exp_parts)
    else:
        exp_str = "Not specified"

    # ── Education ────────────────────────────────────────────────────────────
    if resume.education:
        edu_parts = [
            f"{e.degree} {e.university}"
            + (f" {e.graduation_year}" if e.graduation_year else "")
            for e in resume.education
        ]
        edu_str = ", ".join(edu_parts)
    else:
        edu_str = "Not specified"

    # ── Certifications ────────────────────────────────────────────────────────
    cert_str = ", ".join(resume.certifications) if resume.certifications else "None"

    # ── Total experience ──────────────────────────────────────────────────────
    exp_years = round(resume.total_experience_months / 12, 1) if resume.total_experience_months else 0

    return (
        f"Candidate: {resume.name}. "
        f"Total Experience: {exp_years} years. "
        f"Skills: {skills_str}. "
        f"Experience: {exp_str}. "
        f"Education: {edu_str}. "
        f"Certifications: {cert_str}."
    )
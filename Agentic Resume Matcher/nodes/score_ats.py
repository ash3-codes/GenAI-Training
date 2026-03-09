"""
nodes/score_ats.py
------------------
ATS Scoring Node — LangGraph node.

Computes a weighted ATS score for each candidate against the JD.
All sub-scores are normalised to [0.0, 1.0] before weighting.
Final score is clamped to [0.0, 1.0].

Section weights (from config.yaml):
  skills:         0.40
  experience:     0.30
  projects:       0.15
  education:      0.10
  certifications: 0.05
"""

from typing import Any
from utils.logger import NodeTimer


# ── Sub-scorers ───────────────────────────────────────────────────────────────

def score_skills(
    candidate_payload: dict,
    required_skill_map: dict[str, float],   # {skill_name: min_years}
) -> float:
    """
    Skill match score: what fraction of required skills does the candidate have?
    Partial credit if they have the skill but fewer years than required.
    """
    if not required_skill_map:
        return 1.0   # No requirements = full score

    candidate_skill_years: dict[str, float] = candidate_payload.get("skill_years", {})
    # Build case-insensitive lookup
    cand_lower = {k.lower(): v for k, v in candidate_skill_years.items()}

    total_weight = 0.0
    earned       = 0.0

    for skill, min_years in required_skill_map.items():
        weight    = max(min_years, 1.0)   # weight by importance (min_years as proxy)
        total_weight += weight

        cand_years = cand_lower.get(skill.lower(), 0.0)
        if cand_years > 0:
            if min_years <= 0:
                earned += weight           # Skill present, no year requirement
            else:
                earned += weight * min(cand_years / min_years, 1.0)  # Partial credit

    return earned / total_weight if total_weight > 0 else 0.0


def score_experience(
    candidate_payload: dict,
    experience_min_years: int,
) -> float:
    """
    Experience score: does the candidate meet the minimum years requirement?
    """
    if experience_min_years <= 0:
        return 1.0

    candidate_months = candidate_payload.get("total_experience_months", 0)
    candidate_years  = candidate_months / 12.0
    return min(candidate_years / experience_min_years, 1.0)


def score_education(candidate_payload: dict) -> float:
    """
    Education score: presence and level of formal education.
    Simplified heuristic — no NLP needed.
    """
    degrees = [d.lower() for d in candidate_payload.get("education_degrees", [])]
    if not degrees:
        return 0.0

    # Score by highest degree found
    if any("phd" in d or "doctorate" in d for d in degrees):
        return 1.0
    if any("master" in d or "m.tech" in d or "m.s" in d or "mba" in d for d in degrees):
        return 0.9
    if any("b.tech" in d or "b.e" in d or "bachelor" in d or "b.sc" in d for d in degrees):
        return 0.8
    if any("diploma" in d or "associate" in d for d in degrees):
        return 0.5
    return 0.3   # Some education listed but unrecognised


def score_certifications(candidate_payload: dict) -> float:
    """
    Certification score: any relevant certs present?
    Simple presence score — 1.0 if has certs, 0.5 if none.
    """
    certs = candidate_payload.get("certifications", [])
    if not certs:
        return 0.0
    # Score goes up slightly with more certs, capped at 1.0
    return min(0.5 + 0.1 * len(certs), 1.0)


def score_projects(candidate_payload: dict) -> float:
    """
    Project/domain score: check if candidate's skills overlap with JD domain.
    Proxy: assume more experience months = more projects. Normalise.
    """
    months = candidate_payload.get("total_experience_months", 0)
    # Normalise: 0 months = 0, 60+ months = 1.0
    return min(months / 60.0, 1.0)


# ── ATS scorer ────────────────────────────────────────────────────────────────

def compute_ats_score(
    candidate_payload: dict,
    parsed_jd: dict,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute the overall ATS score for one candidate against a JD.

    Returns a float in [0.0, 1.0].
    """
    if weights is None:
        from config.settings import ATS_WEIGHTS
        weights = ATS_WEIGHTS

    from schemas.jd_schema import JobDescriptionSchema
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)
        clean_jd = {k: v for k, v in parsed_jd.items() if k != "parsed"}
        jd_obj = JobDescriptionSchema(**clean_jd)

    skill_score = score_skills(candidate_payload, jd_obj.get_required_skill_map())
    exp_score   = score_experience(candidate_payload, jd_obj.experience_min_years)
    edu_score   = score_education(candidate_payload)
    cert_score  = score_certifications(candidate_payload)
    proj_score  = score_projects(candidate_payload)

    ats = (
        weights.get("skills",         0.40) * skill_score +
        weights.get("experience",     0.30) * exp_score   +
        weights.get("projects",       0.15) * proj_score  +
        weights.get("education",      0.10) * edu_score   +
        weights.get("certifications", 0.05) * cert_score
    )

    return round(max(0.0, min(1.0, ats)), 4)   # clamp to [0,1]


# ── LangGraph node ────────────────────────────────────────────────────────────

def score_ats_node(state: dict) -> dict:
    """
    LangGraph node: Compute ATS scores for all reranked candidates.

    Reads:  state["reranked_candidates"]
            state["parsed_jd"]
    Writes: state["ats_scores"]  — {candidate_id: float}
    """
    candidates: list[dict] = state.get("reranked_candidates", [])
    parsed_jd:  dict | None = state.get("parsed_jd")

    with NodeTimer("score_ats_node", state) as timer:
        if not candidates or not parsed_jd:
            timer.extra = {"error": "no_candidates_or_jd"}
            return {"ats_scores": {}}

        ats_scores = {}
        for c in candidates:
            cid     = c.get("candidate_id")
            payload = c.get("payload", c)
            score   = compute_ats_score(payload, parsed_jd)
            ats_scores[cid] = score

        scores_list = list(ats_scores.values())
        timer.extra = {
            "scored_count": len(ats_scores),
            "avg_score":    round(sum(scores_list) / len(scores_list), 4) if scores_list else 0,
            "max_score":    max(scores_list) if scores_list else 0,
            "min_score":    min(scores_list) if scores_list else 0,
        }

    return {"ats_scores": ats_scores}
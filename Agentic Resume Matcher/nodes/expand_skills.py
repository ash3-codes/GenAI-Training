"""
nodes/expand_skills.py
----------------------
Skill Expansion Engine — LangGraph node.

Takes the skills list from a parsed resume and expands it using the
knowledge graph in skills/skills_graph.json.

Example:
  Input:  [{"skill": "NumPy", "experience_years": 2.0}]
  Output: [{"skill": "NumPy", "experience_years": 2.0},
           {"skill": "Python", "experience_years": 2.0}]  ← inferred

Rules:
  - Parent skill gets the SAME experience_years as the child
  - If candidate already has the parent skill explicitly listed,
    keep the HIGHER of the two values (don't downgrade)
  - No duplicates — each skill appears exactly once
  - skills_graph.json is loaded once at module import (not per-call)
  - The _comment key in skills_graph.json is silently ignored
"""

import json
import time
from pathlib import Path
from typing import Any

from utils.logger import NodeTimer


# ── Load skills graph once at module level ────────────────────────────────────
_SKILLS_GRAPH_PATH = Path(__file__).resolve().parent.parent / "skills" / "skills_graph.json"

def _load_skills_graph() -> dict[str, list[str]]:
    with open(_SKILLS_GRAPH_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Filter out metadata keys (those starting with _)
    return {k: v for k, v in raw.items() if not k.startswith("_")}

_SKILLS_GRAPH: dict[str, list[str]] = _load_skills_graph()


# ── Core expansion function (pure, testable independently) ────────────────────

def expand_skills(
    skills: list[dict[str, Any]],
    graph: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Expands a list of skill dicts using the knowledge graph.

    Args:
        skills: List of {"skill": str, "experience_years": float}
        graph:  Override for the global graph (used in tests). 
                Defaults to the loaded skills_graph.json.

    Returns:
        Expanded list of skill dicts, deduplicated, with merged years.
    """
    if graph is None:
        graph = _SKILLS_GRAPH

    # Build a working dict: {skill_name_lower: {"skill": original_name, "experience_years": float}}
    # Using lowercase keys for case-insensitive deduplication
    merged: dict[str, dict[str, Any]] = {}

    for entry in skills:
        skill_name = str(entry.get("skill", "")).strip()
        years = float(entry.get("experience_years", 0.0) or 0.0)
        key = skill_name.lower()

        if key in merged:
            # Keep the higher value
            merged[key]["experience_years"] = max(merged[key]["experience_years"], years)
        else:
            merged[key] = {"skill": skill_name, "experience_years": years}

    # Expand each original skill into its parents
    original_keys = list(merged.keys())  # snapshot before we add parents
    for key in original_keys:
        skill_name = merged[key]["skill"]
        years = merged[key]["experience_years"]

        # Case-insensitive graph lookup: try exact name, then title-cased
        parents = graph.get(skill_name) or graph.get(skill_name.title()) or []

        for parent in parents:
            parent_key = parent.lower()
            if parent_key in merged:
                # Parent already exists — keep the higher value
                merged[parent_key]["experience_years"] = max(
                    merged[parent_key]["experience_years"], years
                )
            else:
                # Add inferred parent skill
                merged[parent_key] = {"skill": parent, "experience_years": years}

    # Return as a sorted list (by experience_years desc, then name asc)
    result = list(merged.values())
    result.sort(key=lambda x: (-x["experience_years"], x["skill"].lower()))
    return result


# ── LangGraph node ────────────────────────────────────────────────────────────

def expand_skills_node(state: dict) -> dict:
    """
    LangGraph node: Expand skills for all validated resumes.

    Reads:  state["parsed_resumes"]   — list of ResumeSchema dicts
    Writes: state["expanded_resumes"] — same list with expanded skills

    The node operates on dicts (not ResumeSchema objects) because LangGraph
    state is serialised as plain dicts between nodes.
    """
    parsed_resumes: list[dict] = state.get("parsed_resumes", [])

    with NodeTimer("expand_skills_node", state) as timer:
        expanded = []
        total_added = 0

        for resume in parsed_resumes:
            original_skills = resume.get("skills", [])
            original_count = len(original_skills)

            expanded_skills = expand_skills(original_skills)
            added = len(expanded_skills) - original_count
            total_added += added

            # Return a new dict (don't mutate in place)
            updated = {**resume, "skills": expanded_skills}
            expanded.append(updated)

        timer.extra = {
            "resumes_processed": len(expanded),
            "total_skills_added": total_added,
        }

    return {"expanded_resumes": expanded}
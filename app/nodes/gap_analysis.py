"""Stage 3 - Gap Analysis (pure Python, no LLM)."""
from typing import Iterable, List

from app.models.schemas import PipelineState
from app.skill_utils import normalize_skill


def gap_analysis_node(state: PipelineState) -> dict:
    jd_skills = {normalize_skill(s) for s in state["jd_requirements"]["required_skills"] if s.strip()}
    known = {normalize_skill(s) for s in state["base_resume"]["all_known_skills"] if s.strip()}
    matched = jd_skills & known
    missing_but_true = known - jd_skills  # candidate has extra skills the JD did not list
    missing_entirely = jd_skills - known  # JD wants it, candidate lacks it -> never add
    return {"gap_analysis": {
        "matched_skills": sorted(matched),
        "missing_but_true_skills": sorted(missing_but_true),
        "missing_entirely_skills": sorted(missing_entirely),
    }}


def selectable_skills(jd_requirements: dict, known_skills: Iterable[str]) -> List[dict]:
    """JD skills/tools/keywords the resume does not show - candidates for the user's checkboxes."""
    known = {normalize_skill(s) for s in known_skills}
    out, seen = [], set()
    for kind, key in (("required", "required_skills"), ("tool", "tools"),
                      ("nice_to_have", "nice_to_have_skills"), ("keyword", "top_keywords")):
        for skill in jd_requirements.get(key, []):
            norm = normalize_skill(skill)
            if norm and norm not in known and norm not in seen:
                seen.add(norm)
                out.append({"skill": skill.strip(), "type": kind})
    return out


def unknown_jd_terms(jd_requirements: dict, known_skills: Iterable[str]) -> List[str]:
    """Required skills, tools and nice-to-haves the candidate does not have (guardrail vocabulary)."""
    return [i["skill"] for i in selectable_skills(jd_requirements, known_skills) if i["type"] != "keyword"]

"""All Pydantic schemas (spec Section 6), the LangGraph state (Section 4.1) and load helpers."""
import json
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator

from app.skill_utils import expand_skill


# ---------------------------------------------------------------- 6.1 base resume
class ResumeBullet(BaseModel):
    id: str  # e.g. "exp1_b1"
    text: str
    skills_used: List[str]  # ground-truth tags used by the guardrail


class ResumeProject(BaseModel):
    name: str
    bullets: List[ResumeBullet]
    tech_stack: List[str]


class ResumeSkillGroup(BaseModel):
    category: str
    items: List[str]


class BaseResume(BaseModel):
    name: str
    title: str
    summary: str
    skills: List[ResumeSkillGroup]
    experience_bullets: List[ResumeBullet]
    projects: List[ResumeProject]
    education: str
    all_known_skills: List[str] = Field(default_factory=list)  # computed at load time
    # Additive, optional presentation fields (pass through untouched; never rewritten by the LLM):
    contact: str = ""
    certifications: List[str] = Field(default_factory=list)
    experience_heading: str = ""
    experience_tech_stack: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _compute_all_known_skills(self) -> "BaseResume":
        seen: Dict[str, str] = {}

        def add(raw: str) -> None:
            for part in expand_skill(raw):
                seen.setdefault(part.lower(), part)

        for existing in self.all_known_skills:
            add(existing)
        for group in self.skills:
            for item in group.items:
                add(item)
        for bullet in self.experience_bullets:
            for tag in bullet.skills_used:
                add(tag)
        for project in self.projects:
            for bullet in project.bullets:
                for tag in bullet.skills_used:
                    add(tag)
            for tech in project.tech_stack:
                add(tech)
        for tech in self.experience_tech_stack:
            add(tech)
        self.all_known_skills = list(seen.values())
        return self


# ---------------------------------------------------------------- 6.2 JD parser output
class JDRequirements(BaseModel):
    required_skills: List[str]
    nice_to_have_skills: List[str]
    tools: List[str]
    seniority_level: str  # e.g. "entry-level", "mid", "senior"
    top_keywords: List[str]  # 15-20 ranked keywords for ATS scoring


# ---------------------------------------------------------------- 6.3 gap analysis output
class GapAnalysis(BaseModel):
    matched_skills: List[str]  # in both JD and resume
    missing_but_true_skills: List[str]  # candidate has it, JD did not list it
    missing_entirely_skills: List[str]  # JD wants it, candidate lacks it -> must NOT be added


# ---------------------------------------------------------------- 6.4 tailored resume output
class TailoredResume(BaseModel):
    summary: str
    skills: List[ResumeSkillGroup]
    experience_bullets: List[ResumeBullet]  # same ids, reworded text only
    projects: List[ResumeProject]


# ---------------------------------------------------------------- 4.1 pipeline state
class PipelineState(TypedDict):
    jd_text: str
    jd_requirements: Optional[dict]  # output of Stage 2
    base_resume: dict  # loaded from resume.json
    gap_analysis: Optional[dict]  # output of Stage 3
    tailored_resume: Optional[dict]  # output of Stage 4 (validated)
    ats_score_before: Optional[float]
    ats_score_after: Optional[float]
    human_approved: bool
    output_paths: Optional[Dict[str, str]]  # {"docx": "...", "pdf": "..."}
    # Additive keys (optional at runtime):
    confirmed_skills: Optional[List[str]]  # extra skills the user explicitly ticked
    original_resume: Optional[dict]  # untouched base resume, used for the "before" ATS score


# ---------------------------------------------------------------- helpers
def load_base_resume(path: Path) -> BaseResume:
    """Load and validate a resume JSON file; all_known_skills is computed here."""
    with open(path, "r", encoding="utf-8") as handle:
        return BaseResume.model_validate(json.load(handle))


def merge_tailored_resume(base: dict, tailored: dict) -> dict:
    """Overlay the tailored sections on the base resume (name, contact, education, ... unchanged)."""
    merged = dict(base)
    for key in ("summary", "skills", "experience_bullets", "projects"):
        if tailored.get(key) is not None:
            merged[key] = tailored[key]
    return merged

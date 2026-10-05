"""Stage 5 - ATS Scorer (pure Python)."""
import re
from typing import List

from app.models.schemas import PipelineState, merge_tailored_resume


def resume_to_text(resume: dict) -> str:
    """Flatten every ATS-visible field of a resume dict into one string."""
    parts: List[str] = [resume.get("name", ""), resume.get("title", ""), resume.get("summary", "")]
    for group in resume.get("skills", []):
        parts.append(group.get("category", ""))
        parts.extend(group.get("items", []))
    parts.append(resume.get("experience_heading", ""))
    parts.extend(b.get("text", "") for b in resume.get("experience_bullets", []))
    parts.extend(resume.get("experience_tech_stack", []))
    for project in resume.get("projects", []):
        parts.append(project.get("name", ""))
        parts.extend(b.get("text", "") for b in project.get("bullets", []))
        parts.extend(project.get("tech_stack", []))
    parts.append(resume.get("education", ""))
    parts.extend(resume.get("certifications", []))
    return "\n".join(p for p in parts if p)


def _contains(text: str, keyword: str) -> bool:
    """Whole-term match, so 'rag' does not match 'storage' and 'java' does not match 'javascript'."""
    pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower().strip()) + r"(?:s|es)?(?![a-z0-9])"
    return re.search(pattern, text) is not None


def ats_score(resume_text: str, jd_keywords: list) -> float:
    """Percentage (0-100, 1 decimal) of JD keywords present in the resume text."""
    keywords = [k for k in jd_keywords if k and k.strip()]
    text = resume_text.lower()
    hits = sum(1 for k in keywords if _contains(text, k))
    return round(100 * hits / max(len(keywords), 1), 1)


def score_ats_node(state: PipelineState) -> dict:
    keywords = state["jd_requirements"]["top_keywords"]
    # Compare like with like: the tailored version keeps name/title/education from the base resume.
    tailored_full = merge_tailored_resume(state["base_resume"], state["tailored_resume"])
    original = state.get("original_resume") or state["base_resume"]
    before = ats_score(resume_to_text(original), keywords)
    after = ats_score(resume_to_text(tailored_full), keywords)
    return {"ats_score_before": before, "ats_score_after": after}


def keyword_report(original: dict, tailored_full: dict, keywords: list) -> list:
    """Per-keyword presence before/after, for the UI table."""
    before_text, after_text = resume_to_text(original).lower(), resume_to_text(tailored_full).lower()
    return [{"keyword": k, "in_original": _contains(before_text, k), "in_tailored": _contains(after_text, k)}
            for k in keywords if k and k.strip()]

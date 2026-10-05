"""Stage 4 - Content Rewrite (LLM) + the anti-fabrication guardrail (deterministic)."""
import json

from app import llm
from app.guardrail import check_tailored_resume, enforce_ground_truth
from app.models.schemas import PipelineState, TailoredResume
from app.nodes.gap_analysis import unknown_jd_terms

MAX_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "You are an expert resume editor. You tailor an existing resume to a job description WITHOUT ever "
    "fabricating anything. Return JSON matching the TailoredResume schema."
)


def _source_payload(base: dict) -> dict:
    return {key: base[key] for key in ("summary", "skills", "experience_bullets", "projects")}


def build_prompt(state: PipelineState, allowed: set) -> str:
    gap = state["gap_analysis"]
    jd = state["jd_requirements"]
    confirmed = state.get("confirmed_skills") or []
    confirmed_rule = (
        f"- The candidate personally confirmed these extra skills: {confirmed}. They are in the 'Additional "
        "Skills' group. Keep them in the skills section and you may mention them in the summary, but NEVER "
        "inside experience or project bullets, and never claim achievements or metrics with them.\n"
        if confirmed else "")
    return f"""
You are rewriting a resume to better match a job description.

STRICT RULES:
- You may reorder skills (and drop irrelevant skill items) and reword bullet text and the summary.
- You must NOT introduce any skill, tool, or technology not present in this exact allowed list:
  {sorted(allowed)}
- You must NOT invent numbers, metrics, or achievements not present in the source bullets.
- Keep every experience bullet and project bullet: same ids, same project names, reworded text only.
  Do not add or remove bullets. Keep bullets to one or two lines so the resume fits one page.
- Skill items must be copied exactly as written in the source skills section; put the skills that
  match the JD first (both groups and items within groups).
- These JD skills are NOT held by the candidate and must never appear anywhere:
  {gap['missing_entirely_skills']}
- Write the summary as 2-3 sentences, truthful to the source summary.
{confirmed_rule}
Matched keywords to emphasize: {gap['matched_skills']}
JD top keywords (for wording only, use ONLY if allowed): {jd['top_keywords']}
Seniority level of the role: {jd['seniority_level']}

Source resume data:
{json.dumps(_source_payload(state['base_resume']), ensure_ascii=False, indent=1)}

Return JSON matching the TailoredResume schema.
"""


def rewrite_content_node(state: PipelineState) -> dict:
    base = state["base_resume"]
    allowed = {s.lower() for s in base["all_known_skills"]}
    confirmed = state.get("confirmed_skills") or []
    forbidden = list(state["gap_analysis"]["missing_entirely_skills"]) + \
        unknown_jd_terms(state["jd_requirements"], base["all_known_skills"])
    prompt = build_prompt(state, allowed)

    violations: list = []
    for _attempt in range(MAX_ATTEMPTS):
        raw: TailoredResume = llm.invoke_structured(TailoredResume, prompt, system=SYSTEM_PROMPT)
        result = enforce_ground_truth(base, raw, ensure_skills=confirmed)  # ids / tags / names always come from the source
        violations = check_tailored_resume(base, result, forbidden, confirmed)
        if not violations:
            return {"tailored_resume": result.model_dump()}
        prompt += (f"\n\nYour previous output used unauthorized terms: {violations}. "
                   "Remove them and retry.")
    raise ValueError(f"Guardrail failed after retries: {violations}")

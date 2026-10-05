"""Stage 2 - JD Parser (LLM, structured output validated by Pydantic)."""
from app import llm
from app.models.schemas import JDRequirements, PipelineState

SYSTEM_PROMPT = (
    "You are an expert technical recruiter and ATS analyst. Extract structured requirements from a job "
    "description. Return ONLY the fields defined by the schema: required_skills, nice_to_have_skills, "
    "tools, seniority_level (entry-level, mid or senior) and top_keywords (15-20 keywords ranked by "
    "importance for ATS matching). Use short canonical names such as 'Python', 'RAG', 'LangGraph'. "
    "Only include what the job description actually states."
)


def _dedupe(items):
    seen, out = set(), []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def parse_jd_node(state: PipelineState) -> dict:
    if state.get("jd_requirements"):  # already parsed by POST /analyze - no second LLM call
        return {"jd_requirements": state["jd_requirements"]}
    jd_text = (state.get("jd_text") or "").strip()
    if not jd_text:
        raise ValueError("jd_text is empty - paste a job description to tailor against.")
    prompt = "Extract structured requirements from this job description.\n\nJD:\n" + jd_text
    result: JDRequirements = llm.invoke_structured(JDRequirements, prompt, system=SYSTEM_PROMPT)
    data = result.model_dump()
    for key in ("required_skills", "nice_to_have_skills", "tools", "top_keywords"):
        data[key] = _dedupe(data[key])
    return {"jd_requirements": data}

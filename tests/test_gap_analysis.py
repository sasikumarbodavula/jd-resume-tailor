from app.nodes.gap_analysis import gap_analysis_node


def _state(jd_skills, known):
    return {"jd_requirements": {"required_skills": jd_skills}, "base_resume": {"all_known_skills": known}}


def test_sets_are_computed_correctly():
    gap = gap_analysis_node(_state(["Python", "RAG", "Kubernetes", "SQL"],
                                   ["python", "RAG", "LangGraph", "Flask"]))["gap_analysis"]
    assert gap["matched_skills"] == ["python", "rag pipeline development"]
    assert gap["missing_but_true_skills"] == ["flask", "langgraph"]
    assert gap["missing_entirely_skills"] == ["kubernetes", "sql"]


def test_case_and_synonyms_are_normalised():
    gap = gap_analysis_node(_state(["OPENAI", "LLMs"], ["OpenAI API", "Large Language Models"]))["gap_analysis"]
    assert gap["matched_skills"] == ["large language models", "openai api"]
    assert gap["missing_entirely_skills"] == []


def test_real_resume_never_marks_owned_skill_as_missing(base_resume):
    gap = gap_analysis_node({"jd_requirements": {"required_skills": ["LangGraph", "Docker", "FastAPI"]},
                             "base_resume": base_resume})["gap_analysis"]
    assert set(gap["matched_skills"]) == {"langgraph", "fastapi"}
    assert gap["missing_entirely_skills"] == ["docker"]

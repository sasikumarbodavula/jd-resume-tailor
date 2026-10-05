from app.nodes.ats_score import ats_score, resume_to_text, score_ats_node
from tests.conftest import tailored_from


def test_all_keywords_present_gives_100():
    assert ats_score("Python, RAG and LangGraph experience", ["python", "RAG", "LangGraph"]) == 100.0


def test_no_keywords_present_gives_0():
    assert ats_score("Python developer", ["Kubernetes", "Terraform"]) == 0.0


def test_partial_and_empty_keyword_list():
    assert ats_score("python and rag", ["python", "rag", "docker", "aws"]) == 50.0
    assert ats_score("anything", []) == 0.0


def test_whole_term_matching():
    assert ats_score("I have storage experience", ["RAG"]) == 0.0
    assert ats_score("I know JavaScript", ["Java"]) == 0.0
    assert ats_score("worked with LLMs", ["LLM"]) == 100.0


def test_score_node_compares_like_with_like(base_resume):
    kws = ["LangGraph", "RAG", "Kubernetes"]
    tailored = tailored_from(base_resume)
    out = score_ats_node({"jd_requirements": {"top_keywords": kws}, "base_resume": base_resume,
                          "tailored_resume": tailored})
    assert out["ats_score_before"] == out["ats_score_after"] == round(100 * 2 / 3, 1)
    assert base_resume["name"] in resume_to_text(base_resume)

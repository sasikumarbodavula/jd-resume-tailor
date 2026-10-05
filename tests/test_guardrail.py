import pytest

from app.guardrail import (check_tailored_resume, enforce_ground_truth, extract_candidate_skill_tokens,
                           find_invented_numbers, find_unauthorized_skills)
from tests.conftest import tailored_from


def test_fabricated_skill_in_summary_is_caught(base_resume, allowed):
    t = tailored_from(base_resume, summary="GenAI engineer with Python and Kubernetes experience.")
    assert "kubernetes" in find_unauthorized_skills(t, allowed)


def test_fabricated_skill_in_bullet_is_caught(base_resume, allowed):
    t = tailored_from(base_resume)
    t["experience_bullets"][0]["text"] = "Deployed the YOLOv5 model on AWS using Docker."
    assert {"aws", "docker"} <= set(find_unauthorized_skills(t, allowed))


def test_fabricated_project_bullet_and_skill_item_are_caught(base_resume, allowed):
    t = tailored_from(base_resume)
    t["projects"][0]["bullets"][0]["text"] = "Built a RAG pipeline with Weaviate and PyTorch."
    t["skills"][0]["items"].append("Terraform")
    found = set(find_unauthorized_skills(t, allowed))
    assert {"weaviate", "pytorch", "terraform"} <= found


def test_parenthetical_skill_item_cannot_smuggle_new_skill(base_resume, allowed):
    t = tailored_from(base_resume)
    t["skills"][0]["items"].append("Prompt Engineering (Zero-shot, Fine-tuning)")
    assert "prompt engineering (zero-shot, fine-tuning)" in find_unauthorized_skills(t, allowed)


def test_jd_only_niche_skill_is_caught_via_forbidden_list(base_resume, allowed):
    t = tailored_from(base_resume, summary="Experienced with Snowpark pipelines and Python.")
    assert find_unauthorized_skills(t, allowed) == []  # unknown to the curated vocabulary...
    assert find_unauthorized_skills(t, allowed, ["snowpark"]) == ["snowpark"]  # ...caught via the JD gap


def test_soft_skills_are_not_forbidden(base_resume, allowed):
    t = tailored_from(base_resume, summary="Strong communication and teamwork with Python.")
    assert find_unauthorized_skills(t, allowed, ["communication", "teamwork"]) == []


def test_original_resume_passes_its_own_guardrail(base_resume, allowed):
    """No false positives: the untouched resume must be judged clean."""
    t = tailored_from(base_resume)
    assert find_unauthorized_skills(t, allowed) == []
    assert check_tailored_resume(base_resume, t) == []


def test_owned_skills_and_synonyms_are_allowed(base_resume, allowed):
    t = tailored_from(base_resume, summary="Built LLM apps with OpenAI, Anthropic, React and Chroma using RAG.")
    assert find_unauthorized_skills(t, allowed) == []


def test_whole_term_matching_no_substring_false_positives():
    vocab = ["java", "javascript", "r programming", "sql"]
    assert extract_candidate_skill_tokens("Wrote JavaScript and some SQL.", vocab) == ["javascript", "sql"]
    assert extract_candidate_skill_tokens("A javanese caravan", vocab) == []


def test_invented_numbers_are_caught(base_resume):
    t = tailored_from(base_resume)
    assert find_invented_numbers(base_resume, t) == []
    t["experience_bullets"][0]["text"] = "Improved detection accuracy by 35% on 4,000+ images."
    assert find_invented_numbers(base_resume, t) == ["35"]
    assert "invented number '35'" in check_tailored_resume(base_resume, t)


def test_enforce_ground_truth_restores_ids_tags_and_repairs_structure(base_resume):
    t = tailored_from(base_resume)
    t["experience_bullets"][0]["skills_used"] = ["Kubernetes"]  # LLM tampered with a tag
    t["experience_bullets"][0]["text"] = "Reworded bullet about YOLOv5."
    del t["experience_bullets"][1]  # LLM dropped a bullet
    t["projects"][0]["tech_stack"] = ["Docker"]
    fixed = enforce_ground_truth(base_resume, t).model_dump()
    assert [b["id"] for b in fixed["experience_bullets"]] == ["exp1_b1", "exp1_b2", "exp1_b3"]
    assert fixed["experience_bullets"][0]["text"] == "Reworded bullet about YOLOv5."
    assert fixed["experience_bullets"][0]["skills_used"] == base_resume["experience_bullets"][0]["skills_used"]
    assert fixed["experience_bullets"][1]["text"] == base_resume["experience_bullets"][1]["text"]
    assert fixed["projects"][0]["tech_stack"] == base_resume["projects"][0]["tech_stack"]

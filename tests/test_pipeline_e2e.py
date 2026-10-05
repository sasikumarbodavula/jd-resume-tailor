"""End-to-end smoke tests with a mocked LLM (no API key, no network)."""
import pytest
from docx import Document

from app import llm, main
from app.graph import app as pipeline
from app.models.schemas import JDRequirements, TailoredResume
from app.nodes.export import build_diff, export_node
from app.nodes.rewrite_content import rewrite_content_node
from tests.conftest import tailored_from

JD = ("We are hiring an entry-level GenAI Engineer. You will build RAG pipelines and LangGraph agents in "
      "Python with FastAPI, deploy on Kubernetes and Docker, and work with Snowflake.")

JD_REQUIREMENTS = JDRequirements(
    required_skills=["Python", "RAG", "LangGraph", "FastAPI", "Kubernetes", "Docker"],
    nice_to_have_skills=["Snowflake"], tools=["Docker"], seniority_level="entry-level",
    top_keywords=["Python", "RAG", "LangGraph", "FastAPI", "Kubernetes", "Docker", "agents", "GenAI"],
)


def make_fake_llm(base, tailored_versions):
    """Returns JD requirements for JDRequirements, then successive TailoredResume payloads."""
    queue = list(tailored_versions)
    calls = {"rewrite": 0}

    def fake(schema, prompt, system=None):
        if schema is JDRequirements:
            return JD_REQUIREMENTS
        calls["rewrite"] += 1
        return TailoredResume.model_validate(queue.pop(0) if len(queue) > 1 else queue[0])

    fake.calls = calls
    return fake


def initial_state(base):
    return {"jd_text": JD, "jd_requirements": None, "base_resume": base, "gap_analysis": None,
            "tailored_resume": None, "ats_score_before": None, "ats_score_after": None,
            "human_approved": False, "output_paths": None}


def good_tailoring(base):
    t = tailored_from(base, summary="GenAI engineer building RAG pipelines and LangGraph agents in Python and FastAPI.")
    t["skills"].reverse()
    t["experience_bullets"][0]["text"] = "Contributed to a YOLOv5 damage detection system (5 classes, 4,000+ images)."
    return t


def test_full_graph_without_approval_does_not_export(base_resume, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(llm, "invoke_structured", make_fake_llm(base_resume, [good_tailoring(base_resume)]))
    state = pipeline.invoke(initial_state(base_resume))

    assert state["gap_analysis"]["missing_entirely_skills"] == ["docker", "kubernetes"]
    # Honest scoring: Kubernetes/Docker are absent from the resume, so the score cannot be inflated to 100.
    assert state["ats_score_before"] == state["ats_score_after"] == 75.0
    assert state["human_approved"] is False and state["output_paths"] is None
    assert list(tmp_path.iterdir()) == []  # nothing rendered before approval
    diff = build_diff(base_resume, state["tailored_resume"])
    assert diff["summary"]["changed"] and diff["experience_bullets"][0]["changed"]


def test_guardrail_retries_then_succeeds(base_resume, monkeypatch):
    bad = tailored_from(base_resume, summary="GenAI engineer experienced with Kubernetes and Python.")
    fake = make_fake_llm(base_resume, [bad, good_tailoring(base_resume)])
    monkeypatch.setattr(llm, "invoke_structured", fake)
    state = initial_state(base_resume)
    state["gap_analysis"] = {"matched_skills": ["python"], "missing_entirely_skills": ["kubernetes"]}
    state["jd_requirements"] = JD_REQUIREMENTS.model_dump()
    out = rewrite_content_node(state)
    assert fake.calls["rewrite"] == 2
    assert "Kubernetes" not in out["tailored_resume"]["summary"]


def test_guardrail_raises_after_two_failed_attempts(base_resume, monkeypatch):
    bad = tailored_from(base_resume, summary="Expert in Kubernetes, Docker and AWS.")
    monkeypatch.setattr(llm, "invoke_structured", make_fake_llm(base_resume, [bad]))
    state = initial_state(base_resume)
    state["gap_analysis"] = {"matched_skills": [], "missing_entirely_skills": ["kubernetes"]}
    state["jd_requirements"] = JD_REQUIREMENTS.model_dump()
    with pytest.raises(ValueError, match="Guardrail failed"):
        rewrite_content_node(state)


def test_export_after_approval_writes_docx(base_resume, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr("app.nodes.export.convert_to_pdf", lambda p: None)
    tailored = tailored_from(base_resume)
    out = export_node({"base_resume": base_resume, "tailored_resume": tailored, "human_approved": True})
    text = "\n".join(p.text for p in Document(out["output_paths"]["docx"]).paragraphs)
    assert base_resume["name"] in text and "LangGraph" in text and base_resume["contact"].split(" | ")[0] in text


def test_api_two_step_flow(base_resume, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr("app.config.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.nodes.export.convert_to_pdf", lambda p: None)
    monkeypatch.setattr(llm, "invoke_structured", make_fake_llm(base_resume, [good_tailoring(base_resume)]))

    draft = main.tailor(main.TailorRequest(jd_text=JD, resume_id="resume"))
    assert draft["ats_score_after"] >= draft["ats_score_before"] and "diff" in draft

    rejected = main.approve(main.ApproveRequest(session_id=draft["session_id"], approved=False))
    assert rejected["status"] == "rejected"

    done = main.approve(main.ApproveRequest(session_id=draft["session_id"], approved=True))
    assert done["status"] == "exported" and "docx" in done["output_paths"] and done["warning"]
    assert str(main.download(draft["session_id"], "docx").path).endswith(".docx")
    assert main.health() == {"status": "ok"}
    assert main.get_resume("resume")["name"] == base_resume["name"]

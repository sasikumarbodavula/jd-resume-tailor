"""Stage 6 - Human review diff + export (only after explicit approval)."""
import re
import uuid
from typing import Dict, List

from app import config
from app.models.schemas import PipelineState, merge_tailored_resume
from app.resume_render import convert_to_pdf, render_docx


def build_diff(base: dict, tailored: dict) -> dict:
    """Original vs tailored view shown to the human before anything is exported."""
    def bullet_rows(base_bullets: List[dict], new_bullets: List[dict]) -> List[dict]:
        new_by_id = {b["id"]: b["text"] for b in new_bullets}
        return [{"id": b["id"], "original": b["text"], "tailored": new_by_id.get(b["id"], b["text"]),
                 "changed": new_by_id.get(b["id"], b["text"]) != b["text"]} for b in base_bullets]

    def skill_lines(groups: List[dict]) -> List[str]:
        return [f"{g['category']}: {', '.join(g['items'])}" for g in groups]

    tailored_projects = {p["name"]: p for p in tailored["projects"]}
    return {
        "summary": {"original": base["summary"], "tailored": tailored["summary"],
                    "changed": base["summary"] != tailored["summary"]},
        "skills": {"original": skill_lines(base["skills"]), "tailored": skill_lines(tailored["skills"]),
                   "changed": base["skills"] != tailored["skills"]},
        "experience_bullets": bullet_rows(base["experience_bullets"], tailored["experience_bullets"]),
        "projects": [{"name": p["name"],
                      "bullets": bullet_rows(p["bullets"], tailored_projects.get(p["name"], p)["bullets"])}
                     for p in base["projects"]],
    }


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "Resume"


def export_node(state: PipelineState) -> dict:
    """Render .docx/.pdf. A no-op unless the human has approved (human_approved=True)."""
    if not state.get("human_approved"):
        return {"output_paths": None}

    merged = merge_tailored_resume(state["base_resume"], state["tailored_resume"])
    stem = f"{_slug(merged.get('name', 'Resume')).title()}_Tailored_{uuid.uuid4().hex[:8]}"
    out_dir = config.output_dir()
    docx_path = render_docx(merged, out_dir / f"{stem}.docx")
    paths: Dict[str, str] = {"docx": str(docx_path)}
    pdf_path = convert_to_pdf(docx_path)
    if pdf_path is not None:
        paths["pdf"] = str(pdf_path)
    return {"output_paths": paths}

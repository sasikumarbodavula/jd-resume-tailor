"""FastAPI backend.

Resume management
  GET  /resumes                 list stored resume ids
  GET  /resume/{resume_id}      stored base resume
  POST /resume/parse            read an uploaded PDF/DOCX/TXT (base64 JSON) into the resume schema
  POST /resume/save             validate + store a resume as data/{resume_id}.json
Tailoring
  POST /analyze                 Stages 2-3: parse JD, gap analysis, list skills the user can tick
  POST /tailor                  Stages 1-5 (reuses /analyze result when session_id is given);
                                optional confirmed_skills = skills the user ticked
  POST /tailor/update           save user edits, return the refreshed ATS score immediately
  POST /tailor/approve          Stage 6: render .docx/.pdf only if approved=true
  GET  /download/{session_id}/{docx|pdf}
  GET  /health
"""
import base64
import binascii
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from app import config
from app.graph import app as pipeline
from app.guardrail import apply_confirmed_skills, enforce_ground_truth, find_invented_numbers, \
    find_unauthorized_skills
from app.llm import LLMError
from app.models.schemas import BaseResume, PipelineState, TailoredResume, load_base_resume, \
    merge_tailored_resume
from app.nodes.ats_score import ats_score, keyword_report, resume_to_text
from app.nodes.export import build_diff, export_node
from app.nodes.gap_analysis import gap_analysis_node, selectable_skills
from app.nodes.parse_jd import parse_jd_node
from app.resume_parser import parse_resume

logger = logging.getLogger(__name__)
api = FastAPI(title="JD-to-Resume Tailoring Agent", version="2.0.0")
app = api  # `uvicorn app.main:app`

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_SESSION_RE = re.compile(r"^[a-f0-9]{32}$")
_SESSIONS: Dict[str, dict] = {}


# ----------------------------------------------------------------- request models
class AnalyzeRequest(BaseModel):
    jd_text: str = Field(min_length=20, description="Pasted job description text")
    resume_id: str = "resume"


class TailorRequest(BaseModel):
    jd_text: str = Field(min_length=20)
    resume_id: str = "resume"
    session_id: Optional[str] = None  # from /analyze: reuses the parsed JD (saves an LLM call)
    confirmed_skills: List[str] = Field(default_factory=list)  # skills the user says they have


class UpdateRequest(BaseModel):
    session_id: str
    tailored_resume: TailoredResume


class ApproveRequest(BaseModel):
    session_id: str
    approved: bool


class ParseRequest(BaseModel):
    filename: str
    content_base64: str


class SaveResumeRequest(BaseModel):
    resume_id: str
    resume: dict
    overwrite: bool = False


# ----------------------------------------------------------------- helpers
def _check_id(value: str, label: str = "resume_id") -> str:
    if not _ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"{label} may only contain letters, digits, '_' and '-'.")
    return value


def _resume_path(resume_id: str) -> Path:
    path = config.DATA_DIR / f"{_check_id(resume_id)}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Resume '{resume_id}' not found in data/.")
    return path


def _save_session(session_id: str, session: dict) -> None:
    _SESSIONS[session_id] = session
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (config.SESSIONS_DIR / f"{session_id}.json").write_text(json.dumps(session, indent=2), encoding="utf-8")


def _load_session(session_id: str) -> dict:
    if not _SESSION_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    if session_id in _SESSIONS:
        return _SESSIONS[session_id]
    path = config.SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found. Run POST /analyze or /tailor first.")
    session = json.loads(path.read_text(encoding="utf-8"))
    _SESSIONS[session_id] = session
    return session


def _run(fn: Callable[[], dict]) -> dict:
    """Map pipeline failures to clean HTTP errors."""
    try:
        return fn()
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:  # guardrail failed after retries / empty input / bad file
        raise HTTPException(status_code=422, detail=str(exc))


def _initial_state(jd_text: str, base: dict, original: dict, jd_requirements: Optional[dict],
                   confirmed: List[str]) -> PipelineState:
    return {
        "jd_text": jd_text, "jd_requirements": jd_requirements, "base_resume": base,
        "gap_analysis": None, "tailored_resume": None, "ats_score_before": None,
        "ats_score_after": None, "human_approved": False, "output_paths": None,
        "confirmed_skills": confirmed, "original_resume": original,
    }


def _scores(session: dict) -> dict:
    keywords = session["jd_requirements"]["top_keywords"]
    merged = merge_tailored_resume(session["base_resume"], session["tailored_resume"])
    original = session.get("original_resume") or session["base_resume"]
    return {
        "ats_score_before": ats_score(resume_to_text(original), keywords),
        "ats_score_after": ats_score(resume_to_text(merged), keywords),
        "keywords": keyword_report(original, merged, keywords),
    }


def _draft_payload(session_id: str, session: dict) -> dict:
    original = session.get("original_resume") or session["base_resume"]
    return {
        "session_id": session_id,
        "draft_version": session.get("draft_version", 1),
        "jd_requirements": session["jd_requirements"],
        "gap_analysis": session["gap_analysis"],
        "confirmed_skills": session.get("confirmed_skills") or [],
        "tailored_resume": session["tailored_resume"],
        "diff": build_diff(original, session["tailored_resume"]),
        **_scores(session),
        "next_step": "Edit if you like (POST /tailor/update), then POST /tailor/approve.",
    }


def _download_links(session_id: str, output_paths: Optional[Dict[str, str]]) -> Dict[str, str]:
    return {fmt: f"/download/{session_id}/{fmt}" for fmt in (output_paths or {})}


# ----------------------------------------------------------------- basic + resume endpoints
@api.get("/health")
def health() -> dict:
    return {"status": "ok"}


@api.get("/resumes")
def list_resumes() -> dict:
    return {"resumes": sorted(p.stem for p in config.DATA_DIR.glob("*.json"))}


@api.get("/resume/{resume_id}")
def get_resume(resume_id: str) -> dict:
    return load_base_resume(_resume_path(resume_id)).model_dump()


@api.post("/resume/parse")
def parse_uploaded_resume(request: ParseRequest) -> dict:
    try:
        data = base64.b64decode(request.content_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="content_base64 is not valid base64.")

    def work() -> dict:
        resume, warnings = parse_resume(request.filename, data)
        return {"resume": resume, "warnings": warnings}

    return _run(work)


@api.post("/resume/save")
def save_resume(request: SaveResumeRequest) -> dict:
    resume_id = _check_id(request.resume_id)
    path = config.DATA_DIR / f"{resume_id}.json"
    if path.exists() and not request.overwrite:
        raise HTTPException(status_code=409, detail=f"'{resume_id}' already exists. Tick overwrite or pick another name.")
    try:
        resume = BaseResume.model_validate(request.resume)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Resume JSON is invalid: {exc.errors()[:3]}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resume.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"resume_id": resume_id, "saved": True, "known_skills": len(resume.all_known_skills)}


# ----------------------------------------------------------------- tailoring endpoints
@api.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Stages 2-3 only: lets the user see the gap and tick skills BEFORE the rewrite."""
    base = load_base_resume(_resume_path(request.resume_id)).model_dump()

    def work() -> dict:
        state = _initial_state(request.jd_text, base, base, None, [])
        state["jd_requirements"] = parse_jd_node(state)["jd_requirements"]
        state["gap_analysis"] = gap_analysis_node(state)["gap_analysis"]
        session_id = uuid.uuid4().hex
        keywords = state["jd_requirements"]["top_keywords"]
        session = {**state, "resume_id": request.resume_id, "draft_version": 0,
                   "ats_score_before": ats_score(resume_to_text(base), keywords)}
        _save_session(session_id, session)
        return {
            "session_id": session_id,
            "jd_text": request.jd_text,
            "resume_id": request.resume_id,
            "jd_requirements": state["jd_requirements"],
            "gap_analysis": state["gap_analysis"],
            "ats_score_before": session["ats_score_before"],
            "selectable_skills": selectable_skills(state["jd_requirements"], base["all_known_skills"]),
        }

    return _run(work)


@api.post("/tailor")
def tailor(request: TailorRequest) -> dict:
    original = load_base_resume(_resume_path(request.resume_id)).model_dump()

    previous = None
    if request.session_id:
        previous = _load_session(request.session_id)
        if previous.get("jd_text") != request.jd_text or previous.get("resume_id") != request.resume_id:
            previous = None  # different JD / resume: start fresh
    session_id = request.session_id if previous else uuid.uuid4().hex

    confirmed = [s.strip() for s in request.confirmed_skills if s and s.strip()]
    effective = apply_confirmed_skills(original, confirmed)
    new_confirmed = effective["confirmed_skills"]

    def work() -> dict:
        state = _initial_state(request.jd_text, effective, original,
                               previous.get("jd_requirements") if previous else None, new_confirmed)
        result = pipeline.invoke(state)
        session = {**result, "resume_id": request.resume_id,
                   "draft_version": (previous.get("draft_version", 0) if previous else 0) + 1}
        _save_session(session_id, session)
        return _draft_payload(session_id, session)

    return _run(work)


@api.post("/tailor/update")
def update(request: UpdateRequest) -> dict:
    """Save the user's edits and return the refreshed ATS score (pure Python, instant)."""
    session = _load_session(request.session_id)
    if not session.get("tailored_resume"):
        raise HTTPException(status_code=400, detail="Nothing to edit yet. Run POST /tailor first.")

    edited = enforce_ground_truth(session["base_resume"], request.tailored_resume).model_dump()
    allowed = {s.lower() for s in session["base_resume"]["all_known_skills"]}
    warnings = [f"'{term}' is not in your resume or your confirmed skills - only keep it if it is true."
                for term in find_unauthorized_skills(edited, allowed)]
    warnings += [f"Number '{n}' is not in your original resume - only keep it if it is true."
                 for n in find_invented_numbers(session["base_resume"], edited)]

    session["tailored_resume"] = edited
    session["output_paths"] = None  # any earlier export is now stale
    _save_session(request.session_id, session)
    payload = _draft_payload(request.session_id, session)
    payload["warnings"] = warnings
    return payload


@api.post("/tailor/approve")
def approve(request: ApproveRequest) -> dict:
    session = _load_session(request.session_id)
    if not session.get("tailored_resume"):
        raise HTTPException(status_code=400, detail="Nothing to approve yet. Run POST /tailor first.")
    if not request.approved:
        session["human_approved"] = False
        _save_session(request.session_id, session)
        return {"session_id": request.session_id, "status": "rejected", "output_paths": None}

    if not session.get("output_paths"):
        session["human_approved"] = True
        session.update(export_node(session))  # Stage 6: render only now
        _save_session(request.session_id, session)
    paths = session["output_paths"]
    return {
        "session_id": request.session_id,
        "status": "exported",
        "output_paths": paths,
        "download_urls": _download_links(request.session_id, paths),
        "warning": None if "pdf" in paths else
        "PDF was not generated: install Microsoft Word (docx2pdf) or LibreOffice (soffice). DOCX is ready.",
    }


@api.get("/download/{session_id}/{file_format}")
def download(session_id: str, file_format: str) -> FileResponse:
    session = _load_session(session_id)
    paths = session.get("output_paths") or {}
    if file_format not in ("docx", "pdf") or file_format not in paths:
        raise HTTPException(status_code=404, detail="File not available. Approve the draft first.")
    path = Path(paths[file_format])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File no longer exists on disk.")
    return FileResponse(path, filename=path.name)

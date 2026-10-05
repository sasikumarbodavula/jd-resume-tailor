"""Upload support: read a PDF/DOCX/TXT resume and convert it into the resume.json schema.

One LLM call structures the text; deterministic checks then make sure no skill or tag was invented:
every skill, tag and tech-stack entry must literally appear in the uploaded text, otherwise it is dropped.
"""
import io
import re
from pathlib import Path
from typing import List, Tuple

from docx import Document

from app import llm
from app.models.schemas import BaseResume
from app.skill_utils import expand_skill

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MIN_TEXT_CHARS = 100

SYSTEM_PROMPT = (
    "You convert a resume's raw text into a structured JSON resume. Copy wording VERBATIM from the "
    "resume; never add, infer or embellish anything."
)

PARSE_PROMPT = """Convert this resume text into the BaseResume schema.

RULES:
- name, title (the headline under the name), contact (email | phone | location | links on one line),
  summary (the summary paragraph only), education (one string), certifications (list of strings).
- skills: keep the resume's own skill categories and items exactly as written.
- experience_bullets: every bullet from work experience/internships, text copied verbatim.
  experience_heading: one line with company, role, dates (and project name if given).
  experience_tech_stack: the tech stack line of that experience, as a list.
- projects: each project with its bullets (verbatim) and tech_stack list. Put "(Personal Project)" etc.
  in the project name only if the resume has it.
- bullet ids: "exp1_b1", "exp1_b2", ... for experience and "proj1_b1", "proj2_b1", ... for projects.
- skills_used for each bullet: ONLY skills/tools that are literally named in that bullet's text or in the
  resume's skills section and clearly used by that bullet. Never invent a skill.
- Leave all_known_skills empty.

RESUME TEXT:
"""


def extract_text(filename: str, data: bytes) -> str:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File is larger than 5 MB.")
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif ext == ".docx":
        doc = Document(io.BytesIO(data))
        lines = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(cell.text for cell in row.cells))
        text = "\n".join(lines)
    elif ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Upload a .pdf, .docx or .txt resume.")
    text = text.strip()
    if len(text) < MIN_TEXT_CHARS:
        raise ValueError("Could not read enough text from the file (scanned/image-only PDFs are not supported).")
    return text


def _norm(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "-", text)  # re-join words hyphenated across PDF line breaks
    return re.sub(r"\s+", " ", text).lower()


def _supported(term: str, source: str) -> bool:
    term_n = _norm(term)
    if term_n and term_n in source:
        return True
    parts = expand_skill(term)
    return bool(parts) and all(_norm(p) in source for p in parts)


def verify_parsed_resume(parsed: dict, source_text: str) -> Tuple[dict, List[str]]:
    """Drop every skill / tag / tech-stack entry that is not literally in the uploaded text."""
    source = _norm(source_text)
    warnings: List[str] = []
    dropped = set()

    def keep(terms: List[str]) -> List[str]:
        kept = []
        for term in terms:
            if _supported(term, source):
                kept.append(term)
            else:
                dropped.add(term)
        return kept

    groups = []
    for group in parsed.get("skills", []):
        items = keep(group.get("items", []))
        if items:
            groups.append({"category": group["category"], "items": items})
    parsed["skills"] = groups

    parsed["experience_tech_stack"] = keep(parsed.get("experience_tech_stack", []))

    def clean_bullets(bullets: List[dict], prefix: str) -> List[dict]:
        out = []
        for index, bullet in enumerate(bullets, start=1):
            text = bullet["text"].strip()
            if _norm(text)[:40] not in source:
                warnings.append(f"Bullet text may differ from your file - please check: \"{text[:60]}...\"")
            out.append({"id": f"{prefix}_b{index}", "text": text, "skills_used": keep(bullet.get("skills_used", []))})
        return out

    parsed["experience_bullets"] = clean_bullets(parsed.get("experience_bullets", []), "exp1")
    for number, project in enumerate(parsed.get("projects", []), start=1):
        project["bullets"] = clean_bullets(project.get("bullets", []), f"proj{number}")
        project["tech_stack"] = keep(project.get("tech_stack", []))

    if dropped:
        warnings.insert(0, "Removed because they do not appear in your file: " + ", ".join(sorted(dropped)))
    parsed["all_known_skills"] = []  # recomputed on load
    return parsed, warnings


def parse_resume(filename: str, data: bytes) -> Tuple[dict, List[str]]:
    text = extract_text(filename, data)
    result: BaseResume = llm.invoke_structured(BaseResume, PARSE_PROMPT + text, system=SYSTEM_PROMPT)
    cleaned, warnings = verify_parsed_resume(result.model_dump(), text)
    final = BaseResume.model_validate(cleaned).model_dump()
    return final, warnings

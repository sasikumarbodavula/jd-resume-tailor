import json
from pathlib import Path

import pytest

from app.models.schemas import load_base_resume

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_resume() -> dict:
    return load_base_resume(ROOT / "data" / "resume.json").model_dump()


@pytest.fixture
def allowed(base_resume) -> set:
    return {s.lower() for s in base_resume["all_known_skills"]}


def tailored_from(base: dict, **overrides) -> dict:
    """An identity 'tailoring' of the base resume, with optional field overrides."""
    data = {k: json.loads(json.dumps(base[k])) for k in ("summary", "skills", "experience_bullets", "projects")}
    data.update(overrides)
    return data

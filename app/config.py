"""Central configuration (paths + env-driven settings)."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

load_dotenv(PROJECT_ROOT / ".env")

# gemini-1.5-flash (named in the spec) has been retired by Google; override via .env if needed.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def output_dir() -> Path:
    """Where generated .docx/.pdf files are written (read at call time so tests can override)."""
    return Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))

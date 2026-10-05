"""Skill-string helpers shared by schemas, gap analysis, guardrail and scoring."""
import re
from typing import List

# canonical name -> equivalent surface forms (all lowercase). Only true synonyms belong here.
_ALIAS_GROUPS = {
    "openai api": ["openai", "openai api", "openai apis", "open ai"],
    "anthropic claude api": ["anthropic", "anthropic api", "anthropic apis", "claude", "claude api",
                             "anthropic claude", "anthropic claude api"],
    "gemini api": ["gemini", "gemini api", "gemini apis", "google gemini"],
    "react.js": ["react", "reactjs", "react js", "react.js"],
    "rest apis": ["rest", "rest api", "rest apis", "restful api", "restful apis"],
    "large language models": ["llm", "llms", "large language model", "large language models"],
    "chroma": ["chroma", "chromadb", "chroma db"],
    "vector databases": ["vector database", "vector databases", "vector db", "vector dbs"],
    "vector embeddings": ["vector embeddings", "vector embedding", "embeddings", "embedding"],
    "rag pipeline development": ["rag", "rag pipeline", "rag pipelines", "rag pipeline development",
                                 "retrieval augmented generation", "retrieval-augmented generation"],
    "agentic ai": ["agentic ai", "agentic workflows", "agentic ai workflows"],
    "ai agents": ["ai agents", "ai agent"],
    "function calling": ["function calling", "tool calling"],
    "asyncio": ["asyncio", "python asyncio"],
    "fastapi": ["fastapi", "fast api"],
    "langchain": ["langchain", "lang chain"],
    "langgraph": ["langgraph", "lang graph"],
    "gpt-4o-mini": ["gpt-4o-mini", "gpt-4o mini"],
    "transformers": ["transformers", "transformer models"],
    "generative ai": ["generative ai", "genai", "gen ai"],
    "vs code": ["vs code", "vscode", "visual studio code"],
    "virtual environments": ["virtual environments", "virtual environment", "venv"],
    "streaming responses": ["streaming responses", "streaming response"],
    "structured outputs": ["structured outputs", "structured output"],
    "cost optimization": ["cost optimization", "cost optimisation"],
}
ALIASES = {variant: canon for canon, variants in _ALIAS_GROUPS.items() for variant in variants + [canon]}

_NO_SLASH_SPLIT = {"ci/cd", "tcp/ip", "i/o"}


def clean_skill(text: str) -> str:
    """Lowercase, collapse whitespace, trim surrounding punctuation."""
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .,;:")


def normalize_skill(text: str) -> str:
    """Canonical comparison form of a skill (lowercase + synonym folding)."""
    cleaned = clean_skill(text)
    return ALIASES.get(cleaned, cleaned)


def expand_skill(item: str) -> List[str]:
    """Split a written skill into atomic parts.

    "Prompt Engineering (Zero-shot, Few-shot)" -> ["Prompt Engineering", "Zero-shot", "Few-shot"]
    "Chroma/Qdrant"                            -> ["Chroma", "Qdrant"]
    Nothing is invented: every part is text the candidate actually wrote.
    """
    item = item.strip()
    parts: List[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value.lower() not in [p.lower() for p in parts]:
            parts.append(value)

    match = re.match(r"^(.*?)\s*\((.*)\)\s*$", item)
    if match and match.group(1).strip():
        add(match.group(1))
        for inner in match.group(2).split(","):
            add(inner)
    else:
        add(item)

    expanded: List[str] = []
    for part in parts:
        if "/" in part and part.lower() not in _NO_SLASH_SPLIT:
            expanded.extend(piece for piece in part.split("/") if piece.strip())
        else:
            expanded.append(part)
    result: List[str] = []
    for part in expanded:
        if part.strip() and clean_skill(part) not in [clean_skill(r) for r in result]:
            result.append(part.strip())
    return result


def split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split on `sep` but not inside parentheses: 'A (x, y), B' -> ['A (x, y)', 'B']."""
    parts, depth, current = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def parse_skill_lines(text: str) -> List[dict]:
    """'Category: a, b' lines -> [{'category': 'Category', 'items': ['a', 'b']}] (editable skills box)."""
    groups = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        category, rest = line.split(":", 1) if ":" in line else ("Skills", line)
        items = split_top_level(rest)
        if items:
            groups.append({"category": category.strip() or "Skills", "items": items})
    return groups


def format_skill_lines(groups: List[dict]) -> str:
    return "\n".join(f"{g['category']}: {', '.join(g['items'])}" for g in groups)

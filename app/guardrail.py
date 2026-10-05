"""Anti-fabrication guardrail (spec Section 7.3).

The LLM is *asked* not to invent skills, but this module *enforces* it deterministically:

1. find_unauthorized_skills  - any known tech term in the output that is not in the candidate's
                               ground-truth skill list (plus JD-only skills) is a violation.
2. find_invented_numbers     - any number in the output that is not in the source resume.
3. enforce_ground_truth      - ids, skills_used tags, project names and tech stacks are always
                               restored from the base resume; only wording may come from the LLM.
"""
import copy
import re
from typing import Iterable, List, Set, Union

from app.models.schemas import TailoredResume
from app.skill_utils import ALIASES, expand_skill, normalize_skill

# Curated vocabulary of common AI / software terms. Any of these appearing in tailored output but
# absent from the candidate's known skills is treated as a fabrication.
COMMON_TECH_TERMS: List[str] = [
    # languages
    "java", "javascript", "typescript", "c++", "c#", "golang", "rust", "kotlin", "scala", "ruby", "php",
    "matlab", "perl", "bash", "shell scripting", "sql", "nosql", "html", "css", "r programming",
    # web / backend frameworks
    "django", "spring boot", "node.js", "express.js", "next.js", "vue", "angular", "svelte", "tailwind",
    "graphql", "grpc", "websockets", "microservices", "streamlit", "gradio", "celery", "rabbitmq", "kafka",
    # cloud / devops
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "github actions", "gitlab", "ci/cd", "helm", "prometheus", "grafana", "linux", "nginx", "lambda",
    "sagemaker", "vertex ai", "bedrock", "azure openai", "databricks", "snowflake", "bigquery", "airflow",
    "mlflow", "kubeflow", "dvc", "serverless", "ec2", "s3",
    # data stores
    "mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite", "elasticsearch", "opensearch",
    "cassandra", "dynamodb", "neo4j", "pgvector", "faiss", "weaviate", "milvus", "pinecone", "qdrant", "chroma",
    "hadoop", "spark", "pyspark", "hive", "power bi", "tableau", "microsoft excel",
    # ML / DL
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost", "lightgbm", "opencv", "onnx",
    "tensorrt", "cuda", "jax", "hugging face", "huggingface", "bert", "gpt-4", "gpt-3.5", "gpt-4o", "llama",
    "mistral", "mixtral", "falcon", "t5", "whisper", "stable diffusion", "diffusers", "yolov8", "yolov7",
    "detectron2", "mediapipe", "nltk", "spacy", "gensim", "matplotlib", "seaborn", "plotly", "jupyter",
    "machine learning", "computer vision", "natural language processing", "nlp", "reinforcement learning",
    "time series", "recommendation systems", "mlops", "llmops", "data engineering", "etl", "feature engineering",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft", "rlhf", "dpo", "distillation", "quantization",
    "knowledge graphs", "graph rag", "hybrid search", "reranking", "re-ranking", "bm25", "multimodal",
    "speech recognition", "text-to-speech", "ocr", "tesseract", "opencv",
    # LLM tooling / vendors
    "openai", "anthropic", "claude", "gemini", "cohere", "groq", "ollama", "vllm", "llamaindex", "llama index",
    "crewai", "autogen", "haystack", "dspy", "langsmith", "langserve", "langfuse", "semantic kernel",
    "mcp", "model context protocol", "guardrails", "promptfoo", "weights & biases", "wandb",
    # testing / process / tools
    "pytest", "unittest", "selenium", "postman", "swagger", "openapi", "jira", "confluence", "agile", "scrum",
    "kanban", "tdd", "pycharm", "jupyter notebook", "colab", "google colab", "poetry", "conda",
    # security / misc
    "oauth2", "jwt", "sso", "saml", "blockchain", "iot", "embedded systems",
]

# Generic soft-skill phrases: never treated as forbidden even if the JD lists them.
SOFT_SKILL_STOPLIST = {
    "communication", "teamwork", "collaboration", "leadership", "problem solving", "problem-solving",
    "analytical skills", "critical thinking", "time management", "adaptability", "creativity",
    "attention to detail", "ownership", "curiosity", "self-motivated", "fast learner",
}

_NUMBER_RE = re.compile(r"(?<![\w.\-])\d[\d,]*(?:\.\d+)?")


def _as_dict(result: Union[TailoredResume, dict]) -> dict:
    return result.model_dump() if hasattr(result, "model_dump") else result


def _canonical_allowed(allowed: Iterable[str]) -> Set[str]:
    canon: Set[str] = set()
    for skill in allowed:
        canon.add(normalize_skill(skill))
        for part in expand_skill(skill):
            canon.add(normalize_skill(part))
    return canon


def build_vocabulary(allowed: Iterable[str], extra_terms: Iterable[str] = ()) -> List[str]:
    """Vocabulary of terms to look for: curated list + candidate skills + JD-only extras."""
    vocab = {t.lower() for t in COMMON_TECH_TERMS}
    vocab.update(s.lower().strip() for s in allowed if s and s.strip())
    vocab.update(t.lower().strip() for t in extra_terms
                 if t and t.strip() and t.lower().strip() not in SOFT_SKILL_STOPLIST)
    return sorted(vocab, key=len, reverse=True)  # longest first => longest match wins


def extract_candidate_skill_tokens(text: str, vocabulary: Iterable[str] = ()) -> List[str]:
    """Return every vocabulary term found in `text` (whole-term matches only, no substrings)."""
    vocab = sorted(set(vocabulary) or {t.lower() for t in COMMON_TECH_TERMS}, key=len, reverse=True)
    if not vocab:
        return []
    pattern = re.compile(
        r"(?<![\w.+#\-])(?:" + "|".join(re.escape(t) for t in vocab) + r")(?![\w+#\-])",
        re.IGNORECASE,
    )
    return [m.group(0).lower() for m in pattern.finditer(text)]


def _text_blob(result: dict) -> str:
    chunks = [result.get("summary", "")]
    for group in result.get("skills", []):
        chunks.append(group.get("category", ""))
    chunks += [b.get("text", "") for b in result.get("experience_bullets", [])]
    for project in result.get("projects", []):
        chunks.append(project.get("name", ""))
        chunks += [b.get("text", "") for b in project.get("bullets", [])]
    return " \n ".join(chunks)


def find_unauthorized_skills(result: Union[TailoredResume, dict], allowed: Set[str],
                             extra_forbidden: Iterable[str] = ()) -> List[str]:
    """Return every skill/tool in the tailored output that is not in the candidate's allowed list.

    Checks (a) free text: summary, bullets, project names, (b) every item of the skills section.
    `extra_forbidden` are JD-only skills (gap_analysis.missing_entirely_skills) - added to the
    vocabulary so even niche fabrications the curated list does not know are caught.
    """
    data = _as_dict(result)
    allowed_canon = _canonical_allowed(allowed)
    vocab = build_vocabulary(allowed, extra_forbidden)

    violations: Set[str] = set()
    for term in extract_candidate_skill_tokens(_text_blob(data), vocab):
        if normalize_skill(term) not in allowed_canon:
            violations.add(term)

    for group in data.get("skills", []):
        for item in group.get("items", []):
            parts = expand_skill(item) or [item]
            if not all(normalize_skill(p) in allowed_canon for p in parts):
                violations.add(item.lower())
    return sorted(violations)


def find_invented_numbers(base_resume: dict, result: Union[TailoredResume, dict]) -> List[str]:
    """Numbers/metrics in the tailored output that do not appear anywhere in the source resume."""
    source_chunks = [base_resume.get("summary", "")]
    source_chunks += [b["text"] for b in base_resume.get("experience_bullets", [])]
    for project in base_resume.get("projects", []):
        source_chunks.append(project.get("name", ""))
        source_chunks += [b["text"] for b in project.get("bullets", [])]
    source_chunks.append(base_resume.get("experience_heading", ""))
    source_numbers = {n.rstrip(",") for n in _NUMBER_RE.findall(" ".join(source_chunks))}
    output_numbers = {n.rstrip(",") for n in _NUMBER_RE.findall(_text_blob(_as_dict(result)))}
    return sorted(output_numbers - source_numbers)


def enforce_ground_truth(base_resume: dict, result: Union[TailoredResume, dict],
                         ensure_skills: Iterable[str] = ()) -> TailoredResume:
    """Rebuild the LLM output so only *wording* (and skill ordering) can differ from the source.

    - experience/project bullets: emitted in source order with source ids; text comes from the LLM
      if it returned that id, otherwise the original text is kept. skills_used always from source.
    - projects: source names, order and tech_stack; extra/missing LLM projects are ignored/repaired.
    - empty summary or skills fall back to the source.
    """
    data = _as_dict(result)

    llm_bullets = {b["id"]: b for b in data.get("experience_bullets", [])}
    bullets = [
        {"id": b["id"], "skills_used": b["skills_used"],
         "text": (llm_bullets.get(b["id"], {}).get("text") or "").strip() or b["text"]}
        for b in base_resume["experience_bullets"]
    ]

    llm_projects = {p["name"].strip().lower(): p for p in data.get("projects", [])}
    projects = []
    for base_project in base_resume["projects"]:
        llm_project = llm_projects.get(base_project["name"].strip().lower(), {})
        llm_pb = {b["id"]: b for b in llm_project.get("bullets", [])}
        projects.append({
            "name": base_project["name"],
            "tech_stack": base_project["tech_stack"],
            "bullets": [
                {"id": b["id"], "skills_used": b["skills_used"],
                 "text": (llm_pb.get(b["id"], {}).get("text") or "").strip() or b["text"]}
                for b in base_project["bullets"]
            ],
        })

    skills = [g for g in data.get("skills", []) if g.get("items")] or base_resume["skills"]
    present = {normalize_skill(p) for g in skills for i in g["items"] for p in (expand_skill(i) or [i])}
    missing = [x for x in ensure_skills if x.strip() and normalize_skill(x) not in present]
    if missing:  # a skill the user ticked must never be silently lost by the LLM
        skills = list(skills) + [{"category": "Additional Skills", "items": missing}]
    summary = (data.get("summary") or "").strip() or base_resume["summary"]
    return TailoredResume.model_validate(
        {"summary": summary, "skills": skills, "experience_bullets": bullets, "projects": projects}
    )


def apply_confirmed_skills(base_resume: dict, confirmed: Iterable[str]) -> dict:
    """Return a copy of the resume where the user-ticked skills count as owned.

    They are added as an 'Additional Skills' group and to all_known_skills, so the guardrail allows
    them. They are *user-asserted*: the tool cannot verify them. `confirmed_skills` records which.
    """
    known = {normalize_skill(s) for s in base_resume["all_known_skills"]}
    new: List[str] = []
    for skill in confirmed:
        skill = (skill or "").strip()
        norm = normalize_skill(skill)
        if skill and norm not in known:
            known.add(norm)
            new.append(skill)
    out = copy.deepcopy(base_resume)
    out["confirmed_skills"] = new
    if new:
        out["skills"] = list(out["skills"]) + [{"category": "Additional Skills", "items": new}]
        out["all_known_skills"] = list(out["all_known_skills"]) + new
    return out


def find_confirmed_in_bullets(result: Union[TailoredResume, dict], confirmed: Iterable[str]) -> List[str]:
    """User-confirmed skills may appear in Skills/Summary but never inside experience or project
    bullets - a bullet is a claim about work actually done, which the user did not provide."""
    confirmed = [c for c in confirmed if c and c.strip()]
    if not confirmed:
        return []
    data = _as_dict(result)
    bullets = [b.get("text", "") for b in data.get("experience_bullets", [])]
    for project in data.get("projects", []):
        bullets += [b.get("text", "") for b in project.get("bullets", [])]
    canon = {normalize_skill(c) for c in confirmed}
    found = extract_candidate_skill_tokens(" \n ".join(bullets), [c.lower() for c in confirmed])
    return sorted({t for t in found if normalize_skill(t) in canon})


def check_tailored_resume(base_resume: dict, result: Union[TailoredResume, dict],
                          forbidden: Iterable[str] = (), confirmed: Iterable[str] = ()) -> List[str]:
    """Full guardrail: unauthorized skills + invented numbers + confirmed skills misused in bullets.
    Empty list == output is safe."""
    allowed = {s.lower() for s in base_resume["all_known_skills"]}
    problems = list(find_unauthorized_skills(result, allowed, forbidden))
    problems += [f"invented number '{n}'" for n in find_invented_numbers(base_resume, result)]
    problems += [f"'{t}' (user-confirmed skill) must not appear inside bullets"
                 for t in find_confirmed_in_bullets(result, confirmed)]
    return problems

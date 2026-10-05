# JD-to-Resume Tailor

A multi-agent Python app that takes a job description (JD) plus your structured resume data, analyses the gap, and produces a tailored, ATS-optimised one-page resume (`.docx` + `.pdf`) **without inventing any skill, tool or metric you don't actually have**. It reports a before/after ATS keyword-match score and requires a human to approve the diff before any file is generated.

Built with LangGraph, Gemini (via LangChain), Pydantic, FastAPI and python-docx. Runtime: Python 3.12.

## Workflow

The pipeline is a linear LangGraph `StateGraph`; every stage is an independently testable node sharing one state object.

| # | Stage / Node | Type | Purpose |
|---|---|---|---|
| 1 | JD Input | I/O | Accept raw JD text. |
| 2 | JD Parser (`parse_jd`) | LLM (Gemini) | Extract required skills, tools, seniority and top keywords as validated JSON. |
| 3 | Gap Analysis (`analyze_gap`) | Pure Python | Split JD skills into matched / missing-but-true / missing-entirely. |
| 4 | Content Rewrite (`rewrite_content`) | LLM + guardrail | Reorder skills and reword existing bullets; output is verified in code. |
| 5 | ATS Scorer (`score_ats`) | Pure Python | Keyword-match % before vs after. |
| 6 | Human Review + Export (`export`) | I/O + rendering | Show a diff; render `.docx`/`.pdf` only after `approved=true`. |

`analyze_gap` is the graph node name for Stage 3 because LangGraph 0.2.62 rejects a node named `gap_analysis` when the state also has a `gap_analysis` key.

## How to use (UI)

1. **My resume tab:** upload your PDF/DOCX/TXT resume. AI structures it, and anything not literally in your file is removed. Review the JSON, give it a name and save. (Your starter resume is `data/resume.json`.)
2. **Tailor tab:** pick the resume, paste the JD, click **Analyze job description**.
3. **Tick the extra skills you genuinely have** from the list of JD skills your resume doesn't show. Ticked skills are added to the Skills section and may be mentioned in the summary. They are never placed inside experience/project bullets.
4. Click **Generate tailored resume**, then **edit** the summary, skills and bullets freely. **Save edits & update ATS score** refreshes the ATS match instantly (pure Python, no LLM call).
5. **Approve and generate files** to get the `.docx`/`.pdf`.

Ticking a skill is *your* assertion: the tool cannot verify it. Tick only what you could discuss in an interview.

## Setup

```bash
# 1. Python 3.12 virtual environment
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Pinned dependencies
pip install -r requirements.txt

# 3. API key
cp .env.example .env              # then set GEMINI_API_KEY=...

# 4. Run the API
uvicorn app.main:app --reload --port 8000

# 5. (optional) run the UI in a second terminal
streamlit run ui/streamlit_app.py
```

Or run `./run.sh` (creates the venv, installs, starts the API).

**Model:** the default is `gemini-2.5-flash`; the spec's `gemini-1.5-flash` has been retired by Google. Set `GEMINI_MODEL` in `.env` to use another model.

### PDF conversion

- **Windows / macOS:** `docx2pdf` is used (needs Microsoft Word installed).
- **Linux (or if docx2pdf fails):** LibreOffice headless is used:
  ```bash
  soffice --headless --convert-to pdf resume.docx
  ```
- If neither is available, the `.docx` is still produced and the API response carries a warning.

### API

| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze` | Body `{jd_text, resume_id}`. Stages 2-3: parsed JD, gap analysis, `selectable_skills` for the checkboxes, `session_id`. |
| POST | `/tailor` | Body `{jd_text, resume_id, session_id?, confirmed_skills?}`. Runs stages 1-5 (reuses the `/analyze` result if `session_id` is given). Returns draft, diff, keyword table, `ats_score_before/after`. |
| POST | `/tailor/update` | Body `{session_id, tailored_resume}`. Saves your edits; returns the refreshed ATS score, keyword table and warnings. |
| POST | `/resume/parse` | Body `{filename, content_base64}`. Reads an uploaded PDF/DOCX/TXT into the resume schema. |
| POST | `/resume/save` | Body `{resume_id, resume, overwrite?}`. Validates and stores `data/{resume_id}.json`. |
| GET | `/resumes` | List stored resume ids. |
| POST | `/tailor/approve` | Body `{session_id, approved}`. Stage 6: exports only if `approved` is true. Returns file paths and download URLs. |
| GET | `/resume/{resume_id}` | The stored base resume (`data/{resume_id}.json`). |
| GET | `/health` | Liveness. |
| GET | `/download/{session_id}/{docx\|pdf}` | Download a generated file. |

```bash
curl -X POST localhost:8000/tailor -H 'Content-Type: application/json' \
  -d '{"jd_text": "<paste JD>", "resume_id": "resume"}'
curl -X POST localhost:8000/tailor/approve -H 'Content-Type: application/json' \
  -d '{"session_id": "<from above>", "approved": true}'
```

`data/resume.json` is a **dummy sample resume** (fake name and contact) so the project runs and tests pass out of the box. Put your real resume in the UI's **My resume** tab (saved as e.g. `data/my_resume.json`) or copy the sample's structure. Everything in `data/*.json` except `resume.json` is git-ignored, so your real resume is never published. Each bullet carries `skills_used` tags, which are the ground truth the guardrail checks against. `all_known_skills` is computed at load time from your skills section, bullet tags and tech stacks; parenthetical and slash lists are split into atomic skills (e.g. `Chroma/Qdrant` -> `Chroma`, `Qdrant`).

## Guardrail: anti-fabrication

Generic AI resume writers add skills to match a JD, which falls apart in interviews. Here the rule is enforced by code, not just by the prompt.

1. **Constrained rewrite.** The LLM sees only your resume, the matched keywords, and an explicit allowed-skill list. It is also told which JD skills you lack and that they must never appear.
2. **Deterministic verification** (`app/guardrail.py`) runs on every LLM output:
   - **Unknown skills:** the output text (summary, bullets, project names) is scanned against a vocabulary of about 200 AI/tech terms, your own skills, and the JD's missing-entirely skills. Any term not in your allowed list is a violation, so even niche fabrications the curated list doesn't know are caught through the JD gap.
   - **Skills section:** every item must be one of your existing skills, so nothing can be smuggled in via a parenthetical.
   - **Invented numbers:** any number not present in your source resume is rejected.
   - **Ground truth restored:** bullet ids, `skills_used` tags, project names and tech stacks are always overwritten from your base resume. Only wording and skill ordering can come from the LLM.
3. **User-confirmed skills.** Skills you tick are treated as yours (the tool can't verify them) but are kept out of experience/project bullets, so no invented achievement can be attached to them. Unticked JD skills stay blocked. In the edit boxes your own typing is not blocked, but any skill or number not in your resume or confirmed list is flagged with a warning.
4. **Retry, then fail loudly.** A violation triggers one retry with the offending terms named. A second failure raises `ValueError("Guardrail failed after retries: ...")`; nothing unsafe is ever returned.

Matching is whole-term and synonym-aware (`OpenAI` = `OpenAI API`, `LLMs` = `Large Language Models`, `React` = `React.js`), so `Java` does not match `JavaScript` and honest rewordings aren't flagged.

**Example: real output of the guardrail** on a deliberately fabricated draft (summary mentions Kubernetes/Docker/AWS, skills section adds Terraform, a bullet adds PyTorch and "35%"):

```
find_unauthorized_skills -> ['aws', 'docker', 'kubernetes', 'pytorch', 'terraform']
check_tailored_resume    -> ['aws', 'docker', 'kubernetes', 'pytorch', 'terraform', "invented number '35'"]
```

**Known limitation (v1):** detection is vocabulary-based (curated list plus the JD's missing skills), not NER. A fabricated skill that is neither in the curated list nor in the JD would not be flagged by the text scan. The skills-section check, the numbers check and the restored ground-truth fields still apply.

## Honest ATS scoring

`ats_score` is the percentage of the JD's `top_keywords` found in the resume text, using whole-term matching. It is computed the same way before and after, on the full resume. The before score is always measured on your original resume. Without ticking any skills, the score rises only when the rewrite surfaces skills you already have; ticking skills (or typing them yourself) raises it further, and that is your call.

## Tests

```bash
pytest
```

38 tests, no API key needed. They cover skill checkboxes, edit + instant rescoring, resume upload/parsing, the guardrail (including adversarial fabrications and false-positive checks against your real resume), gap analysis, ATS scoring, and an end-to-end graph run and API flow with a mocked LLM.

## Results

Scores depend on the job posting and a live Gemini run, so none are hard-coded here. To record yours: run the app on a real posting and note the before/after percentages shown in the UI or in the `/tailor` response.

| Job posting (redacted) | ATS before | ATS after | Missing skills correctly not added |
|---|---|---|---|

## Project structure

```
app/
  main.py            FastAPI endpoints
  graph.py           LangGraph wiring
  guardrail.py       anti-fabrication checks
  resume_render.py   python-docx rendering + PDF conversion
  resume_parser.py   PDF/DOCX/TXT upload -> resume schema (+ verification)
  llm.py             Gemini structured-output helper (one retry)
  skill_utils.py     skill normalisation / synonyms
  config.py          paths and env settings
  nodes/             parse_jd, gap_analysis, rewrite_content, ats_score, export
  models/schemas.py  all Pydantic models + pipeline state
data/resume.json     base resume
tests/               pytest suite
ui/streamlit_app.py  optional UI
```

## Publishing safely (GitHub)

Never commit `.env` (API key), real resumes (`data/*.json` except the sample), generated files (`output/`) or sessions (`data/sessions/`). `.gitignore` already excludes them. Before every push, run `git status` and check none of those appear.

## Using your own resume (privacy)

`data/resume.json` is a **dummy** resume (fake name and contact) so the project runs out of the box. To use your real resume, upload it in the UI (**My resume** tab) and save it under a name such as `my_resume`; it is stored as `data/my_resume.json`, which `.gitignore` keeps out of Git. Then pick `my_resume` in the **Tailor** tab. Never commit `.env` (your API key), generated resumes in `output/`, or your real resume JSON.

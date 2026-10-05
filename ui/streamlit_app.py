"""Streamlit front-end.

Start the API first:   uvicorn app.main:app --reload --port 8000
Then in a 2nd terminal: streamlit run ui/streamlit_app.py
"""
import base64
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `app.*` imports work from ui/
from app.skill_utils import format_skill_lines, parse_skill_lines  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="JD-to-Resume Tailor", layout="wide")
st.title("JD-to-Resume Tailor")


def api(method: str, path: str, **kwargs):
    """Call the backend; show a readable error and return None on failure."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=240, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}. Is uvicorn running? ({exc})")
        return None
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"{resp.status_code}: {detail}")
        return None
    return resp.json()


tab_resume, tab_tailor = st.tabs(["1. My resume", "2. Tailor to a job"])

# ======================================================================= TAB 1: resume upload
with tab_resume:
    listing = api("GET", "/resumes")
    resume_ids = listing["resumes"] if listing else ["resume"]
    st.write("Stored resumes: " + ", ".join(f"`{r}`" for r in resume_ids))

    st.subheader("Upload a new resume")
    st.caption("PDF, DOCX or TXT. AI structures it; anything not literally in your file is removed.")
    uploaded = st.file_uploader("Resume file", type=["pdf", "docx", "txt"])
    if uploaded and st.button("Read my resume"):
        with st.spinner("Reading your resume..."):
            res = api("POST", "/resume/parse", json={
                "filename": uploaded.name,
                "content_base64": base64.b64encode(uploaded.getvalue()).decode()})
        if res:
            st.session_state["parsed_json"] = json.dumps(res["resume"], indent=2, ensure_ascii=False)
            st.session_state["parse_warnings"] = res["warnings"]

    if "parsed_json" in st.session_state:
        for warning in st.session_state.get("parse_warnings", []):
            st.warning(warning)
        st.text_area("Review and fix anything wrong (this is your resume data)", key="parsed_json", height=420)
        c1, c2 = st.columns([2, 1])
        new_id = c1.text_input("Save as (letters, digits, _ or -)", value="my_resume")
        overwrite = c2.checkbox("Overwrite if it exists")
        if st.button("Save resume", type="primary"):
            try:
                resume_obj = json.loads(st.session_state["parsed_json"])
            except json.JSONDecodeError as exc:
                st.error(f"That is not valid JSON: {exc}")
            else:
                saved = api("POST", "/resume/save", json={"resume_id": new_id, "resume": resume_obj,
                                                           "overwrite": overwrite})
                if saved:
                    st.success(f"Saved as '{saved['resume_id']}'. Select it in the next tab.")

# ======================================================================= TAB 2: tailoring
with tab_tailor:
    resume_id = st.selectbox("Resume to use", resume_ids)
    jd_text = st.text_area("Paste the job description", height=240)

    if st.button("1. Analyze job description", type="primary", disabled=len(jd_text.strip()) < 20):
        with st.spinner("Parsing the job description..."):
            res = api("POST", "/analyze", json={"jd_text": jd_text, "resume_id": resume_id})
        if res:
            st.session_state["analysis"] = res
            st.session_state.pop("draft", None)
            st.session_state.pop("exported", None)

    analysis = st.session_state.get("analysis")
    if analysis:
        aid = analysis["session_id"]
        st.divider()
        st.subheader("Step 2: choose the extra skills you genuinely have")
        st.metric("ATS match of your original resume", f"{analysis['ats_score_before']}%")
        st.caption("Tick a skill only if you really have it. The tool cannot verify it, and you may be asked "
                   "about it in the interview. Ticked skills go into your Skills section and summary - "
                   "never into the experience/project bullets.")
        options = analysis["selectable_skills"]
        chosen = []
        columns = st.columns(3)
        for index, item in enumerate(options):
            label = f"{item['skill']}  ·  {item['type'].replace('_', ' ')}"
            if columns[index % 3].checkbox(label, key=f"skill_{aid}_{index}"):
                chosen.append(item["skill"])
        extra = st.text_input("Any other skill you have? (comma-separated, optional)", key=f"extra_{aid}")
        chosen += [s.strip() for s in extra.split(",") if s.strip()]

        if st.button("3. Generate tailored resume", type="primary"):
            with st.spinner("Rewriting and checking the guardrail..."):
                res = api("POST", "/tailor", json={
                    "jd_text": analysis["jd_text"], "resume_id": analysis["resume_id"],
                    "session_id": aid, "confirmed_skills": chosen})
            if res:
                st.session_state["draft"] = res
                st.session_state.pop("exported", None)

    draft = st.session_state.get("draft")
    if draft:
        sid, key = draft["session_id"], f"{draft['session_id']}_{draft['draft_version']}"
        st.divider()
        st.subheader("Step 4: review, edit and save")
        metrics_box = st.container()  # filled at the end so it always shows the latest score

        tr = draft["tailored_resume"]
        summary = st.text_area("Summary", tr["summary"], key=f"sum_{key}", height=130)
        skills_text = st.text_area("Skills (one group per line:  Category: skill, skill, ...)",
                                   format_skill_lines(tr["skills"]), key=f"sk_{key}", height=220)

        exp_bullets = [{"id": b["id"], "skills_used": b["skills_used"],
                        "text": st.text_area(f"Experience bullet ({b['id']})", b["text"],
                                             key=f"b_{key}_{b['id']}", height=90)}
                       for b in tr["experience_bullets"]]
        projects = []
        for pi, project in enumerate(tr["projects"]):
            st.markdown(f"**{project['name']}**")
            bullets = [{"id": b["id"], "skills_used": b["skills_used"],
                        "text": st.text_area(f"Bullet ({b['id']})", b["text"], key=f"b_{key}_{b['id']}", height=90)}
                       for b in project["bullets"]]
            projects.append({"name": project["name"], "tech_stack": project["tech_stack"], "bullets": bullets})

        edited = {"summary": summary, "skills": parse_skill_lines(skills_text),
                  "experience_bullets": exp_bullets, "projects": projects}

        def save_edits():
            res = api("POST", "/tailor/update", json={"session_id": sid, "tailored_resume": edited})
            if res:
                st.session_state["draft"].update(res)
            return res

        b1, b2, b3 = st.columns(3)
        if b1.button("Save edits & update ATS score"):
            if save_edits():
                st.success("Saved. ATS score updated above.")
        if b2.button("Approve and generate files", type="primary"):
            if save_edits():
                res = api("POST", "/tailor/approve", json={"session_id": sid, "approved": True})
                if res:
                    st.session_state["exported"] = res
        if b3.button("Discard draft"):
            api("POST", "/tailor/approve", json={"session_id": sid, "approved": False})
            st.session_state.pop("draft", None)
            st.rerun()

        # ---- metrics (rendered last so they reflect any edit saved above)
        draft = st.session_state["draft"]
        with metrics_box:
            before, after = draft["ats_score_before"], draft["ats_score_after"]
            m1, m2, m3 = st.columns(3)
            m1.metric("ATS match before", f"{before}%")
            m2.metric("ATS match after", f"{after}%", delta=f"{round(after - before, 1)} pts")
            m3.metric("Skills you confirmed", len(draft.get("confirmed_skills", [])))
            for warning in draft.get("warnings", []):
                st.warning(warning)
            with st.expander("Keyword breakdown"):
                st.dataframe(pd.DataFrame(draft["keywords"]).rename(columns={
                    "keyword": "JD keyword", "in_original": "In original", "in_tailored": "In tailored"}),
                    use_container_width=True, hide_index=True)

    exported = st.session_state.get("exported")
    if exported:
        st.success("Files generated.")
        if exported.get("warning"):
            st.warning(exported["warning"])
        for fmt, url in exported["download_urls"].items():
            file_resp = requests.get(f"{API_URL}{url}", timeout=60)
            st.download_button(f"Download {fmt.upper()}", file_resp.content,
                               file_name=os.path.basename(exported["output_paths"][fmt]), key=f"dl_{fmt}")

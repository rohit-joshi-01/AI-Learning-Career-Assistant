from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import get_settings
from backend.services.pipeline import RecommendationPipeline

st.set_page_config(page_title="AI Career Assistant", page_icon="🎓", layout="wide")

st.title("🎓 AI Learning & Career Assistant")
st.caption("Upload your resume and get personalised career recommendations powered by AI.")
st.info("🔒 Your resume is processed locally and not stored. AI-generated guidance should be reviewed before use.")

backend_url = st.sidebar.text_input("Backend URL", value=get_settings()["streamlit_backend_url"])
use_backend = st.sidebar.checkbox("Use FastAPI backend", value=False)

st.sidebar.subheader("About")
st.sidebar.write("The LLM is used only for explanations, roadmaps, and interview tips. Recommendation ranking is deterministic.")

with st.form("profile_form"):
    st.header("1. Your Profile")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        career_goal = st.text_area("Career Goal", placeholder="e.g. Become an ML Engineer specialising in NLP")
        skills = st.multiselect(
            "Your Current Skills",
            options=[
                "Python", "SQL", "Machine Learning", "Deep Learning", "NLP", "Docker",
                "JavaScript", "React", "FastAPI", "Pandas", "NumPy", "TensorFlow", "PyTorch",
                "OpenCV", "Kubernetes", "AWS", "Tableau", "Power BI",
            ],
            default=[],
        )
    with col2:
        interests = st.multiselect(
            "Your Interests / Domains",
            options=[
                "AI/ML", "Data Science", "NLP", "Computer Vision", "Backend",
                "DevOps", "Cloud", "Security", "Analytics", "Frontend",
            ],
            default=[],
        )
        resume_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    submitted = st.form_submit_button("🚀 Get Recommendations", use_container_width=True)

if submitted:
    if not career_goal or not resume_file:
        st.warning("Please provide a career goal and upload a PDF resume.")
        st.stop()

    if use_backend:
        try:
            files = {"resume": (resume_file.name, resume_file.getvalue(), "application/pdf")}
            data = {
                "name": name,
                "career_goal": career_goal,
                "skills": ", ".join(skills),
                "interests": ", ".join(interests),
            }
            response = requests.post(f"{backend_url.rstrip('/')}/recommend", data=data, files=files, timeout=120)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            st.error(f"Backend request failed: {exc}")
            st.stop()
    else:
        tmp_path = ROOT / "datasets" / "_streamlit_resume.pdf"
        tmp_path.write_bytes(resume_file.getvalue())
        pipeline = RecommendationPipeline()
        result = pipeline.recommend_from_upload(
            {
                "name": name,
                "career_goal": career_goal,
                "skills": skills,
                "interests": interests,
            },
            str(tmp_path),
            top_k=5,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    st.success("Recommendations generated successfully.")
    profile = result.get("profile_summary", {})
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Skills in profile", profile.get("skills_count", 0))
    col_b.metric("Top recommendations", len(result.get("recommendations", [])))
    col_c.metric("Anonymous user ID", result.get("user_id", "n/a"))

    for idx, item in enumerate(result.get("recommendations", []), start=1):
        title = item.get("role_title") or item.get("title") or f"Recommendation {idx}"
        with st.expander(f"{idx}. {title} — {item.get('confidence_score', 0)}%", expanded=idx == 1):
            st.write(f"**Domain / Company:** {item.get('domain') or item.get('company', 'N/A')}")
            st.write(f"**Confidence:** {item.get('confidence_score', 0)}%")
            st.write("**Reasons**")
            for reason in item.get("reasons", []):
                st.write(f"- {reason}")
            st.write("**Skill gaps**")
            st.write(", ".join(item.get("skill_gaps", [])) or "None")
            st.write("**Explanation**")
            st.write(item.get("explanation", ""))
            st.write("**Learning roadmap**")
            for step in item.get("learning_roadmap", []):
                st.write(f"- {step}")
            st.write("**Interview tips**")
            for tip in item.get("interview_tips", []):
                st.write(f"- {tip}")
            st.caption(item.get("disclaimer", "These suggestions are AI-generated. Verify before acting on them."))

    st.subheader("Skill Gap Overview")
    gap_rows = []
    for item in result.get("recommendations", []):
        gap_rows.append(
            {
                "Recommendation": item.get("role_title") or item.get("title"),
                "Missing skills": len(item.get("skill_gaps", [])),
                "Confidence": item.get("confidence_score", 0),
            }
        )
    if gap_rows:
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True)
    else:
        st.info("No recommendations returned.")

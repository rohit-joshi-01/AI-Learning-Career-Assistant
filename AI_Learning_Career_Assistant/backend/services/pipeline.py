from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.core.config import get_data_dir
from backend.services.data_loader import build_all_indexes
from backend.services.llm_service import build_llm_output
from backend.utils.resume_parser import build_profile, extract_text_from_pdf
from recommendation_engine.embedding_generator import generate_embedding
from recommendation_engine.ranking_engine import rank_results
from recommendation_engine.vector_search import search_careers, search_internships, search_resources

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RecommendationPipeline:
    resources_df: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.resources_df = self._load_resources() if self.resources_df is None else self.resources_df
        build_all_indexes(force=False)

    def _load_resources(self) -> pd.DataFrame:
        path = get_data_dir() / "learning_resources.csv"
        return pd.read_csv(path)

    def _build_profile_text(self, profile: dict[str, Any]) -> str:
        parts = [
            " ".join(profile.get("skills", [])),
            " ".join(profile.get("interests", [])),
            profile.get("career_goal", ""),
            profile.get("resume_summary", ""),
            profile.get("education", {}).get("degree", ""),
            profile.get("education", {}).get("institution", ""),
            " ".join(profile.get("experience", [])),
        ]
        return "\n".join(part for part in parts if part).strip()

    def create_profile(self, user_inputs: dict[str, Any], resume_pdf_path: str) -> dict[str, Any]:
        resume_text = extract_text_from_pdf(resume_pdf_path)
        profile = build_profile(resume_text, user_inputs)
        profile["embedding"] = generate_embedding(self._build_profile_text(profile))
        return profile

    def recommend(self, profile: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        profile_embedding = profile.get("embedding", [])

        career_candidates = search_careers(profile_embedding, top_k=max(5, top_k))
        internship_candidates = search_internships(profile_embedding, top_k=max(5, top_k))

        career_ranked = rank_results([{**dict(item), "collection": "careers"} for item in career_candidates], profile, self.resources_df)
        internship_ranked = rank_results([{**dict(item), "collection": "internships"} for item in internship_candidates], profile, self.resources_df)

        # Provide a balanced recommendation mix: top careers + top internships.
        combined = career_ranked[: max(1, top_k // 2)] + internship_ranked[: max(1, top_k - max(1, top_k // 2))]
        ranked = sorted(combined, key=lambda row: row["confidence_score_raw"], reverse=True)

        seen_ids: set[str] = set()
        top_results = []
        for item in ranked:
            record_id = str(item.get("id") or "")
            if record_id and record_id not in seen_ids:
                seen_ids.add(record_id)
                top_results.append(item)
            if len(top_results) >= top_k:
                break

        enriched_results = []
        for item in top_results:
            context_docs = self._context_docs_for_item(item)
            llm_bundle = build_llm_output(profile, item, context_docs)
            enriched_results.append(
                {
                    **item,
                    "explanation": llm_bundle.get("explanation", ""),
                    "learning_roadmap": llm_bundle.get("learning_roadmap", []),
                    "interview_tips": llm_bundle.get("interview_tips", []),
                    "profile_summary": llm_bundle.get("profile_summary", ""),
                    "disclaimer": llm_bundle.get("disclaimer", ""),
                }
            )

        return {
            "user_id": profile.get("user_id"),
            "profile_summary": {
                "skills_count": len(profile.get("skills", [])),
                "interests": profile.get("interests", []),
                "career_goal": profile.get("career_goal", ""),
                "education": profile.get("education", {}),
            },
            "recommendations": enriched_results,
            "skill_gap_summary": [
                {
                    "title": item.get("role_title") or item.get("title"),
                    "skill_gaps": item.get("skill_gaps", []),
                }
                for item in enriched_results
            ],
        }

    def _context_docs_for_item(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        collection = item.get("collection")
        if collection == "careers":
            return search_resources(generate_embedding(item.get("description", "")), top_k=3)
        if collection == "internships":
            return search_resources(generate_embedding(" ".join(item.get("required_skills", []) if isinstance(item.get("required_skills"), list) else [str(item.get("required_skills", ""))])), top_k=3)
        return []

    def recommend_from_upload(self, user_inputs: dict[str, Any], resume_pdf_path: str, top_k: int = 5) -> dict[str, Any]:
        profile = self.create_profile(user_inputs, resume_pdf_path)
        return self.recommend(profile, top_k=top_k)

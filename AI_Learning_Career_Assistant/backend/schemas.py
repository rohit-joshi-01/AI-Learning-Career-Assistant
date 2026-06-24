from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProfileInput(BaseModel):
    name: str = Field(default="", description="Student name")
    skills: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    career_goal: str = Field(default="")


class RecommendationItem(BaseModel):
    id: str | None = None
    collection: str | None = None
    title: str | None = None
    role_title: str | None = None
    company: str | None = None
    location: str | None = None
    domain: str | None = None
    description: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    skill_gaps: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    confidence_score_raw: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    explanation: str = ""
    learning_roadmap: list[str] = Field(default_factory=list)
    interview_tips: list[str] = Field(default_factory=list)
    disclaimer: str = ""


class RecommendationResponse(BaseModel):
    user_id: str
    profile_summary: dict[str, Any]
    recommendations: list[RecommendationItem]
    skill_gap_summary: list[dict[str, Any]]

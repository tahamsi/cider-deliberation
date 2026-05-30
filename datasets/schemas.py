from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskExample(BaseModel):
    dataset: str
    id: str
    question: str
    choices: list[str] | None = None
    answer: str
    answer_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must be non-empty")
        return value


class PredictionRecord(BaseModel):
    dataset: str
    id: str
    method: str
    prediction: str
    prediction_index: int | None = None
    answer: str
    answer_index: int | None = None
    correct: bool
    confidence: float | None = None
    transcript: list[dict[str, Any]]
    exposure_matrix: list[list[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)

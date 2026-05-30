from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets.schemas import TaskExample


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    answer_index: int | None
    rationale: str
    confidence: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] | None = None


class BaseAgent:
    def __init__(self, name: str, model_name: str, seed: int = 0, **config: Any) -> None:
        self.name = name
        self.model_name = model_name
        self.seed = seed
        self.config = config

    def answer(self, task: TaskExample, context: list[dict[str, Any]] | None = None, mode: str = "independent") -> AgentResponse:
        raise NotImplementedError

    def config_dict(self) -> dict[str, Any]:
        return {"name": self.name, "model_name": self.model_name, "seed": self.seed, **self.config}

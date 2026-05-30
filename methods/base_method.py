from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agents.base import AgentResponse, BaseAgent
from datasets.schemas import TaskExample


@dataclass
class MethodResult:
    prediction: str
    prediction_index: int | None
    confidence: float | None
    transcript: list[dict[str, Any]]
    exposure_matrix: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMethod:
    name = "base"
    deviation_note = "abstract method"

    def __init__(self, agents: list[BaseAgent], seed: int, **params: Any) -> None:
        if not agents:
            raise ValueError("At least one agent is required")
        self.agents = agents
        self.seed = seed
        self.params = params

    def run(self, task: TaskExample) -> MethodResult:
        raise NotImplementedError

    def _response_record(self, agent: BaseAgent, response: AgentResponse, round_id: int, visible: list[int]) -> dict[str, Any]:
        return {
            "agent": agent.name,
            "round": round_id,
            "answer": response.answer,
            "answer_index": response.answer_index,
            "rationale": response.rationale,
            "confidence": response.confidence,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "cost_usd": response.cost_usd,
            "visible_prior_messages": visible,
            "metadata": response.metadata or {},
        }

    def _empty_exposure(self, n: int) -> np.ndarray:
        return np.zeros((n, n), dtype=float)

    def _majority(self, transcript: list[dict[str, Any]]) -> tuple[str, int | None, float]:
        votes = [(r["answer"], r.get("answer_index")) for r in transcript if r.get("answer")]
        if not votes:
            raise RuntimeError(f"{self.name} produced no votes")
        key_counts = Counter(votes)
        (answer, answer_index), count = key_counts.most_common(1)[0]
        return answer, answer_index, count / len(votes)

    def _weighted_vote(self, transcript: list[dict[str, Any]], weights: list[float]) -> tuple[str, int | None, float]:
        totals: dict[tuple[str, int | None], float] = {}
        for rec, weight in zip(transcript, weights):
            key = (rec["answer"], rec.get("answer_index"))
            totals[key] = totals.get(key, 0.0) + float(weight)
        if not totals:
            raise RuntimeError(f"{self.name} produced no weighted votes")
        key, total = max(totals.items(), key=lambda item: item[1])
        denom = sum(max(w, 0.0) for w in weights) or 1.0
        return key[0], key[1], total / denom

    def metadata(self) -> dict[str, Any]:
        return {"method": self.name, "seed": self.seed, "params": self.params, "deviation_note": self.deviation_note}

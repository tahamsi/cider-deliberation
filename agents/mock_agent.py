from __future__ import annotations

import hashlib
import random

from agents.base import AgentResponse, BaseAgent
from datasets.schemas import TaskExample


class MockAgent(BaseAgent):
    """Deterministic local agent for reproducible smoke tests."""

    def answer(self, task: TaskExample, context: list[dict] | None = None, mode: str = "independent") -> AgentResponse:
        key = f"{self.seed}:{self.name}:{task.dataset}:{task.id}:{len(context or [])}:{mode}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:12], 16))
        if task.choices:
            if "2 + 2" in task.question:
                idx = 1
            elif "sky is blue" in task.question.lower():
                idx = 1
            else:
                idx = rng.randrange(len(task.choices))
            answer = chr(ord("A") + idx)
            answer_index = idx
        else:
            numbers = [part for part in task.question.replace("?", " ").split() if part.isdigit()]
            answer = numbers[-1] if numbers else str(rng.randrange(10))
            answer_index = None
        confidence = 0.55 + (rng.random() * 0.4)
        return AgentResponse(
            answer=answer,
            answer_index=answer_index,
            rationale=f"mock deterministic response from {self.name}",
            confidence=round(confidence, 4),
            tokens_in=len(task.question.split()),
            tokens_out=8,
            cost_usd=0.0,
            metadata={"mock": True},
        )

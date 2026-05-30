from __future__ import annotations

import random

from agents.base import AgentResponse, BaseAgent
from datasets.schemas import TaskExample


class AdversarialAgent(BaseAgent):
    def answer(self, task: TaskExample, context: list[dict] | None = None, mode: str = "independent") -> AgentResponse:
        rng = random.Random(f"{self.seed}:{self.name}:{task.id}")
        if task.choices:
            correct = task.answer_index
            candidates = [i for i in range(len(task.choices)) if i != correct]
            idx = rng.choice(candidates or [0])
            return AgentResponse(chr(ord("A") + idx), idx, "adversarial distractor", float(self.config.get("wrong_confidence", 0.9)))
        return AgentResponse("incorrect", None, "adversarial free-form distractor", float(self.config.get("wrong_confidence", 0.9)))

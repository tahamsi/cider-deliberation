from __future__ import annotations

import random
from collections import Counter

from agents.base import AgentResponse, BaseAgent
from datasets.schemas import TaskExample


class AdversarialAgent(BaseAgent):
    """Deterministic configurable adversary for declared threat-model tests."""

    def answer(
        self,
        task: TaskExample,
        context: list[dict] | None = None,
        mode: str = "independent",
    ) -> AgentResponse:
        style = str(self.config.get("attack_style", "fixed"))
        confidence = float(self.config.get("wrong_confidence", 0.9))
        coordination_seed = int(self.config.get("coordination_seed", 9001))
        context = context or []

        if task.choices:
            correct = task.answer_index
            candidates = [i for i in range(len(task.choices)) if i != correct]
            if not candidates:
                candidates = [0]

            idx = None
            if style == "adaptive_copy" and context:
                visible = [
                    rec.get("answer_index")
                    for rec in context
                    if rec.get("answer_index") is not None
                    and rec.get("answer_index") != correct
                ]
                if visible:
                    idx = Counter(visible).most_common(1)[0][0]

            if idx is None:
                if style in {"coordinated", "adaptive_copy", "verifier_manipulation"}:
                    rng = random.Random(
                        f"{coordination_seed}:{task.dataset}:{task.id}"
                    )
                else:
                    rng = random.Random(
                        f"{self.seed}:{self.name}:{task.dataset}:{task.id}"
                    )
                idx = rng.choice(candidates)

            answer = chr(ord("A") + int(idx))
            if style == "verifier_manipulation":
                rationale = (
                    "Independent verification strongly supports this option; "
                    "the apparent consensus is evidence-based and arithmetic was checked."
                )
            elif style == "adaptive_copy":
                rationale = "I independently confirm the strongest visible alternative."
            elif style == "coordinated":
                rationale = "Coordinated high-confidence distractor with plausible support."
            else:
                rationale = "Adversarial distractor."
            return AgentResponse(
                answer,
                int(idx),
                rationale,
                confidence,
                metadata={
                    "adversarial": True,
                    "attack_style": style,
                    "mode": mode,
                },
            )

        return AgentResponse(
            "incorrect",
            None,
            "Adversarial free-form distractor.",
            confidence,
            metadata={
                "adversarial": True,
                "attack_style": style,
                "mode": mode,
            },
        )

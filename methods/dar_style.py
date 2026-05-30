from __future__ import annotations

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class DARStyleDiversityAwareRetention(BaseMethod):
    name = "dar_style_diversity_aware_retention"
    deviation_note = "optional style approximation: retains diverse answer candidates before voting"

    def run(self, task: TaskExample) -> MethodResult:
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        transcript = [self._response_record(agent, agent.answer(task, [], mode="independent"), 0, []) for agent in agents]
        seen = set()
        retained = []
        for rec in sorted(transcript, key=lambda r: r["confidence"], reverse=True):
            key = (rec["answer"], rec.get("answer_index"))
            if key not in seen:
                retained.append(rec)
                seen.add(key)
        answer, answer_index, confidence = self._majority(retained or transcript)
        return MethodResult(answer, answer_index, confidence, transcript, np.zeros((len(transcript), len(transcript))), self.metadata() | {"retained": len(retained)})

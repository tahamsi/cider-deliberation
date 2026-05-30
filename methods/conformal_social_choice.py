from __future__ import annotations

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class ConformalSocialChoice(BaseMethod):
    name = "conformal_social_choice"
    deviation_note = "adapted baseline: confidence thresholded majority without calibration split"

    def run(self, task: TaskExample) -> MethodResult:
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        transcript = [self._response_record(agent, agent.answer(task, [], mode="independent"), 0, []) for agent in agents]
        threshold = float(self.params.get("conformal_threshold", 0.6))
        filtered = [r for r in transcript if r["confidence"] >= threshold] or transcript
        answer, answer_index, confidence = self._majority(filtered)
        return MethodResult(answer, answer_index, confidence, transcript, np.zeros((len(transcript), len(transcript))), self.metadata() | {"threshold": threshold})

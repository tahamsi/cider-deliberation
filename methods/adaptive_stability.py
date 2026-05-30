from __future__ import annotations

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class AdaptiveStabilityDetection(BaseMethod):
    name = "adaptive_stability_detection"
    deviation_note = "optional adapted baseline: stops independent sampling after stable majority"

    def run(self, task: TaskExample) -> MethodResult:
        max_samples = int(self.params.get("num_samples", 7))
        patience = int(self.params.get("stability_patience", 2))
        transcript = []
        last_answer = None
        stable = 0
        for i in range(max_samples):
            agent = self.agents[i % len(self.agents)]
            rec = self._response_record(agent, agent.answer(task, [], mode="self_consistency"), 0, [])
            transcript.append(rec)
            current = self._majority(transcript)[:2]
            stable = stable + 1 if current == last_answer else 0
            last_answer = current
            if stable >= patience:
                break
        answer, answer_index, confidence = self._majority(transcript)
        return MethodResult(answer, answer_index, confidence, transcript, np.zeros((len(transcript), len(transcript))), self.metadata() | {"samples_used": len(transcript)})

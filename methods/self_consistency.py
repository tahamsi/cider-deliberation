from __future__ import annotations

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class SelfConsistency(BaseMethod):
    name = "self_consistency"
    deviation_note = "multiple independent samples from one agent; majority aggregation"

    def run(self, task: TaskExample) -> MethodResult:
        n = int(self.params.get("num_samples", 5))
        transcript = []
        for i in range(n):
            agent = self.agents[i % len(self.agents)]
            response = agent.answer(task, [], mode="self_consistency")
            transcript.append(self._response_record(agent, response, 0, []))
        answer, answer_index, confidence = self._majority(transcript)
        return MethodResult(answer, answer_index, confidence, transcript, self._empty_exposure(n), self.metadata())

from __future__ import annotations

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class MajorityVote(BaseMethod):
    name = "majority_vote"
    deviation_note = "independent agents with no cross-agent exposure"

    def run(self, task: TaskExample) -> MethodResult:
        n = int(self.params.get("num_agents", len(self.agents)))
        transcript = []
        for agent in self.agents[:n]:
            response = agent.answer(task, [], mode="independent")
            transcript.append(self._response_record(agent, response, 0, []))
        answer, answer_index, confidence = self._majority(transcript)
        return MethodResult(answer, answer_index, confidence, transcript, self._empty_exposure(len(transcript)), self.metadata())

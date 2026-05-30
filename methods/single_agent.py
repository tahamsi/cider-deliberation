from __future__ import annotations

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class SingleAgent(BaseMethod):
    name = "single_agent"
    deviation_note = "direct single response baseline"

    def run(self, task: TaskExample) -> MethodResult:
        response = self.agents[0].answer(task, [], mode="independent")
        transcript = [self._response_record(self.agents[0], response, 0, [])]
        return MethodResult(response.answer, response.answer_index, response.confidence, transcript, self._empty_exposure(1), self.metadata())

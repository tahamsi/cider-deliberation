from __future__ import annotations

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class AgentAuditorStyle(BaseMethod):
    name = "agentauditor_style"
    deviation_note = "style approximation: independent answers plus final auditor-weighted vote"

    def run(self, task: TaskExample) -> MethodResult:
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        transcript = []
        for agent in agents:
            response = agent.answer(task, [], mode="audit")
            transcript.append(self._response_record(agent, response, 0, []))
        weights = [max(0.1, float(rec["confidence"])) for rec in transcript]
        answer, answer_index, confidence = self._weighted_vote(transcript, weights)
        metadata = self.metadata() | {"auditor_weights": weights}
        return MethodResult(answer, answer_index, confidence, transcript, np.zeros((len(transcript), len(transcript))), metadata)

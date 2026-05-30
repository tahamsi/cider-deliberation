from __future__ import annotations

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class StandardMultiAgentDebate(BaseMethod):
    name = "standard_multi_agent_debate"
    deviation_note = "fully visible synchronous debate approximation"

    def run(self, task: TaskExample) -> MethodResult:
        rounds = int(self.params.get("rounds", 2))
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        transcript = []
        exposure = np.zeros((len(agents) * rounds, len(agents) * rounds), dtype=float)
        for r in range(rounds):
            for ai, agent in enumerate(agents):
                visible = list(range(len(transcript)))
                response = agent.answer(task, transcript, mode="debate")
                row = len(transcript)
                exposure[row, visible] = 1.0
                transcript.append(self._response_record(agent, response, r, visible))
        answer, answer_index, confidence = self._majority(transcript[-len(agents):])
        return MethodResult(answer, answer_index, confidence, transcript, exposure, self.metadata())

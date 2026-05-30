from __future__ import annotations

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


class FreeMADStyle(BaseMethod):
    name = "free_mad_style"
    deviation_note = "style approximation: agents see a seeded partial subset of prior messages"

    def run(self, task: TaskExample) -> MethodResult:
        rng = np.random.default_rng(self.seed)
        rounds = int(self.params.get("rounds", 2))
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        transcript = []
        exposure = np.zeros((len(agents) * rounds, len(agents) * rounds), dtype=float)
        for r in range(rounds):
            for agent in agents:
                prior = list(range(len(transcript)))
                visible = [i for i in prior if rng.random() < 0.5]
                context = [transcript[i] for i in visible]
                response = agent.answer(task, context, mode="free_mad")
                row = len(transcript)
                exposure[row, visible] = 1.0
                transcript.append(self._response_record(agent, response, r, visible))
        answer, answer_index, confidence = self._majority(transcript[-len(agents):])
        return MethodResult(answer, answer_index, confidence, transcript, exposure, self.metadata())

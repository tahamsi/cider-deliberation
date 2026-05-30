from __future__ import annotations

import numpy as np

from datasets.schemas import TaskExample
from methods.free_mad_style import FreeMADStyle


class C3StyleCausalCreditAnalysis(FreeMADStyle):
    name = "c3_style_causal_credit_analysis"
    deviation_note = "style approximation: estimates credit by downweighting exposed copy-like final votes"

    def run(self, task: TaskExample):
        result = super().run(task)
        exposure_load = result.exposure_matrix.sum(axis=1)
        weights = list(1.0 / (1.0 + exposure_load[-len(self.agents):]))
        final = result.transcript[-len(weights):]
        answer, answer_index, confidence = self._weighted_vote(final, weights)
        result.prediction = answer
        result.prediction_index = answer_index
        result.confidence = confidence
        result.metadata = self.metadata() | {"causal_credit_weights": weights}
        return result

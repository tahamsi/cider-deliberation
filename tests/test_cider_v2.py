from __future__ import annotations

from agents.base import AgentResponse, BaseAgent
from datasets.schemas import TaskExample
from experiments.run_benchmark import build_agents
from methods.cider_v2 import CiderAdaptiveGated


class ScriptedAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        independent_answer: str,
        final_answer: str,
        verifier_answer: str,
        persona: str,
    ) -> None:
        super().__init__(name, "scripted", persona=persona)
        self.independent_answer = independent_answer
        self.final_answer = final_answer
        self.verifier_answer = verifier_answer

    def answer(self, task, context=None, mode="independent"):
        if mode == "independent":
            answer = self.independent_answer
            rationale = "initial evidence"
        elif "verify" in mode:
            answer = self.verifier_answer
            rationale = "skeptical verifier decision"
        else:
            answer = self.final_answer
            rationale = "same"
        index = ord(answer) - ord("A")
        return AgentResponse(
            answer,
            index,
            rationale,
            0.8,
            metadata={"persona": self.config["persona"], "mode": mode},
        )


def task() -> TaskExample:
    return TaskExample(
        dataset="unit",
        id="1",
        question="Choose the correct option.",
        choices=["wrong", "right"],
        answer="B",
        answer_index=1,
    )


def test_cider_v2_early_stops_on_unanimous_answers():
    agents = [
        ScriptedAgent(f"a{i}", "B", "A", "B", "concise_solver")
        for i in range(4)
    ]
    result = CiderAdaptiveGated(agents, seed=7, num_agents=4).run(task())
    assert result.prediction == "B"
    assert result.metadata["v2_early_stopped"] is True
    assert len(result.transcript) == 4
    assert float(result.exposure_matrix.sum()) == 0.0


def test_cider_v2_rejects_unsupported_switches():
    agents = [
        ScriptedAgent("a0", "A", "B", "A", "concise_solver"),
        ScriptedAgent("a1", "B", "B", "A", "skeptical_solver"),
        ScriptedAgent("a2", "A", "B", "A", "counterexample_seeker"),
        ScriptedAgent("a3", "B", "B", "A", "domain_expert"),
    ]
    result = CiderAdaptiveGated(agents, seed=9, num_agents=4).run(task())
    switched = [
        item for item in result.metadata["v2_gate_decisions"] if item["switched"]
    ]
    assert switched
    assert all(item["accepted"] is False for item in switched)
    assert all(
        item["reason"] in {"unsupported_copy", "insufficient_evidence", "verifier_disagrees"}
        for item in switched
    )


def test_build_agents_supports_heterogeneous_model_pool():
    agents = build_agents(
        {
            "type": "mock",
            "model_names": ["model-a", "model-b"],
            "personas": ["concise_solver", "skeptical_solver"],
        },
        seed=1,
        n=4,
    )
    assert [agent.model_name for agent in agents] == [
        "model-a",
        "model-b",
        "model-a",
        "model-b",
    ]

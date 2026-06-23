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


def _dummy_agents(n: int = 4) -> list[ScriptedAgent]:
    """Provide valid inert agents for tests that call internal helpers directly."""
    return [
        ScriptedAgent(
            name=f"dummy_{index}",
            independent_answer="A",
            final_answer="A",
            verifier_answer="A",
            persona="general_solver",
        )
        for index in range(n)
    ]


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

def _record(
    agent: str,
    answer: str,
    confidence: float,
    rationale: str,
    *,
    visible: list[int] | None = None,
):
    return {
        "agent": agent,
        "answer": answer,
        "answer_index": ord(answer) - ord("A"),
        "confidence": confidence,
        "rationale": rationale,
        "visible_prior_messages": list(visible or []),
        "metadata": {"persona": "general_solver", "model_name": "scripted"},
    }


def test_selective_gate_accepts_corroborated_recovery(monkeypatch):
    method = CiderAdaptiveGated(
        _dummy_agents(4),
        seed=1,
        num_agents=4,
        v2_gate_mode="selective",
    )
    monkeypatch.setattr(method, "_evidence_improvement", lambda *_: 0.18)
    monkeypatch.setattr(method, "_answer_validity", lambda *_: 1.0)
    monkeypatch.setattr(
        method,
        "_rationale_quality",
        lambda rec: 0.35 if rec["answer"] == "A" else 0.78,
    )
    monkeypatch.setattr(method, "_deterministic_numeric_check", lambda *_: None)

    independent = [
        _record("a0", "A", 0.35, "uncertain initial guess"),
        _record("a1", "B", 0.78, "independent derivation one"),
        _record("a2", "B", 0.74, "independent derivation two"),
        _record("a3", "B", 0.72, "independent derivation three"),
    ]
    proposals = [
        _record("a0", "B", 0.58, "corrected after locating the error"),
        _record("a1", "B", 0.80, "stable"),
        _record("a2", "B", 0.76, "stable"),
        _record("a3", "B", 0.73, "stable"),
    ]
    decision = method._gate_switch(
        task(),
        independent[0],
        proposals[0],
        independent + proposals,
        ("B", 1),
        agent_index=0,
        independent=independent,
        proposals=proposals,
    )

    assert decision["accepted"] is True
    assert decision["reason"] in {
        "corroborated_recovery",
        "peer_confirmed_correction",
        "selective_score",
    }
    assert decision["acceptance_weight"] >= 0.78
    assert decision["proposal_independent_peer_support"] == 3


def test_selective_gate_protects_strong_supported_initial(monkeypatch):
    method = CiderAdaptiveGated(
        _dummy_agents(4),
        seed=1,
        num_agents=4,
        v2_gate_mode="selective",
    )
    monkeypatch.setattr(method, "_evidence_improvement", lambda *_: 0.18)
    monkeypatch.setattr(method, "_answer_validity", lambda *_: 1.0)
    monkeypatch.setattr(method, "_rationale_quality", lambda *_: 0.80)
    monkeypatch.setattr(method, "_deterministic_numeric_check", lambda *_: None)

    independent = [
        _record("a0", "A", 0.95, "strong independent derivation"),
        _record("a1", "A", 0.91, "separate supporting derivation"),
        _record("a2", "A", 0.88, "third supporting derivation"),
        _record("a3", "B", 0.70, "minority view"),
    ]
    proposals = [
        _record("a0", "B", 0.70, "tentative revision"),
        _record("a1", "A", 0.90, "stable"),
        _record("a2", "A", 0.87, "stable"),
        _record("a3", "B", 0.72, "stable"),
    ]
    decision = method._gate_switch(
        task(),
        independent[0],
        proposals[0],
        independent + proposals,
        ("B", 1),
        agent_index=0,
        independent=independent,
        proposals=proposals,
    )

    assert decision["protected_initial"] is True
    assert decision["accepted"] is False
    assert decision["reason"] == "protected_initial"



def test_selective_gate_keeps_unsupported_copy_veto(monkeypatch):
    method = CiderAdaptiveGated(
        _dummy_agents(4),
        seed=1,
        num_agents=4,
        v2_gate_mode="selective",
    )
    monkeypatch.setattr(method, "_evidence_improvement", lambda *_: 0.02)
    monkeypatch.setattr(method, "_answer_validity", lambda *_: 1.0)
    monkeypatch.setattr(method, "_rationale_quality", lambda *_: 0.60)
    monkeypatch.setattr(method, "_deterministic_numeric_check", lambda *_: None)

    # No independent or post-exposure peer supports B.
    independent = [
        _record("a0", "A", 0.60, "original reasoning"),
        _record("a1", "C", 0.60, "different independent view"),
        _record("a2", "D", 0.60, "another independent view"),
        _record("a3", "E", 0.60, "final independent view"),
    ]

    proposals = [
        _record(
            "a0",
            "B",
            0.61,
            "copied phrase with exact structure",
            visible=[1],
        ),
        _record("a1", "C", 0.60, "stable"),
        _record("a2", "D", 0.60, "stable"),
        _record("a3", "E", 0.60, "stable"),
    ]

    # The only visible B record is an exposed message, not independent
    # corroboration from another agent.
    visible_copy = _record(
        "visible_source",
        "B",
        0.95,
        "copied phrase with exact structure",
    )
    transcript = [
        independent[0],
        visible_copy,
        *independent[1:],
        *proposals,
    ]

    decision = method._gate_switch(
        task(),
        independent[0],
        proposals[0],
        transcript,
        ("A", 0),
        agent_index=0,
        independent=independent,
        proposals=proposals,
    )

    assert decision["copied_visible_majority"] is True
    assert decision["copy_similarity"] >= 0.99
    assert decision["proposal_independent_peer_support"] == 0
    assert decision["proposal_post_exposure_peer_support"] == 0
    assert decision["unsupported_copy"] is True
    assert decision["accepted"] is False
    assert decision["reason"] == "unsupported_copy"


def test_candidate_features_fractionally_weight_accepted_correction(monkeypatch):
    method = CiderAdaptiveGated(_dummy_agents(1), seed=1, num_agents=1)
    monkeypatch.setattr(method, "_answer_validity", lambda *_: 1.0)
    monkeypatch.setattr(method, "_rationale_quality", lambda *_: 1.0)
    monkeypatch.setattr(method, "_role_weight", lambda *_: 1.0)

    independent = [_record("a0", "A", 0.80, "initial")]
    final = [_record("a0", "B", 0.80, "correction")]
    decisions = [
        {
            "accepted": True,
            "acceptance_weight": 0.60,
            "proposed_answer": "B",
            "proposed_answer_index": 1,
            "unsupported_copy": False,
        }
    ]
    rows = method._candidate_features(
        task=task(),
        independent=independent,
        final=final,
        proposals=final,
        decisions=decisions,
        verifier_record=None,
    )
    b_row = next(row for row in rows if row["answer"] == "B")
    assert b_row["features"]["accepted_correction_support"] == 0.48


def test_gate_summary_reports_acceptance_distribution():
    method = CiderAdaptiveGated(_dummy_agents(1), seed=1, num_agents=1)
    summary = method._gate_summary(
        [
            {"switched": False, "accepted": True, "reason": "stable"},
            {
                "switched": True,
                "accepted": True,
                "reason": "corroborated_recovery",
                "acceptance_score": 0.60,
                "acceptance_threshold": 0.40,
                "acceptance_weight": 0.80,
            },
            {
                "switched": True,
                "accepted": False,
                "reason": "unsupported_copy",
                "acceptance_score": 0.10,
                "acceptance_threshold": 0.50,
                "acceptance_weight": 0.0,
            },
        ]
    )
    assert summary["switches_proposed"] == 2
    assert summary["switches_accepted"] == 1
    assert summary["switch_acceptance_rate"] == 0.5
    assert summary["reasons"] == {
        "corroborated_recovery": 1,
        "unsupported_copy": 1,
    }

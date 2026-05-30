from agents.mock_agent import MockAgent
from datasets.schemas import TaskExample
from methods import METHODS


def test_all_methods_smoke():
    task = TaskExample(dataset="x", id="1", question="What is 2 + 2?", choices=["3", "4"], answer="B", answer_index=1)
    agents = [MockAgent(f"a{i}", "mock", seed=i) for i in range(5)]
    for name, cls in METHODS.items():
        result = cls(agents, seed=11, num_agents=3, num_samples=5, rounds=2).run(task)
        assert result.prediction
        assert len(result.transcript) >= 1
        assert result.exposure_matrix.shape[0] == result.exposure_matrix.shape[1]
        assert "deviation_note" in result.metadata

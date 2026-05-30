from agents.mock_agent import MockAgent
from datasets.schemas import TaskExample
from methods.cider import CiderFull


def test_cider_exposure_shape():
    task = TaskExample(dataset="x", id="1", question="What is 2 + 2?", choices=["3", "4"], answer="B", answer_index=1)
    agents = [MockAgent(f"a{i}", "mock", seed=i) for i in range(3)]
    result = CiderFull(agents, seed=5, num_agents=3, rounds=2).run(task)
    assert result.exposure_matrix.shape == (6, 6)
    assert result.exposure_matrix.diagonal().sum() == 0

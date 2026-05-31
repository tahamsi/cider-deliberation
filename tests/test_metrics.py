from metrics.accuracy import extract_number, is_correct
from metrics.causal_influence import causal_thesis_metrics, exposure_density


def test_multiple_choice_accuracy():
    assert is_correct("B", "B", 1, 1)
    assert not is_correct("A", "B", 0, 1)


def test_numeric_extraction():
    assert extract_number("final answer is 42") == 42
    assert is_correct("42.0", "42")


def test_exposure_density():
    assert exposure_density([[0, 1], [0, 0]]) == 0.5


def test_causal_thesis_metrics_counts_correction_harm_and_contamination():
    transcript = [
        {"agent": "a0", "round": 0, "answer": "A", "answer_index": 0, "visible_prior_messages": [], "metadata": {"mode": "independent"}},
        {"agent": "a1", "round": 0, "answer": "B", "answer_index": 1, "visible_prior_messages": [], "metadata": {"mode": "independent"}},
        {"agent": "a0", "round": 1, "answer": "B", "answer_index": 1, "visible_prior_messages": [1], "metadata": {"mode": "cider_final"}},
        {"agent": "a1", "round": 1, "answer": "A", "answer_index": 0, "visible_prior_messages": [0], "metadata": {"mode": "cider_final"}},
    ]
    metrics = causal_thesis_metrics(transcript, "B", 1)
    assert metrics["ccs"] == 1.0
    assert metrics["chs"] == 1.0
    assert metrics["pci"] == 0.5
    assert metrics["pds"] == 1.0
    assert metrics["wcr"] == 0.0

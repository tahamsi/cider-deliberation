from metrics.accuracy import extract_number, is_correct
from metrics.causal_influence import exposure_density


def test_multiple_choice_accuracy():
    assert is_correct("B", "B", 1, 1)
    assert not is_correct("A", "B", 0, 1)


def test_numeric_extraction():
    assert extract_number("final answer is 42") == 42
    assert is_correct("42.0", "42")


def test_exposure_density():
    assert exposure_density([[0, 1], [0, 0]]) == 0.5

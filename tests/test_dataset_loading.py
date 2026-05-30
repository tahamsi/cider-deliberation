from datasets.loaders import load_from_config


def test_inline_dataset_loading():
    rows = load_from_config({
        "inline": [{
            "dataset": "x",
            "id": "1",
            "question": "Q?",
            "choices": ["A", "B"],
            "answer": "A",
            "answer_index": 0,
            "metadata": {},
        }]
    }, seed=1)
    assert rows[0].id == "1"
    assert rows[0].answer_index == 0

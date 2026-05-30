from pathlib import Path

import yaml

from experiments.run_benchmark import run


def test_smoke_reproducibility(tmp_path):
    config = yaml.safe_load(Path("configs/small_smoke_test.yaml").read_text())
    config["output_dir"] = str(tmp_path / "a")
    records_a, _ = run(config)
    config["output_dir"] = str(tmp_path / "b")
    records_b, _ = run(config)
    slim_a = [(r["method"], r["id"], r["prediction"], r["exposure_matrix"]) for r in records_a]
    slim_b = [(r["method"], r["id"], r["prediction"], r["exposure_matrix"]) for r in records_b]
    assert slim_a == slim_b

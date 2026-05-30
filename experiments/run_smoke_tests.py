#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_benchmark import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small_smoke_test.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    records, rows = run(config)
    expected = len(config["datasets"]["inline"]) * len(config["methods"])
    if len(records) != expected:
        raise RuntimeError(f"Smoke test wrote {len(records)} records, expected {expected}")
    print(f"Smoke test complete: {len(records)} predictions, {len(rows)} aggregate rows")


if __name__ == "__main__":
    main()

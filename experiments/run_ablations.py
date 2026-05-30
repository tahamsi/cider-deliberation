#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_benchmark import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for exposure in [0.0, 0.2, 0.5, 1.0]:
        cfg = deepcopy(base)
        cfg["methods"] = ["cider_full"]
        cfg.setdefault("method_params", {})["max_exposure_probability"] = exposure
        cfg["output_dir"] = str(Path(base.get("output_dir", "outputs/ablations")) / f"exposure_{exposure}")
        run(cfg)


if __name__ == "__main__":
    main()

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
    parser.add_argument("--config", default="configs/real_llm_v6_mistral_160.yaml")
    parser.add_argument("--out_dir", default="outputs/full_experiment_v6/ablations_mistral_40")
    parser.add_argument("--max_examples", type=int, default=40)
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base["datasets"] = {"paths": ["data/processed/real_llm_v6_mistral_160.jsonl"]}
    variants = {
        "cider_full": {"methods": ["cider_full"], "params": {}},
        "cider_full_tuned": {"methods": ["cider_full_tuned"], "params": {}},
        "cider_verified_verifier_on": {"methods": ["cider_verified"], "params": {"verified_use_verifier_agent": True}},
        "cider_verified_verifier_off": {"methods": ["cider_verified"], "params": {"verified_use_verifier_agent": False}},
        "cider_sota_verifier_on": {"methods": ["cider_sota"], "params": {"sota_use_verifier_agent": True}},
        "cider_sota_verifier_off": {"methods": ["cider_sota"], "params": {"sota_use_verifier_agent": False}},
        "cider_sota_exposure_0_00": {"methods": ["cider_sota"], "params": {"sota_max_exposure_probability": 0.0}},
        "cider_sota_exposure_0_20": {"methods": ["cider_sota"], "params": {"sota_max_exposure_probability": 0.2}},
        "cider_sota_exposure_0_35": {"methods": ["cider_sota"], "params": {"sota_max_exposure_probability": 0.35}},
        "cider_sota_exposure_0_50": {"methods": ["cider_sota"], "params": {"sota_max_exposure_probability": 0.5}},
        "cider_sota_exposure_1_00": {"methods": ["cider_sota"], "params": {"sota_max_exposure_probability": 1.0}},
    }

    for name, spec in variants.items():
        cfg = deepcopy(base)
        cfg["methods"] = spec["methods"]
        cfg["output_dir"] = str(Path(args.out_dir) / name)
        cfg.setdefault("method_params", {}).update(spec["params"])
        run(cfg, max_examples=args.max_examples)


if __name__ == "__main__":
    main()

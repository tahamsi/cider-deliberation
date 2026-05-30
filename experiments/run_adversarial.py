#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.adversarial_agent import AdversarialAgent
from experiments import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    original = run_benchmark.build_agents
    adversarial_params = cfg.get("adversarial", {})
    num_adversaries = int(adversarial_params.get("num_adversaries", 1))
    wrong_confidence = float(adversarial_params.get("wrong_confidence", 0.9))

    def build_with_adversary(config, seed, n):
        honest_n = max(n - num_adversaries, 1)
        agents = original(config, seed, honest_n)
        for i in range(num_adversaries):
            agents.append(
                AdversarialAgent(
                    f"adversary_{i}",
                    "deterministic-adversary",
                    seed + 999 + i,
                    wrong_confidence=wrong_confidence,
                )
            )
        return agents

    run_benchmark.build_agents = build_with_adversary
    run_benchmark.run(cfg, max_examples=args.max_examples)


if __name__ == "__main__":
    main()

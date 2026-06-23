#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_benchmark import run as run_benchmark


BASELINES = [
    "single_agent",
    "self_consistency",
    "majority_vote",
    "standard_multi_agent_debate",
    "free_mad_style",
    "free_mad_official_adapter",
    "dar_official_adapter",
    "cider_sota",
    "cider_adaptive_gated",
]

ADVERSARIAL_METHODS = [
    "majority_vote",
    "standard_multi_agent_debate",
    "free_mad_official_adapter",
    "dar_official_adapter",
    "cider_sota",
    "cider_adaptive_gated",
]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in split_csv(value)]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in split_csv(value)]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def base_agent(args: argparse.Namespace, model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "llm",
        "provider": "ollama",
        "model_name": model,
        "base_url": args.ollama_base_url,
        "max_tokens": args.max_tokens,
        "num_ctx": args.num_ctx,
        "timeout_seconds": args.timeout_seconds,
        "request_retries": 3,
        "cache": False,
        "personas": [
            "concise_solver",
            "skeptical_solver",
            "step_by_step_solver",
            "counterexample_seeker",
        ],
        "agent_temperatures": [0.0, 0.1, 0.2, 0.15],
    }
    if args.agent_model_pool:
        payload["model_names"] = split_csv(args.agent_model_pool)
    return payload


def method_params(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "num_agents": args.num_agents,
        "num_samples": args.num_agents,
        "rounds": 2,
        "v2_deliberation_threshold": args.deliberation_threshold,
        "v2_max_visible_messages": args.max_visible_messages,
        "v2_use_verifier": True,
        "v2_verifier_required_for_switch": True,
        "v2_gate_mode": args.gate_mode,
        "v2_min_evidence_gain": args.min_evidence_gain,
        "v2_soft_evidence_gain": args.soft_evidence_gain,
        "v2_min_confidence_gain": args.min_confidence_gain,
        "v2_switch_accept_threshold": args.switch_accept_threshold,
        "v2_weak_initial_confidence": args.weak_initial_confidence,
        "v2_strong_initial_confidence": args.strong_initial_confidence,
        "v2_weak_initial_threshold_relief": args.weak_initial_threshold_relief,
        "v2_peer_threshold_relief": args.peer_threshold_relief,
        "v2_post_exposure_threshold_relief": args.post_exposure_threshold_relief,
        "v2_protected_initial_penalty": args.protected_initial_penalty,
        "v2_verifier_disagreement_penalty": args.verifier_disagreement_penalty,
        "v2_copy_similarity_threshold": args.copy_similarity_threshold,
        "v2_allow_strong_switch_without_verifier": False,
    }
    if args.weight_file:
        payload["v2_weight_file"] = args.weight_file
    return payload


def make_config(
    args: argparse.Namespace,
    *,
    model: str,
    seed: int,
    methods: list[str],
    output_dir: Path,
    params_update: dict[str, Any] | None = None,
    adversarial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = method_params(args)
    if params_update:
        params.update(params_update)
    cfg: dict[str, Any] = {
        "seed": seed,
        "output_dir": str(output_dir),
        "agent": base_agent(args, model),
        "datasets": {"paths": [args.dataset]},
        "methods": methods,
        "method_params": params,
    }
    if adversarial:
        cfg["adversarial"] = adversarial
    return cfg


def run_one(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    config_path: Path,
    *,
    adversarial: bool = False,
) -> None:
    write_yaml(config_path, cfg)
    if args.dry_run:
        print(f"DRY-RUN {config_path}")
        return
    if adversarial:
        cmd = [
            sys.executable,
            "experiments/run_adversarial.py",
            "--config",
            str(config_path),
        ]
        if args.max_examples is not None:
            cmd += ["--max_examples", str(args.max_examples)]
        subprocess.run(cmd, cwd=ROOT, check=True)
    else:
        run_benchmark(cfg, max_examples=args.max_examples)


def collect(out_dir: Path, manifest: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for entry in manifest:
        metrics = Path(entry["output_dir"]) / "aggregate_metrics.csv"
        if not metrics.exists():
            continue
        with metrics.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(entry | row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if rows:
        columns = sorted({key for row in rows for key in row})
        with (out_dir / "campaign_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["dev", "mechanisms", "confirm", "adversarial-grid"],
        required=True,
    )
    parser.add_argument("--out_dir", default="outputs/cider_v2")
    parser.add_argument(
        "--dataset",
        default="data/processed/real_llm_v4_test_balanced.jsonl",
    )
    parser.add_argument("--models", default="mistral:latest,qwen3:32b")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--agent_model_pool", default="")
    parser.add_argument("--weight_file", default="")
    parser.add_argument("--ollama_base_url", default="http://127.0.0.1:11434")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument("--timeout_seconds", type=float, default=300)
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--deliberation_threshold", type=float, default=0.20)
    parser.add_argument("--max_visible_messages", type=int, default=2)
    parser.add_argument(
        "--gate_mode",
        choices=["selective", "strict"],
        default="selective",
    )
    parser.add_argument("--min_evidence_gain", type=float, default=0.25)
    parser.add_argument("--soft_evidence_gain", type=float, default=0.10)
    parser.add_argument("--min_confidence_gain", type=float, default=0.05)
    parser.add_argument("--switch_accept_threshold", type=float, default=0.52)
    parser.add_argument("--weak_initial_confidence", type=float, default=0.62)
    parser.add_argument("--strong_initial_confidence", type=float, default=0.82)
    parser.add_argument("--weak_initial_threshold_relief", type=float, default=0.08)
    parser.add_argument("--peer_threshold_relief", type=float, default=0.08)
    parser.add_argument("--post_exposure_threshold_relief", type=float, default=0.04)
    parser.add_argument("--protected_initial_penalty", type=float, default=0.14)
    parser.add_argument("--verifier_disagreement_penalty", type=float, default=0.10)
    parser.add_argument("--copy_similarity_threshold", type=float, default=0.72)
    parser.add_argument("--attackers", default="1,2,3")
    parser.add_argument("--attack_confidences", default="0.6,0.8,0.95")
    parser.add_argument(
        "--attack_styles",
        default="fixed,coordinated,adaptive_copy,verifier_manipulation",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    models = split_csv(args.models)
    seeds = parse_ints(args.seeds)
    out_dir = Path(args.out_dir) / args.stage
    config_dir = out_dir / "configs"
    manifest: list[dict[str, Any]] = []

    if args.stage == "confirm" and not args.weight_file:
        raise SystemExit("--stage confirm requires --weight_file from the frozen dev fit")

    if args.stage in {"dev", "confirm"}:
        for model in models:
            for seed in seeds:
                output = out_dir / safe_name(model) / f"seed_{seed}"
                cfg = make_config(
                    args,
                    model=model,
                    seed=seed,
                    methods=BASELINES,
                    output_dir=output,
                )
                config_path = config_dir / safe_name(model) / f"seed_{seed}.yaml"
                run_one(args, cfg, config_path)
                manifest.append(
                    {
                        "stage": args.stage,
                        "model": model,
                        "seed": seed,
                        "variant": "comparison",
                        "output_dir": str(output),
                        "config": str(config_path),
                    }
                )

    elif args.stage == "mechanisms":
        variants = {
            "selective_default": {},
            "legacy_strict_gate": {"v2_gate_mode": "strict"},
            "selective_relaxed": {
                "v2_switch_accept_threshold": 0.46,
                "v2_weak_initial_threshold_relief": 0.10,
                "v2_peer_threshold_relief": 0.10,
            },
            "selective_conservative": {
                "v2_switch_accept_threshold": 0.60,
                "v2_protected_initial_penalty": 0.18,
            },
            "selective_no_weak_initial_relief": {
                "v2_weak_initial_threshold_relief": 0.0,
            },
            "selective_no_peer_relief": {
                "v2_peer_threshold_relief": 0.0,
                "v2_post_exposure_threshold_relief": 0.0,
            },
            "selective_no_initial_protection": {
                "v2_protected_initial_penalty": 0.0,
            },
            "full_deliberation": {"v2_force_deliberation": True},
            "gate_off": {"v2_disable_gate": True},
            "verifier_off": {
                "v2_use_verifier": False,
                "v2_verifier_required_for_switch": False,
            },
            "role_weight_off": {"v2_disable_role_weight": True},
            "adaptive_routing_only": {
                "v2_disable_gate": True,
                "v2_use_verifier": False,
                "v2_verifier_required_for_switch": False,
            },
        }
        for model in models:
            for seed in seeds:
                for variant, update in variants.items():
                    output = (
                        out_dir
                        / safe_name(model)
                        / f"seed_{seed}"
                        / variant
                    )
                    cfg = make_config(
                        args,
                        model=model,
                        seed=seed,
                        methods=["cider_adaptive_gated"],
                        output_dir=output,
                        params_update=update,
                    )
                    config_path = (
                        config_dir
                        / safe_name(model)
                        / f"seed_{seed}"
                        / f"{variant}.yaml"
                    )
                    run_one(args, cfg, config_path)
                    manifest.append(
                        {
                            "stage": args.stage,
                            "model": model,
                            "seed": seed,
                            "variant": variant,
                            "output_dir": str(output),
                            "config": str(config_path),
                        }
                    )

    else:
        for model in models:
            for seed in seeds:
                for attackers in parse_ints(args.attackers):
                    for confidence in parse_floats(args.attack_confidences):
                        for style in split_csv(args.attack_styles):
                            variant = (
                                f"{style}_a{attackers}_c"
                                f"{str(confidence).replace('.', '_')}"
                            )
                            output = (
                                out_dir
                                / safe_name(model)
                                / f"seed_{seed}"
                                / variant
                            )
                            attack = {
                                "num_adversaries": attackers,
                                "wrong_confidence": confidence,
                                "attack_style": style,
                                "coordination_seed": 9001,
                            }
                            cfg = make_config(
                                args,
                                model=model,
                                seed=seed,
                                methods=ADVERSARIAL_METHODS,
                                output_dir=output,
                                adversarial=attack,
                            )
                            config_path = (
                                config_dir
                                / safe_name(model)
                                / f"seed_{seed}"
                                / f"{variant}.yaml"
                            )
                            run_one(args, cfg, config_path, adversarial=True)
                            manifest.append(
                                {
                                    "stage": args.stage,
                                    "model": model,
                                    "seed": seed,
                                    "variant": variant,
                                    "attackers": attackers,
                                    "attack_confidence": confidence,
                                    "attack_style": style,
                                    "output_dir": str(output),
                                    "config": str(config_path),
                                }
                            )

    collect(out_dir, manifest)
    print(out_dir / "campaign_results.csv")


if __name__ == "__main__":
    main()

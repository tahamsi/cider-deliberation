#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_benchmark import run as run_benchmark  # noqa: E402


ALL_METHODS = [
    "single_agent",
    "self_consistency",
    "majority_vote",
    "standard_multi_agent_debate",
    "consensagent_style",
    "free_mad_style",
    "free_mad_official_adapter",
    "agentauditor_style",
    "c3_style_causal_credit_analysis",
    "conformal_social_choice",
    "dar_style_diversity_aware_retention",
    "dar_official_adapter",
    "adaptive_stability_detection",
    "cider_full",
    "cider_full_tuned",
    "cider_verified",
    "cider_sota",
]

ADVERSARIAL_METHODS = [
    "majority_vote",
    "standard_multi_agent_debate",
    "free_mad_official_adapter",
    "dar_official_adapter",
    "cider_full",
    "cider_full_tuned",
    "cider_verified",
    "cider_sota",
]

TOKEN_SENSITIVITY_METHODS = [
    "single_agent",
    "majority_vote",
    "standard_multi_agent_debate",
    "cider_verified",
    "cider_sota",
]

SMOKE_DATASETS = {
    "inline": [
        {
            "dataset": "smoke_mc",
            "id": "mc_1",
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "answer": "B",
            "answer_index": 1,
            "metadata": {},
        },
        {
            "dataset": "smoke_bool",
            "id": "bool_1",
            "question": "The sky is blue on a clear day.",
            "choices": ["False", "True"],
            "answer": "B",
            "answer_index": 1,
            "metadata": {},
        },
    ]
}


def split_csv(value: str) -> list[str]:
    return [item.strip() for part in value.split(",") for item in part.split() if item.strip()]


def safe_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in model).strip("_").lower()


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in split_csv(value)]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in split_csv(value)]


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def agent_config(args: argparse.Namespace, model: str, max_tokens: int) -> dict[str, Any]:
    if args.provider == "mock":
        return {"type": "mock", "model_name": model}
    cfg: dict[str, Any] = {
        "type": "llm",
        "provider": "ollama",
        "base_url": args.ollama_base_url,
        "model_name": model,
        "model_selection_note": "A100 80GB campaign configuration.",
        "base_temperature": 0.0,
        "agent_temperatures": [0.0, 0.15, 0.3, 0.45],
        "personas": [
            "concise_solver",
            "skeptical_solver",
            "step_by_step_solver",
            "counterexample_seeker",
        ],
        "max_tokens": max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "num_ctx": args.num_ctx,
        "cache": True,
    }
    if model.lower().startswith("qwen3"):
        cfg["think"] = False
    return cfg


def dataset_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        return deepcopy(SMOKE_DATASETS)
    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}. Run scripts/download_preprocess_all.sh first.")
    return {"paths": [str(path)]}


def method_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_agents": args.num_agents,
        "num_samples": args.num_samples,
        "rounds": args.rounds,
        "max_exposure_probability": 0.35,
        "max_visible_messages": 2,
        "tuned_max_exposure_probability": 0.5,
        "tuned_max_visible_messages": 2,
        "verified_max_exposure_probability": 0.5,
        "verified_max_visible_messages": 2,
        "verified_use_verifier_agent": True,
        "verified_verifier_bonus": 1.25,
        "sota_max_exposure_probability": 0.35,
        "sota_max_visible_messages": 2,
        "sota_use_verifier_agent": True,
        "sota_verifier_tie_margin": 1.18,
        "sota_verifier_bonus": 0.75,
    }


def base_config(
    args: argparse.Namespace,
    *,
    model: str,
    seed: int,
    methods: list[str],
    output_dir: Path,
    max_tokens: int,
    params_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = method_params(args)
    if params_update:
        params.update(params_update)
    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "agent": agent_config(args, model, max_tokens),
        "datasets": dataset_config(args),
        "methods": methods,
        "method_params": params,
    }


def run_config(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    config_path: Path,
    manifest: list[dict[str, Any]],
    *,
    run_type: str,
    model: str,
    seed: int,
    variant: str,
    adversarial: bool = False,
) -> None:
    write_yaml(config_path, cfg)
    entry = {
        "run_type": run_type,
        "model": model,
        "seed": seed,
        "variant": variant,
        "config": str(config_path),
        "output_dir": str(cfg["output_dir"]),
        "adversarial": adversarial,
    }
    manifest.append(entry)
    if args.dry_run:
        print(f"DRY-RUN {run_type} {model} seed={seed} variant={variant} -> {cfg['output_dir']}")
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


def ablation_specs(args: argparse.Namespace) -> list[tuple[str, list[str], dict[str, Any]]]:
    specs: list[tuple[str, list[str], dict[str, Any]]] = []
    for exposure in args.exposure_grid:
        label = str(exposure).replace(".", "_")
        specs.append((
            f"cider_sota_exposure_{label}",
            ["cider_sota"],
            {"sota_max_exposure_probability": exposure},
        ))
        specs.append((
            f"cider_verified_exposure_{label}",
            ["cider_verified"],
            {"verified_max_exposure_probability": exposure},
        ))
    specs.extend([
        ("cider_sota_verifier_off", ["cider_sota"], {"sota_use_verifier_agent": False}),
        ("cider_verified_verifier_off", ["cider_verified"], {"verified_use_verifier_agent": False}),
        ("cider_sota_no_copy_penalty", ["cider_sota"], {"disable_copy_penalty": True}),
        ("cider_verified_no_copy_penalty", ["cider_verified"], {"disable_copy_penalty": True}),
        ("cider_sota_no_correction_bonus", ["cider_sota"], {"disable_correction_bonus": True}),
        ("cider_verified_no_correction_bonus", ["cider_verified"], {"disable_correction_bonus": True}),
        ("cider_sota_no_exposure_penalty", ["cider_sota"], {"disable_exposure_penalty": True}),
        ("cider_verified_no_exposure_penalty", ["cider_verified"], {"disable_exposure_penalty": True}),
    ])
    return specs


def collect_results(out_dir: Path, manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for entry in manifest:
        metrics_path = Path(entry["output_dir"]) / "aggregate_metrics.csv"
        if not metrics_path.exists():
            continue
        for row in load_csv(metrics_path):
            rows.append(entry | row)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "a100_campaign_results.csv"
    if rows:
        columns = sorted({key for row in rows for key in row})
        with results_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "a100_campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return rows, manifest


def best_rows(rows: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    def acc(row: dict[str, Any]) -> float:
        try:
            return float(row.get("accuracy", 0.0))
        except Exception:
            return 0.0

    return sorted(rows, key=acc, reverse=True)[:limit]


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(args: argparse.Namespace, out_dir: Path, rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> Path:
    summary = out_dir / "a100_campaign_summary.md"
    models = sorted({entry["model"] for entry in manifest})
    seeds = sorted({str(entry["seed"]) for entry in manifest})
    run_types = sorted({entry["run_type"] for entry in manifest})
    causal_cols = ["ccs", "chs", "pci", "wcr", "pds"]
    has_causal = bool(rows) and all(col in rows[0] for col in causal_cols)
    lines = [
        "# A100 Full Campaign Summary",
        "",
        f"Generated: {datetime.now().replace(microsecond=0).isoformat()}",
        "",
        "## Scope",
        "",
        f"- Smoke mode: `{args.smoke}`",
        f"- Dry run: `{args.dry_run}`",
        f"- Dataset: `{args.dataset if not args.smoke else 'inline smoke dataset'}`",
        f"- Models: `{', '.join(models)}`",
        f"- Seeds: `{', '.join(seeds)}`",
        f"- Run types: `{', '.join(run_types)}`",
        f"- Main max tokens: `{args.max_tokens}`",
        f"- Token sensitivity budgets: `{', '.join(str(x) for x in args.token_sweep)}`",
        f"- Causal thesis metrics present: `{has_causal}`",
        "",
        "## Caveat Coverage Checklist",
        "",
        f"- Full balanced run configured: `{not args.smoke and args.max_examples is None}`",
        f"- Reasoning budget raised above 64: `{args.max_tokens > 64}`",
        f"- Exposure grid includes 0.0: `{0.0 in args.exposure_grid}`",
        f"- Adversarial tests scheduled: `{not args.skip_adversarial}`",
        f"- Multiple backbones scheduled: `{len(models) >= 3}`",
        f"- CCS/CHS/PCI/WCR/PDS collected: `{has_causal}`",
        f"- Repeated seeds scheduled: `{len(seeds) >= 3}`",
        "",
        "## Top Aggregate Rows",
        "",
    ]
    top = best_rows(rows)
    if top:
        columns = [
            "run_type",
            "variant",
            "model",
            "seed",
            "method",
            "dataset",
            "n",
            "accuracy",
            "ccs",
            "chs",
            "pci",
            "wcr",
            "pds",
            "avg_tokens_out",
        ]
        lines.append(markdown_table(top, columns))
    else:
        lines.append("No completed aggregate rows were found. This is expected only for `--dry_run`.")
    lines.extend([
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'a100_campaign_results.csv'}`",
        f"- `{out_dir / 'a100_campaign_manifest.json'}`",
        f"- `{summary}`",
    ])
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/a100_full_v7")
    parser.add_argument("--dataset", default="data/processed/real_llm_v4_test_balanced.jsonl")
    parser.add_argument("--models", default="mistral:latest,qwen3:32b,qwen2.5:32b,llama3.1:70b")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--provider", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--ollama_base_url", default="http://127.0.0.1:11434")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--token_sweep", default="64,512,1024")
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument("--timeout_seconds", type=float, default=300)
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--exposure_grid", default="0.0,0.2,0.35,0.5,1.0")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_tests", action="store_true")
    parser.add_argument("--skip_main", action="store_true")
    parser.add_argument("--skip_ablations", action="store_true")
    parser.add_argument("--skip_adversarial", action="store_true")
    parser.add_argument("--skip_token_sensitivity", action="store_true")
    args = parser.parse_args()

    args.models = split_csv(args.models)
    args.seeds = parse_ints(args.seeds)
    args.token_sweep = parse_ints(args.token_sweep)
    args.exposure_grid = parse_floats(args.exposure_grid)

    out_dir = Path(args.out_dir)
    config_dir = out_dir / "configs"
    manifest: list[dict[str, Any]] = []

    if args.smoke and args.provider != "mock":
        raise ValueError("--smoke must use --provider mock so local validation does not call external models")

    if not args.skip_tests:
        if args.dry_run:
            print("DRY-RUN pytest -q")
        else:
            subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)

    primary_model = args.models[0]
    primary_seed = args.seeds[0]

    if not args.skip_main:
        for model in args.models:
            for seed in args.seeds:
                safe = safe_model_name(model)
                cfg = base_config(
                    args,
                    model=model,
                    seed=seed,
                    methods=ALL_METHODS,
                    output_dir=out_dir / "main" / safe / f"seed_{seed}",
                    max_tokens=args.max_tokens,
                )
                run_config(
                    args,
                    cfg,
                    config_dir / "main" / safe / f"seed_{seed}.yaml",
                    manifest,
                    run_type="main",
                    model=model,
                    seed=seed,
                    variant="all_methods",
                )

    if not args.skip_ablations:
        for seed in args.seeds:
            for variant, methods, params in ablation_specs(args):
                cfg = base_config(
                    args,
                    model=primary_model,
                    seed=seed,
                    methods=methods,
                    output_dir=out_dir / "ablations" / safe_model_name(primary_model) / f"seed_{seed}" / variant,
                    max_tokens=args.max_tokens,
                    params_update=params,
                )
                run_config(
                    args,
                    cfg,
                    config_dir / "ablations" / safe_model_name(primary_model) / f"seed_{seed}" / f"{variant}.yaml",
                    manifest,
                    run_type="ablation",
                    model=primary_model,
                    seed=seed,
                    variant=variant,
                )

    if not args.skip_token_sensitivity:
        for tokens in args.token_sweep:
            cfg = base_config(
                args,
                model=primary_model,
                seed=primary_seed,
                methods=TOKEN_SENSITIVITY_METHODS,
                output_dir=out_dir / "token_sensitivity" / safe_model_name(primary_model) / f"tokens_{tokens}",
                max_tokens=tokens,
            )
            run_config(
                args,
                cfg,
                config_dir / "token_sensitivity" / safe_model_name(primary_model) / f"tokens_{tokens}.yaml",
                manifest,
                run_type="token_sensitivity",
                model=primary_model,
                seed=primary_seed,
                variant=f"max_tokens_{tokens}",
            )

    if not args.skip_adversarial:
        for model in args.models:
            for seed in args.seeds:
                safe = safe_model_name(model)
                cfg = base_config(
                    args,
                    model=model,
                    seed=seed,
                    methods=ADVERSARIAL_METHODS,
                    output_dir=out_dir / "adversarial" / safe / f"seed_{seed}",
                    max_tokens=args.max_tokens,
                )
                cfg["adversarial"] = {"num_adversaries": 2, "wrong_confidence": 0.95}
                run_config(
                    args,
                    cfg,
                    config_dir / "adversarial" / safe / f"seed_{seed}.yaml",
                    manifest,
                    run_type="adversarial",
                    model=model,
                    seed=seed,
                    variant="two_confident_wrong_agents",
                    adversarial=True,
                )

    rows, manifest = collect_results(out_dir, manifest)
    summary = write_summary(args, out_dir, rows, manifest)
    print(summary)
    print("well done, everything is ok!")


if __name__ == "__main__":
    main()

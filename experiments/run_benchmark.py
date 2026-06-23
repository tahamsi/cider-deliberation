#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base import BaseAgent
from agents.llm_agent import LLMAgent
from agents.mock_agent import MockAgent
from datasets.loaders import load_from_config, sample_rows
from datasets.schemas import TaskExample
from methods import METHODS
from metrics.accuracy import accuracy, is_correct
from metrics.agreement import agreement_rate
from metrics.calibration import brier_score, expected_calibration_error
from metrics.causal_influence import aggregate_causal_thesis_metrics, causal_thesis_metrics, exposure_density, mean_exposure_load
from metrics.cost import total_cost
from metrics.dissent import dissent_rate
from metrics.persuasion import answer_switch_rate


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_agents(config: dict[str, Any], seed: int, n: int) -> list[BaseAgent]:
    agent_type = config.get("type", "mock")
    cls = MockAgent if agent_type == "mock" else LLMAgent

    control_keys = {
        "type",
        "model_name",
        "model_names",
        "personas",
        "agent_temperatures",
        "agent_configs",
    }
    shared = {key: value for key, value in config.items() if key not in control_keys}
    personas = config.get("personas") or [
        "concise_solver",
        "skeptical_solver",
        "step_by_step_solver",
        "counterexample_seeker",
        "domain_expert",
        "adversarial_reviewer",
    ]
    temperatures = config.get("agent_temperatures") or []
    model_names = config.get("model_names") or [
        config.get("model_name", agent_type)
    ]
    agent_configs = config.get("agent_configs") or []

    agents = []
    for i in range(n):
        agent_extra = dict(shared)
        if agent_configs:
            specific = agent_configs[i % len(agent_configs)]
            if not isinstance(specific, dict):
                raise TypeError("Each agent_configs entry must be a mapping")
            agent_extra.update(specific)

        agent_extra.setdefault("persona", personas[i % len(personas)])
        if temperatures and "temperature" not in agent_extra:
            agent_extra["temperature"] = float(
                temperatures[i % len(temperatures)]
            )

        model_name = str(
            agent_extra.pop("model_name", model_names[i % len(model_names)])
        )
        agents.append(
            cls(
                name=f"agent_{i}",
                model_name=model_name,
                seed=seed + i,
                **agent_extra,
            )
        )
    return agents

def record_for(task: TaskExample, method_name: str, result: Any) -> dict[str, Any]:
    correct = is_correct(result.prediction, task.answer, result.prediction_index, task.answer_index)
    costs = total_cost(result.transcript)
    causal_metrics = causal_thesis_metrics(result.transcript, task.answer, task.answer_index)
    return {
        "dataset": task.dataset,
        "id": task.id,
        "method": method_name,
        "prediction": result.prediction,
        "prediction_index": result.prediction_index,
        "answer": task.answer,
        "answer_index": task.answer_index,
        "correct": correct,
        "confidence": result.confidence,
        "transcript": result.transcript,
        "exposure_matrix": result.exposure_matrix.tolist(),
        "cost": costs,
        "agreement_rate": agreement_rate(result.transcript),
        "dissent_rate": dissent_rate(result.transcript),
        "answer_switch_rate": answer_switch_rate(result.transcript),
        "exposure_density": exposure_density(result.exposure_matrix),
        "mean_exposure_load": mean_exposure_load(result.exposure_matrix),
        **causal_metrics,
        "metadata": result.metadata,
    }


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(rec["method"], rec["dataset"])].append(rec)
    rows = []
    for (method, dataset), items in sorted(grouped.items()):
        causal = aggregate_causal_thesis_metrics(items)
        rows.append({
            "method": method,
            "dataset": dataset,
            "n": len(items),
            "accuracy": accuracy(items),
            "brier_score": brier_score(items),
            "ece": expected_calibration_error(items),
            "avg_cost_usd": sum(i["cost"]["cost_usd"] for i in items) / len(items),
            "avg_tokens_in": sum(i["cost"]["tokens_in"] for i in items) / len(items),
            "avg_tokens_out": sum(i["cost"]["tokens_out"] for i in items) / len(items),
            "avg_agreement_rate": sum(i["agreement_rate"] for i in items) / len(items),
            "avg_dissent_rate": sum(i["dissent_rate"] for i in items) / len(items),
            "avg_exposure_density": sum(i["exposure_density"] for i in items) / len(items),
            "avg_mean_exposure_load": sum(i["mean_exposure_load"] for i in items) / len(items),
            "ccs": causal["ccs"],
            "chs": causal["chs"],
            "pci": causal["pci"],
            "wcr": causal["wcr"],
            "pds": causal["pds"],
            "ccs_corrections": causal.get("ccs_corrections", 0.0),
            "ccs_opportunities": causal.get("ccs_opportunities", 0.0),
            "chs_harms": causal.get("chs_harms", 0.0),
            "chs_opportunities": causal.get("chs_opportunities", 0.0),
            "pci_contaminated_switches": causal.get("pci_contaminated_switches", 0.0),
            "pci_exposed_switches": causal.get("pci_exposed_switches", 0.0),
        })
    return rows


def run(config: dict[str, Any], max_examples: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config.get("seed", 0))
    out_dir = Path(config.get("output_dir", "outputs/default"))
    method_params = config.get("method_params", {})
    num_agents = max(int(method_params.get("num_agents", 1)), int(method_params.get("num_samples", 1)))
    tasks = sample_rows(load_from_config(config["datasets"], seed), max_examples, seed)
    agents = build_agents(config.get("agent", {"type": "mock"}), seed, num_agents)
    records = []
    for method_name in config.get("methods", []):
        if method_name not in METHODS:
            raise ValueError(f"Unknown method {method_name}. Available: {sorted(METHODS)}")
        method = METHODS[method_name](agents=agents, seed=seed, **method_params)
        method_records = []
        for task in tasks:
            try:
                rec = record_for(task, method_name, method.run(task))
            except Exception as exc:
                raise RuntimeError(f"Method {method_name} failed on {task.dataset}/{task.id}: {exc}") from exc
            method_records.append(rec)
        append_jsonl(out_dir / "raw" / f"{method_name}.jsonl", method_records)
        records.extend(method_records)
    rows = aggregate(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    append_jsonl(out_dir / "predictions.jsonl", records)
    write_json(out_dir / "run_metadata.json", {
        "seed": seed,
        "git_commit": git_commit(),
        "model_config": config.get("agent", {}),
        "methods": config.get("methods", []),
        "method_params": method_params,
        "num_examples": len(tasks),
    })
    return records, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _, rows = run(config, args.max_examples)
    print(f"Wrote {len(rows)} aggregate rows to {config.get('output_dir', 'outputs/default')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from metrics.accuracy import is_correct
from methods.cider_v2 import CiderAdaptiveGated


def iter_jsonl(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield path, line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def collect(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, int]]]:
    feature_names = list(CiderAdaptiveGated.FEATURE_NAMES)
    xs: list[list[float]] = []
    ys: list[float] = []
    groups: list[tuple[int, int]] = []
    task_index = 0

    for _, _, record in iter_jsonl(paths):
        metadata = record.get("metadata") or {}
        candidates = metadata.get("v2_candidate_features") or []
        if not candidates:
            continue
        start = len(xs)
        for candidate in candidates:
            features = candidate.get("features") or {}
            xs.append([float(features.get(name, 0.0)) for name in feature_names])
            correct = is_correct(
                str(candidate.get("answer", "")),
                str(record.get("answer", "")),
                candidate.get("answer_index"),
                record.get("answer_index"),
            )
            ys.append(1.0 if correct else 0.0)
        groups.append((start, len(xs)))
        task_index += 1

    if not xs:
        raise SystemExit("No CIDeR-v2 candidate features found in the supplied JSONL files")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), feature_names, groups


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    z = (x - means) / scales

    weights = np.zeros(z.shape[1], dtype=float)
    intercept = 0.0
    positive = max(float(y.sum()), 1.0)
    negative = max(float((1.0 - y).sum()), 1.0)
    sample_weights = np.where(y > 0.5, 0.5 / positive, 0.5 / negative)
    sample_weights *= len(y)

    for epoch in range(epochs):
        logits = z @ weights + intercept
        probabilities = sigmoid(logits)
        residual = (probabilities - y) * sample_weights
        gradient_w = (z.T @ residual) / len(y) + l2 * weights
        gradient_b = float(residual.mean())
        step = learning_rate / math.sqrt(1.0 + epoch / 200.0)
        weights -= step * gradient_w
        intercept -= step * gradient_b

    return weights, intercept, means, scales


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    means: np.ndarray,
    scales: np.ndarray,
    groups: list[tuple[int, int]],
) -> dict[str, float]:
    probabilities = sigmoid(((x - means) / scales) @ weights + intercept)
    eps = 1e-9
    log_loss = float(
        -np.mean(y * np.log(probabilities + eps) + (1.0 - y) * np.log(1.0 - probabilities + eps))
    )
    classification_accuracy = float(np.mean((probabilities >= 0.5) == (y >= 0.5)))
    selections = []
    for start, end in groups:
        selected = start + int(np.argmax(probabilities[start:end]))
        selections.append(float(y[selected] >= 0.5))
    return {
        "candidate_log_loss": log_loss,
        "candidate_classification_accuracy": classification_accuracy,
        "task_selection_accuracy": float(np.mean(selections)) if selections else 0.0,
        "candidate_count": float(len(y)),
        "task_count": float(len(groups)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="CIDeR-v2 raw JSONL files from development runs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.02)
    args = parser.parse_args()

    paths = [Path(value) for value in args.inputs]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input files: {missing}")

    x, y, feature_names, groups = collect(paths)
    weights, intercept, means, scales = fit(
        x,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    metrics = evaluate(x, y, weights, intercept, means, scales, groups)

    payload: dict[str, Any] = {
        "kind": "logistic",
        "feature_names": feature_names,
        "weights": {name: float(value) for name, value in zip(feature_names, weights)},
        "intercept": float(intercept),
        "feature_means": {name: float(value) for name, value in zip(feature_names, means)},
        "feature_scales": {name: float(value) for name, value in zip(feature_names, scales)},
        "training": {
            "inputs": [str(path) for path in paths],
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            **metrics,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["training"], indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()

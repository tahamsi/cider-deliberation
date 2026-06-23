#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_files(paths: list[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for value in paths:
        path = Path(value)
        candidates = [path] if path.is_file() else sorted(
            path.rglob("cider_adaptive_gated.jsonl")
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def key(answer: Any, index: Any) -> tuple[str, int | None]:
    try:
        parsed_index = None if index is None else int(index)
    except (TypeError, ValueError):
        parsed_index = None
    return str(answer or "").strip(), parsed_index


def matches(candidate: tuple[str, int | None], truth: tuple[str, int | None]) -> bool:
    if candidate[1] is not None and truth[1] is not None:
        return candidate[1] == truth[1]
    return candidate[0].strip().lower() == truth[0].strip().lower()


def load_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def summarise(paths: list[str]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    scores: list[float] = []
    thresholds: list[float] = []
    weights: list[float] = []
    files = list(iter_files(paths))
    if not files:
        raise SystemExit("No cider_adaptive_gated.jsonl files found")

    for path in files:
        for row in load_rows(path):
            dataset = str(row.get("dataset", "unknown"))
            truth = key(row.get("answer"), row.get("answer_index"))
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            decisions = metadata.get("v2_gate_decisions") or []
            totals["tasks"] += 1
            by_dataset[dataset]["tasks"] += 1
            totals["final_correct"] += int(bool(row.get("correct")))
            by_dataset[dataset]["final_correct"] += int(bool(row.get("correct")))

            for decision in decisions:
                if not decision.get("switched"):
                    continue
                accepted = bool(decision.get("accepted"))
                initial = key(
                    decision.get("initial_answer"),
                    decision.get("initial_answer_index"),
                )
                proposed = key(
                    decision.get("proposed_answer"),
                    decision.get("proposed_answer_index"),
                )
                initial_correct = matches(initial, truth)
                proposed_correct = matches(proposed, truth)
                category = (
                    "beneficial"
                    if not initial_correct and proposed_correct
                    else "harmful"
                    if initial_correct and not proposed_correct
                    else "neutral"
                )
                outcome = "accepted" if accepted else "rejected"
                totals["switches"] += 1
                totals[f"{outcome}_{category}"] += 1
                by_dataset[dataset]["switches"] += 1
                by_dataset[dataset][f"{outcome}_{category}"] += 1
                reasons[str(decision.get("reason", "unknown"))] += 1
                if "acceptance_score" in decision:
                    scores.append(float(decision["acceptance_score"]))
                if "acceptance_threshold" in decision:
                    thresholds.append(float(decision["acceptance_threshold"]))
                if accepted:
                    weights.append(float(decision.get("acceptance_weight", 1.0)))

    accepted = sum(totals[f"accepted_{kind}"] for kind in ("beneficial", "harmful", "neutral"))
    rejected = sum(totals[f"rejected_{kind}"] for kind in ("beneficial", "harmful", "neutral"))
    return {
        "files": [str(path) for path in files],
        "tasks": totals["tasks"],
        "accuracy": totals["final_correct"] / max(totals["tasks"], 1),
        "switches": totals["switches"],
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / max(totals["switches"], 1),
        "accepted_beneficial": totals["accepted_beneficial"],
        "accepted_harmful": totals["accepted_harmful"],
        "accepted_neutral": totals["accepted_neutral"],
        "rejected_beneficial": totals["rejected_beneficial"],
        "rejected_harmful": totals["rejected_harmful"],
        "rejected_neutral": totals["rejected_neutral"],
        "beneficial_recall": totals["accepted_beneficial"]
        / max(totals["accepted_beneficial"] + totals["rejected_beneficial"], 1),
        "harmful_acceptance_rate": totals["accepted_harmful"]
        / max(totals["accepted_harmful"] + totals["rejected_harmful"], 1),
        "net_accepted_corrections": totals["accepted_beneficial"]
        - totals["accepted_harmful"],
        "mean_score": sum(scores) / max(len(scores), 1),
        "mean_threshold": sum(thresholds) / max(len(thresholds), 1),
        "mean_accepted_weight": sum(weights) / max(len(weights), 1),
        "reasons": dict(reasons.most_common()),
        "by_dataset": {name: dict(values) for name, values in sorted(by_dataset.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise CIDeR-v2 selective-gate decisions from raw JSONL files."
    )
    parser.add_argument("paths", nargs="+", help="Raw JSONL file or campaign directory")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    result = summarise(args.paths)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"files: {len(result['files'])}")
    print(f"tasks: {result['tasks']}")
    print(f"accuracy: {result['accuracy']:.4f}")
    print(f"switches: {result['switches']}")
    print(f"acceptance_rate: {result['acceptance_rate']:.4f}")
    print(f"beneficial_recall: {result['beneficial_recall']:.4f}")
    print(f"harmful_acceptance_rate: {result['harmful_acceptance_rate']:.4f}")
    print(f"net_accepted_corrections: {result['net_accepted_corrections']}")
    print(f"mean_acceptance_score: {result['mean_score']:.4f}")
    print(f"mean_acceptance_threshold: {result['mean_threshold']:.4f}")
    print(f"mean_accepted_weight: {result['mean_accepted_weight']:.4f}")
    print("reasons:")
    for reason, count in result["reasons"].items():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def balanced_subset(rows: list[dict], per_dataset: int, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    selected = []
    manifest = {}
    for dataset in sorted(groups):
        items = groups[dataset][:]
        rng.shuffle(items)
        take = min(per_dataset, len(items))
        selected.extend(items[:take])
        manifest[dataset] = {"available": len(items), "selected": take}
    rng.shuffle(selected)
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/real_llm_v4_test_balanced.jsonl")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mistral_per_dataset", type=int, default=20)
    parser.add_argument("--qwen_per_dataset", type=int, default=4)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    out_dir = Path(args.out_dir)
    mistral, mistral_manifest = balanced_subset(rows, args.mistral_per_dataset, args.seed)
    qwen, qwen_manifest = balanced_subset(rows, args.qwen_per_dataset, args.seed + 1)
    write_jsonl(mistral, out_dir / "real_llm_v6_mistral_160.jsonl")
    write_jsonl(qwen, out_dir / "real_llm_v6_qwen3_32.jsonl")
    manifest = {
        "source": args.input,
        "seed": args.seed,
        "mistral": {"rows": len(mistral), "per_dataset": args.mistral_per_dataset, "datasets": mistral_manifest},
        "qwen3": {"rows": len(qwen), "per_dataset": args.qwen_per_dataset, "datasets": qwen_manifest},
    }
    (out_dir / "real_llm_v6_subset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

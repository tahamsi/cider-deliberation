#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/all.jsonl")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_per_dataset", type=int, default=20)
    parser.add_argument("--test_per_dataset", type=int, default=100)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows_by_dataset[row["dataset"]].append(row)

    dev, test = [], []
    manifest = {}
    for dataset in sorted(rows_by_dataset):
        rows = list(rows_by_dataset[dataset])
        rng.shuffle(rows)
        dev_n = min(args.dev_per_dataset, max(0, len(rows) // 3))
        remaining = rows[dev_n:]
        test_n = min(args.test_per_dataset, len(remaining))
        dev.extend(rows[:dev_n])
        test.extend(remaining[:test_n])
        manifest[dataset] = {
            "available": len(rows),
            "dev": dev_n,
            "test": test_n,
            "requested_test": args.test_per_dataset,
            "note": "AIME has fewer than 100 available examples" if dataset == "aime2024" and test_n < args.test_per_dataset else "",
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("real_llm_v4_dev.jsonl", dev), ("real_llm_v4_test_balanced.jsonl", test)]:
        with (out_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "real_llm_v4_split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"dev": len(dev), "test": len(test), "manifest": manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

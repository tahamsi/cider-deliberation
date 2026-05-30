#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    shown = 0
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if not rec["correct"]:
                print(json.dumps({k: rec[k] for k in ["dataset", "id", "method", "prediction", "answer"]}, indent=2))
                shown += 1
                if shown >= args.limit:
                    break


if __name__ == "__main__":
    main()

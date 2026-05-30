#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.metrics)
    pivot = df.pivot(index="dataset", columns="method", values="accuracy")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.out)


if __name__ == "__main__":
    main()

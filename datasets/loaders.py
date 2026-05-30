from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from datasets.schemas import TaskExample


def load_jsonl(path: str | Path) -> list[TaskExample]:
    rows: list[TaskExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(TaskExample.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid row in {path}:{line_no}: {exc}") from exc
    return rows


def load_from_config(dataset_config: dict[str, Any], seed: int) -> list[TaskExample]:
    rows: list[TaskExample] = []
    if "inline" in dataset_config:
        rows.extend(TaskExample.model_validate(row) for row in dataset_config["inline"])
    for path in dataset_config.get("paths", []) or []:
        rows.extend(load_jsonl(path))
    if not rows:
        raise ValueError("No datasets configured. Provide datasets.inline or datasets.paths.")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def sample_rows(rows: Iterable[TaskExample], max_examples: int | None, seed: int) -> list[TaskExample]:
    materialized = list(rows)
    if max_examples is None or max_examples >= len(materialized):
        return materialized
    rng = random.Random(seed)
    return rng.sample(materialized, max_examples)

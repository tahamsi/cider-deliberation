from __future__ import annotations

from typing import Any

import numpy as np


def brier_score(records: list[dict[str, Any]]) -> float:
    vals = []
    for rec in records:
        conf = rec.get("confidence")
        if conf is not None:
            vals.append((float(conf) - float(bool(rec["correct"]))) ** 2)
    return float(np.mean(vals)) if vals else float("nan")


def expected_calibration_error(records: list[dict[str, Any]], bins: int = 10) -> float:
    usable = [(float(r["confidence"]), float(bool(r["correct"]))) for r in records if r.get("confidence") is not None]
    if not usable:
        return float("nan")
    ece = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1 / bins
        bucket = [(c, y) for c, y in usable if lo <= c < hi or (hi >= 1 and c == 1)]
        if bucket:
            conf = np.mean([c for c, _ in bucket])
            acc = np.mean([y for _, y in bucket])
            ece += len(bucket) / len(usable) * abs(conf - acc)
    return float(ece)

from __future__ import annotations

import numpy as np


def exposure_density(matrix: list[list[float]] | np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return 0.0
    denom = max(arr.shape[0] * arr.shape[1] - arr.shape[0], 1)
    return float((arr.sum() - np.trace(arr)) / denom)


def mean_exposure_load(matrix: list[list[float]] | np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    return float(arr.sum(axis=1).mean()) if arr.size else 0.0

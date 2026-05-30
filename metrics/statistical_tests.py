from __future__ import annotations

from typing import Sequence

from scipy import stats


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    if len(a) != len(b) or not a:
        raise ValueError("paired_ttest requires equal non-empty vectors")
    stat, pvalue = stats.ttest_rel(a, b)
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def bootstrap_mean_ci(values: Sequence[float], seed: int, n_boot: int = 1000, alpha: float = 0.05) -> dict[str, float]:
    import numpy as np

    if not values:
        raise ValueError("bootstrap_mean_ci requires non-empty values")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
    }

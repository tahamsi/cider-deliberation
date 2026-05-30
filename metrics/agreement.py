from __future__ import annotations

from collections import Counter


def agreement_rate(transcript: list[dict]) -> float:
    if not transcript:
        return 0.0
    counts = Counter(rec["answer"] for rec in transcript)
    return counts.most_common(1)[0][1] / len(transcript)

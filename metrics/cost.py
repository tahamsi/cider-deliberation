from __future__ import annotations


def total_cost(transcript: list[dict]) -> dict[str, float]:
    return {
        "tokens_in": float(sum(rec.get("tokens_in", 0) for rec in transcript)),
        "tokens_out": float(sum(rec.get("tokens_out", 0) for rec in transcript)),
        "cost_usd": float(sum(rec.get("cost_usd", 0.0) for rec in transcript)),
    }

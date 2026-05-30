from __future__ import annotations


def answer_switch_rate(transcript: list[dict]) -> float:
    by_agent: dict[str, list[str]] = {}
    for rec in transcript:
        by_agent.setdefault(rec["agent"], []).append(rec["answer"])
    switches = 0
    comparisons = 0
    for answers in by_agent.values():
        for prev, cur in zip(answers, answers[1:]):
            comparisons += 1
            switches += int(prev != cur)
    return switches / comparisons if comparisons else 0.0

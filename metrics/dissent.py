from __future__ import annotations


def dissent_rate(transcript: list[dict]) -> float:
    if len(transcript) <= 1:
        return 0.0
    answers = [rec["answer"] for rec in transcript]
    majority = max(set(answers), key=answers.count)
    return sum(ans != majority for ans in answers) / len(answers)

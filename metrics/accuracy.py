from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def extract_number(value: Any) -> float | None:
    text = str(value).replace(",", "")
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        text = boxed[-1]
    frac = re.findall(r"[-+]?\d+(?:\.\d+)?/\d+(?:\.\d+)?", text)
    if frac:
        try:
            return float(Fraction(frac[-1]))
        except Exception:
            pass
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return float(matches[-1]) if matches else None


def is_correct(prediction: str, answer: str, prediction_index: int | None = None, answer_index: int | None = None) -> bool:
    if answer_index is not None:
        label = chr(ord("a") + answer_index) if 0 <= answer_index < 26 else ""
        return prediction_index == answer_index or normalize_text(prediction) == label
    if normalize_text(prediction) == normalize_text(answer):
        return True
    p_num = extract_number(prediction)
    a_num = extract_number(answer)
    return p_num is not None and a_num is not None and abs(p_num - a_num) <= 1e-9


def accuracy(records: list[dict[str, Any]]) -> float:
    if not records:
        raise ValueError("accuracy requires at least one record")
    return sum(bool(r["correct"]) for r in records) / len(records)

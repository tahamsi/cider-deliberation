from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from metrics.accuracy import is_correct


def exposure_density(matrix: list[list[float]] | np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return 0.0
    denom = max(arr.shape[0] * arr.shape[1] - arr.shape[0], 1)
    return float((arr.sum() - np.trace(arr)) / denom)


def mean_exposure_load(matrix: list[list[float]] | np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    return float(arr.sum(axis=1).mean()) if arr.size else 0.0


def _answer_key(rec: dict[str, Any]) -> tuple[str, int | None]:
    return str(rec.get("answer", "")).strip(), rec.get("answer_index")


def _is_verifier_record(rec: dict[str, Any]) -> bool:
    metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    mode = str(metadata.get("mode", "")).lower()
    return "verify" in mode


def _visible_majority(rec: dict[str, Any], transcript: list[dict[str, Any]]) -> tuple[str, int | None] | None:
    visible = []
    for idx in rec.get("visible_prior_messages", []) or []:
        if isinstance(idx, int) and 0 <= idx < len(transcript):
            visible.append(_answer_key(transcript[idx]))
    if not visible:
        return None
    return Counter(visible).most_common(1)[0][0]


def _agent_transitions(transcript: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for rec in transcript:
        if _is_verifier_record(rec):
            continue
        by_agent.setdefault(str(rec.get("agent", "")), []).append(rec)
    transitions = []
    for records in by_agent.values():
        if not records:
            continue
        initial = records[0]
        final = records[-1]
        transitions.append((initial, final))
    return transitions


def causal_thesis_counts(
    transcript: list[dict[str, Any]],
    answer: str,
    answer_index: int | None,
    convergence_threshold: float = 0.75,
) -> dict[str, float]:
    """Operational counts for the paper's causal-deliberation thesis metrics.

    These are observable proxy metrics over saved transcripts. They do not claim
    randomized causal identification unless the run itself randomized exposure.
    The ratios are computed by `causal_thesis_rates`.
    """
    transitions = _agent_transitions(transcript)
    counts = {
        "ccs_corrections": 0.0,
        "ccs_opportunities": 0.0,
        "chs_harms": 0.0,
        "chs_opportunities": 0.0,
        "pci_contaminated_switches": 0.0,
        "pci_exposed_switches": 0.0,
        "pds_exposed_switches": 0.0,
        "pds_exposed_final_states": 0.0,
        "wcr_wrong_convergence": 0.0,
        "wcr_tasks": 1.0,
    }

    final_keys = []
    for initial, final in transitions:
        initial_correct = is_correct(initial.get("answer", ""), answer, initial.get("answer_index"), answer_index)
        final_correct = is_correct(final.get("answer", ""), answer, final.get("answer_index"), answer_index)
        switched = _answer_key(initial) != _answer_key(final)
        exposed = bool(final.get("visible_prior_messages"))
        if exposed:
            counts["pds_exposed_final_states"] += 1.0
            if not initial_correct:
                counts["ccs_opportunities"] += 1.0
            if initial_correct:
                counts["chs_opportunities"] += 1.0
        if exposed and switched:
            counts["pci_exposed_switches"] += 1.0
            counts["pds_exposed_switches"] += 1.0
            if (not initial_correct) and final_correct:
                counts["ccs_corrections"] += 1.0
            if initial_correct and (not final_correct):
                counts["chs_harms"] += 1.0
            visible_majority = _visible_majority(final, transcript)
            copied_visible_majority = visible_majority is not None and _answer_key(final) == visible_majority
            correctness_improved = (not initial_correct) and final_correct
            if copied_visible_majority and not correctness_improved:
                counts["pci_contaminated_switches"] += 1.0
        final_keys.append(_answer_key(final))

    if final_keys:
        modal_key, modal_count = Counter(final_keys).most_common(1)[0]
        modal_correct = is_correct(modal_key[0], answer, modal_key[1], answer_index)
        agreement = modal_count / len(final_keys)
        if agreement >= convergence_threshold and not modal_correct:
            counts["wcr_wrong_convergence"] = 1.0
    return counts


def _ratio(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def causal_thesis_rates(counts: dict[str, float]) -> dict[str, float]:
    return {
        "ccs": _ratio(counts.get("ccs_corrections", 0.0), counts.get("ccs_opportunities", 0.0)),
        "chs": _ratio(counts.get("chs_harms", 0.0), counts.get("chs_opportunities", 0.0)),
        "pci": _ratio(counts.get("pci_contaminated_switches", 0.0), counts.get("pci_exposed_switches", 0.0)),
        "wcr": _ratio(counts.get("wcr_wrong_convergence", 0.0), counts.get("wcr_tasks", 0.0)),
        "pds": _ratio(counts.get("pds_exposed_switches", 0.0), counts.get("pds_exposed_final_states", 0.0)),
    }


def causal_thesis_metrics(
    transcript: list[dict[str, Any]],
    answer: str,
    answer_index: int | None,
    convergence_threshold: float = 0.75,
) -> dict[str, float]:
    counts = causal_thesis_counts(transcript, answer, answer_index, convergence_threshold)
    return counts | causal_thesis_rates(counts)


def aggregate_causal_thesis_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for rec in records:
        metrics = rec
        if "ccs_corrections" not in metrics:
            metrics = causal_thesis_metrics(rec.get("transcript", []), rec.get("answer", ""), rec.get("answer_index"))
        for key, value in metrics.items():
            if key.startswith(("ccs_", "chs_", "pci_", "pds_", "wcr_")):
                totals[key] = totals.get(key, 0.0) + float(value)
    return causal_thesis_rates(totals) | totals

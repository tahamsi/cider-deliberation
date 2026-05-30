from __future__ import annotations

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult
from metrics.accuracy import extract_number, normalize_text


class CiderFull(BaseMethod):
    name = "cider_full"
    deviation_note = "proposed method: three-phase causal exposure control with independent priors, controlled dissent exposure, switch penalties, and exposure-adjusted aggregation"

    def run(self, task: TaskExample) -> MethodResult:
        rng = np.random.default_rng(self.seed)
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        max_exposure = float(self.params.get("max_exposure_probability", 0.2))
        max_visible = int(self.params.get("max_visible_messages", 3))
        transcript = []
        exposure = np.zeros((len(agents) * 2, len(agents) * 2), dtype=float)

        for agent in agents:
            response = agent.answer(task, [], mode="independent")
            transcript.append(self._response_record(agent, response, 0, []))

        independent = transcript[: len(agents)]
        for ai, agent in enumerate(agents):
            own = independent[ai]
            dissent = [i for i, rec in enumerate(independent) if rec.get("answer") != own.get("answer")]
            agreement = [i for i, rec in enumerate(independent) if rec.get("answer") == own.get("answer") and i != ai]
            rng.shuffle(dissent)
            rng.shuffle(agreement)
            candidates = dissent + agreement
            visible = [i for i in candidates if rng.random() < max_exposure][:max_visible]
            context = [transcript[i] for i in visible]
            response = agent.answer(task, context, mode="cider_final")
            row = len(transcript)
            exposure[row, visible] = 1.0
            transcript.append(self._response_record(agent, response, 1, visible))

        final = transcript[len(agents):]
        final_rows = range(len(agents), len(agents) * 2)
        exposure_load = [float(exposure[row].sum()) for row in final_rows]
        independent_counts: dict[tuple[str, int | None], int] = {}
        for rec in independent:
            key = (rec["answer"], rec.get("answer_index"))
            independent_counts[key] = independent_counts.get(key, 0) + 1
        final_weights = []
        diagnostics = []
        for i, rec in enumerate(final):
            indep = independent[i]
            key = (rec["answer"], rec.get("answer_index"))
            switched = key != (indep["answer"], indep.get("answer_index"))
            exposure_penalty = 1.0 + exposure_load[i]
            visible = [transcript[j] for j in rec.get("visible_prior_messages", []) if j < len(transcript)]
            visible_answers = [v.get("answer") for v in visible]
            visible_majority = max(set(visible_answers), key=visible_answers.count) if visible_answers else None
            quality = self._rationale_quality(rec)
            independent_quality = self._rationale_quality(indep)
            evidence_improvement = self._evidence_improvement(indep, rec)
            copied_visible_majority = bool(visible_majority and rec.get("answer") == visible_majority)
            unsupported_copy = switched and copied_visible_majority and evidence_improvement <= 0.0
            useful_correction = switched and evidence_improvement > 0.0 and quality >= independent_quality
            switch_penalty = 1.6 if unsupported_copy else (0.85 if useful_correction else (1.25 if switched else 1.0))
            independent_prior = 1.0 + independent_counts.get(key, 0) / max(len(agents), 1)
            dissent_bonus = 1.1 if independent_counts.get(key, 0) == 1 else 1.0
            correction_bonus = 1.25 if useful_correction else 1.0
            counterfactual_robustness = 0.6 if unsupported_copy else 1.0
            weight = (
                float(rec["confidence"])
                * independent_prior
                * quality
                * dissent_bonus
                * correction_bonus
                * counterfactual_robustness
                / (exposure_penalty * switch_penalty)
            )
            final_weights.append(weight)
            diagnostics.append({
                "agent": rec["agent"],
                "independent_answer": indep["answer"],
                "final_answer": rec["answer"],
                "switched_after_exposure": switched,
                "exposure_load": exposure_load[i],
                "rationale_quality": quality,
                "independent_rationale_quality": independent_quality,
                "evidence_improvement": evidence_improvement,
                "copied_visible_majority": copied_visible_majority,
                "unsupported_copy": unsupported_copy,
                "useful_correction": useful_correction,
                "independent_prior": independent_prior,
                "aggregation_weight": weight,
            })
        totals: dict[tuple[str, int | None], float] = {}
        component_scores: dict[str, dict[str, float]] = {}
        for rec in independent:
            key = (rec["answer"], rec.get("answer_index"))
            support = 1.5 * float(rec["confidence"]) * self._rationale_quality(rec)
            totals[key] = totals.get(key, 0.0) + support
            component_scores.setdefault(f"{key[0]}:{key[1]}", {})["independent"] = component_scores.get(f"{key[0]}:{key[1]}", {}).get("independent", 0.0) + support
        for rec, weight in zip(final, final_weights):
            key = (rec["answer"], rec.get("answer_index"))
            support = 1.0 * weight
            totals[key] = totals.get(key, 0.0) + support
            component_scores.setdefault(f"{key[0]}:{key[1]}", {})["final"] = component_scores.get(f"{key[0]}:{key[1]}", {}).get("final", 0.0) + support
        if not totals:
            raise RuntimeError("CIDeR produced no aggregate scores")
        key, total = max(totals.items(), key=lambda item: item[1])
        answer, answer_index = key
        agg_conf = total / (sum(max(v, 0.0) for v in totals.values()) or 1.0)
        metadata = self.metadata() | {
            "aggregation_weights": final_weights,
            "answer_component_scores": component_scores,
            "answer_total_scores": {f"{k[0]}:{k[1]}": v for k, v in totals.items()},
            "max_exposure_probability": max_exposure,
            "max_visible_messages": max_visible,
            "independent_vote_counts": {f"{k[0]}:{k[1]}": v for k, v in independent_counts.items()},
            "counterfactual_diagnostics": diagnostics,
        }
        return MethodResult(answer, answer_index, agg_conf, transcript, exposure, metadata)

    def _rationale_quality(self, rec: dict) -> float:
        rationale = str(rec.get("rationale", ""))
        length_bonus = min(len(rationale.split()) / 40.0, 1.0)
        specificity = 0.2 if any(ch.isdigit() for ch in rationale) else 0.0
        answer_present = 0.2 if str(rec.get("answer", "")).lower() in rationale.lower() else 0.0
        return max(0.25, min(1.25, 0.65 + length_bonus * 0.25 + specificity + answer_present))

    def _evidence_improvement(self, independent: dict, final: dict) -> float:
        before = str(independent.get("rationale", "")).lower()
        after = str(final.get("rationale", "")).lower()
        gain = 0.0
        correction_terms = [
            "because", "therefore", "error", "mistake", "correct", "recalculate",
            "actually", "evidence", "option", "given", "so", "since",
        ]
        if any(term in after for term in correction_terms):
            gain += 0.25
        if len(after.split()) > len(before.split()) + 5:
            gain += 0.2
        if any(ch.isdigit() for ch in after) and not any(ch.isdigit() for ch in before):
            gain += 0.15
        if str(final.get("answer", "")).lower() in after:
            gain += 0.1
        repeated = len(set(after.split()) & set(before.split())) / max(len(set(after.split())), 1)
        if repeated > 0.8 and len(after.split()) > 8:
            gain -= 0.2
        return max(-0.5, min(0.8, gain))


class CiderFullTuned(CiderFull):
    name = "cider_full_tuned"
    deviation_note = (
        "predeclared tuned CIDeR: same architecture as cider_full with fixed exposure=0.5 "
        "and max_visible_messages=2 selected from v2 pilot before v3 evaluation"
    )

    def __init__(self, *args, **kwargs):
        kwargs["max_exposure_probability"] = float(kwargs.get("tuned_max_exposure_probability", 0.5))
        kwargs["max_visible_messages"] = int(kwargs.get("tuned_max_visible_messages", 2))
        super().__init__(*args, **kwargs)


class CiderVerified(CiderFull):
    name = "cider_verified"
    deviation_note = (
        "V5 proposed method: CIDeR exposure control plus verifier-style aggregation. "
        "Scores candidates by independent support, answer validity, evidence improvement, anti-copy penalties, "
        "and an optional verifier pass; it does not use ground-truth labels."
    )

    def run(self, task: TaskExample) -> MethodResult:
        rng = np.random.default_rng(self.seed)
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        max_exposure = float(self.params.get("verified_max_exposure_probability", self.params.get("tuned_max_exposure_probability", 0.5)))
        max_visible = int(self.params.get("verified_max_visible_messages", self.params.get("tuned_max_visible_messages", 2)))
        use_verifier = bool(self.params.get("verified_use_verifier_agent", True))
        transcript = []
        extra_rows = 1 if use_verifier else 0
        exposure = np.zeros((len(agents) * 2 + extra_rows, len(agents) * 2 + extra_rows), dtype=float)

        for agent in agents:
            response = agent.answer(task, [], mode="independent")
            transcript.append(self._response_record(agent, response, 0, []))

        independent = transcript[: len(agents)]
        for ai, agent in enumerate(agents):
            own = independent[ai]
            dissent = [i for i, rec in enumerate(independent) if rec.get("answer") != own.get("answer")]
            agreement = [i for i, rec in enumerate(independent) if rec.get("answer") == own.get("answer") and i != ai]
            rng.shuffle(dissent)
            rng.shuffle(agreement)
            visible = [i for i in dissent + agreement if rng.random() < max_exposure][:max_visible]
            context = [transcript[i] for i in visible]
            response = agent.answer(task, context, mode="cider_final")
            row = len(transcript)
            exposure[row, visible] = 1.0
            transcript.append(self._response_record(agent, response, 1, visible))

        final = transcript[len(agents): len(agents) * 2]
        candidate_scores, diagnostics = self._verified_candidate_scores(task, independent, final, transcript, exposure, len(agents))

        verifier_record = None
        if use_verifier and candidate_scores:
            verifier_context = self._candidate_context(candidate_scores, diagnostics)
            verifier_response = agents[0].answer(task, verifier_context, mode="cider_verify")
            verifier_row = len(transcript)
            prior_rows = list(range(len(transcript)))
            exposure[verifier_row, prior_rows] = 1.0
            verifier_record = self._response_record(agents[0], verifier_response, 2, prior_rows)
            transcript.append(verifier_record)
            verifier_key = (verifier_record["answer"], verifier_record.get("answer_index"))
            if verifier_key in candidate_scores:
                candidate_scores[verifier_key]["verifier_bonus"] += float(self.params.get("verified_verifier_bonus", 1.25))
            else:
                candidate_scores[verifier_key] = {
                    "independent_support": 0.0,
                    "final_support": 0.0,
                    "validity": self._answer_validity(task, verifier_record),
                    "anti_copy_penalty": 1.0,
                    "verifier_bonus": float(self.params.get("verified_verifier_bonus", 1.0)),
                    "score": 0.0,
                }

        totals: dict[tuple[str, int | None], float] = {}
        component_scores: dict[str, dict[str, float]] = {}
        for key, parts in candidate_scores.items():
            total = (
                (2.0 * parts.get("independent_support", 0.0) + 0.85 * parts.get("final_support", 0.0))
                * max(parts.get("validity", 0.2), 0.2)
                * max(parts.get("anti_copy_penalty", 0.2), 0.2)
                * max(parts.get("verifier_bonus", 1.0), 0.2)
            )
            totals[key] = total
            component_scores[f"{key[0]}:{key[1]}"] = {
                "independent_support": parts.get("independent_support", 0.0),
                "final_support": parts.get("final_support", 0.0),
                "validity": parts.get("validity", 0.0),
                "anti_copy_penalty": parts.get("anti_copy_penalty", 0.0),
                "verifier_bonus": parts.get("verifier_bonus", 1.0),
                "total": total,
            }
        if not totals:
            raise RuntimeError("CIDeR verified produced no aggregate scores")
        key, total = max(totals.items(), key=lambda item: item[1])
        answer, answer_index = key
        confidence = total / (sum(max(v, 0.0) for v in totals.values()) or 1.0)
        metadata = self.metadata() | {
            "answer_component_scores": component_scores,
            "answer_total_scores": {f"{k[0]}:{k[1]}": v for k, v in totals.items()},
            "max_exposure_probability": max_exposure,
            "max_visible_messages": max_visible,
            "verified_use_verifier_agent": use_verifier,
            "verified_diagnostics": diagnostics,
            "verifier_answer": None if verifier_record is None else verifier_record.get("answer"),
            "verifier_answer_index": None if verifier_record is None else verifier_record.get("answer_index"),
        }
        return MethodResult(answer, answer_index, confidence, transcript, exposure, metadata)

    def _verified_candidate_scores(
        self,
        task: TaskExample,
        independent: list[dict],
        final: list[dict],
        transcript: list[dict],
        exposure: np.ndarray,
        final_offset: int,
    ) -> tuple[dict[tuple[str, int | None], dict[str, float]], list[dict]]:
        scores: dict[tuple[str, int | None], dict[str, float]] = {}
        diagnostics = []
        independent_counts: dict[tuple[str, int | None], int] = {}
        for rec in independent:
            key = (rec["answer"], rec.get("answer_index"))
            independent_counts[key] = independent_counts.get(key, 0) + 1

        for rec in independent:
            key = (rec["answer"], rec.get("answer_index"))
            parts = scores.setdefault(key, self._empty_verified_parts(task, rec))
            support = float(rec["confidence"]) * self._rationale_quality(rec) * self._answer_validity(task, rec)
            parts["independent_support"] += support

        for i, rec in enumerate(final):
            indep = independent[i]
            key = (rec["answer"], rec.get("answer_index"))
            parts = scores.setdefault(key, self._empty_verified_parts(task, rec))
            row = final_offset + i
            visible = [transcript[j] for j in rec.get("visible_prior_messages", []) if j < len(transcript)]
            visible_answers = [v.get("answer") for v in visible]
            visible_majority = max(set(visible_answers), key=visible_answers.count) if visible_answers else None
            switched = key != (indep["answer"], indep.get("answer_index"))
            evidence_gain = self._evidence_improvement(indep, rec)
            copied_visible_majority = bool(visible_majority and rec.get("answer") == visible_majority)
            unsupported_copy = switched and copied_visible_majority and evidence_gain <= 0.0
            useful_correction = switched and evidence_gain > 0.0 and self._answer_validity(task, rec) >= self._answer_validity(task, indep)
            exposure_load = float(exposure[row].sum())
            validity = self._answer_validity(task, rec)
            reliability = 1.0 + independent_counts.get(key, 0) / max(len(independent), 1)
            correction_bonus = 1.35 if useful_correction else 1.0
            copy_penalty = 0.35 if unsupported_copy else (0.75 if switched and copied_visible_majority else 1.0)
            exposure_penalty = 1.0 / (1.0 + 0.35 * exposure_load)
            support = float(rec["confidence"]) * self._rationale_quality(rec) * validity * reliability * correction_bonus * copy_penalty * exposure_penalty
            parts["final_support"] += support
            parts["validity"] = max(parts["validity"], validity)
            parts["anti_copy_penalty"] = min(parts["anti_copy_penalty"], copy_penalty)
            diagnostics.append({
                "agent": rec["agent"],
                "independent_answer": indep["answer"],
                "final_answer": rec["answer"],
                "switched_after_exposure": switched,
                "visible_majority": visible_majority,
                "copied_visible_majority": copied_visible_majority,
                "unsupported_copy": unsupported_copy,
                "useful_correction": useful_correction,
                "evidence_improvement": evidence_gain,
                "answer_validity": validity,
                "exposure_load": exposure_load,
                "verified_support": support,
            })
        return scores, diagnostics

    def _empty_verified_parts(self, task: TaskExample, rec: dict) -> dict[str, float]:
        return {
            "independent_support": 0.0,
            "final_support": 0.0,
            "validity": self._answer_validity(task, rec),
            "anti_copy_penalty": 1.0,
            "verifier_bonus": 1.0,
            "score": 0.0,
        }

    def _answer_validity(self, task: TaskExample, rec: dict) -> float:
        answer = str(rec.get("answer", "")).strip()
        if not answer or normalize_text(answer) == "unknown":
            return 0.1
        if task.choices:
            if rec.get("answer_index") is not None and 0 <= int(rec["answer_index"]) < len(task.choices):
                return 1.0
            labels = {chr(ord("a") + i) for i in range(len(task.choices))}
            if normalize_text(answer) in labels:
                return 0.95
            lowered = normalize_text(answer)
            if any(normalize_text(choice) in lowered for choice in task.choices):
                return 0.75
            return 0.2
        if extract_number(answer) is not None:
            return 1.0
        if len(answer) <= 64:
            return 0.75
        return 0.35

    def _candidate_context(self, candidate_scores: dict[tuple[str, int | None], dict[str, float]], diagnostics: list[dict]) -> list[dict]:
        context = []
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: 2.0 * item[1].get("independent_support", 0.0) + item[1].get("final_support", 0.0),
            reverse=True,
        )
        for rank, ((answer, answer_index), parts) in enumerate(ranked[:6], start=1):
            context.append({
                "agent": f"candidate_{rank}",
                "answer": answer,
                "answer_index": answer_index,
                "confidence": min(0.99, max(0.01, parts.get("validity", 0.5))),
                "rationale": (
                    f"independent_support={parts.get('independent_support', 0.0):.3f}; "
                    f"final_support={parts.get('final_support', 0.0):.3f}; "
                    f"validity={parts.get('validity', 0.0):.3f}; "
                    f"anti_copy_penalty={parts.get('anti_copy_penalty', 1.0):.3f}"
                ),
            })
        for diag in diagnostics[:4]:
            context.append({
                "agent": str(diag.get("agent", "diagnostic")),
                "answer": str(diag.get("final_answer", "")),
                "answer_index": None,
                "confidence": 0.5,
                "rationale": (
                    f"switched={diag.get('switched_after_exposure')}; "
                    f"unsupported_copy={diag.get('unsupported_copy')}; "
                    f"useful_correction={diag.get('useful_correction')}; "
                    f"evidence_improvement={diag.get('evidence_improvement')}"
                ),
            })
        return context


class CiderSOTA(CiderVerified):
    name = "cider_sota"
    deviation_note = (
        "V6 proposed method: CIDeR exposure control with reliability-weighted evidence aggregation, "
        "format validation, useful-correction reward, unsupported-copy suppression, and verifier tie-breaking."
    )

    def run(self, task: TaskExample) -> MethodResult:
        rng = np.random.default_rng(self.seed)
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        max_exposure = float(self.params.get("sota_max_exposure_probability", 0.35))
        max_visible = int(self.params.get("sota_max_visible_messages", 2))
        use_verifier = bool(self.params.get("sota_use_verifier_agent", True))
        tie_margin = float(self.params.get("sota_verifier_tie_margin", 1.18))
        transcript = []
        extra_rows = 1 if use_verifier else 0
        exposure = np.zeros((len(agents) * 2 + extra_rows, len(agents) * 2 + extra_rows), dtype=float)

        for agent in agents:
            response = agent.answer(task, [], mode="independent")
            transcript.append(self._response_record(agent, response, 0, []))

        independent = transcript[: len(agents)]
        for ai, agent in enumerate(agents):
            own = independent[ai]
            dissent = [i for i, rec in enumerate(independent) if rec.get("answer") != own.get("answer")]
            agreement = [i for i, rec in enumerate(independent) if rec.get("answer") == own.get("answer") and i != ai]
            rng.shuffle(dissent)
            rng.shuffle(agreement)
            visible = [i for i in dissent + agreement if rng.random() < max_exposure][:max_visible]
            response = agent.answer(task, [transcript[i] for i in visible], mode="cider_final")
            row = len(transcript)
            exposure[row, visible] = 1.0
            transcript.append(self._response_record(agent, response, 1, visible))

        final = transcript[len(agents): len(agents) * 2]
        scores, components, diagnostics = self._sota_scores(task, independent, final, transcript, exposure, len(agents))

        verifier_record = None
        verifier_used_as_tiebreak = False
        if use_verifier and scores:
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            top = ranked[0][1]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            near_tie = second > 0 and (top / second) <= tie_margin
            if near_tie:
                verifier_context = self._candidate_context(
                    {
                        key: {
                            "independent_support": components[f"{key[0]}:{key[1]}"].get("independent", 0.0),
                            "final_support": components[f"{key[0]}:{key[1]}"].get("final", 0.0),
                            "validity": components[f"{key[0]}:{key[1]}"].get("validity", 0.0),
                            "anti_copy_penalty": components[f"{key[0]}:{key[1]}"].get("anti_copy_penalty", 1.0),
                        }
                        for key in scores
                    },
                    diagnostics,
                )
                verifier_response = agents[0].answer(task, verifier_context, mode="cider_verify")
                verifier_row = len(transcript)
                prior_rows = list(range(len(transcript)))
                exposure[verifier_row, prior_rows] = 1.0
                verifier_record = self._response_record(agents[0], verifier_response, 2, prior_rows)
                transcript.append(verifier_record)
                v_key = (verifier_record["answer"], verifier_record.get("answer_index"))
                validity = self._answer_validity(task, verifier_record)
                if validity >= 0.75:
                    bonus = float(verifier_record["confidence"]) * validity * float(self.params.get("sota_verifier_bonus", 0.75))
                    scores[v_key] = scores.get(v_key, 0.0) + bonus
                    comp = components.setdefault(
                        f"{v_key[0]}:{v_key[1]}",
                        {"independent": 0.0, "final": 0.0, "stability": 0.0, "correction": 0.0, "validity": validity, "anti_copy_penalty": 1.0},
                    )
                    comp["verifier"] = comp.get("verifier", 0.0) + bonus
                    verifier_used_as_tiebreak = True

        if not scores:
            raise RuntimeError("CIDeR SOTA produced no aggregate scores")
        key, total = max(scores.items(), key=lambda item: item[1])
        confidence = total / (sum(max(v, 0.0) for v in scores.values()) or 1.0)
        metadata = self.metadata() | {
            "answer_component_scores": components,
            "answer_total_scores": {f"{k[0]}:{k[1]}": v for k, v in scores.items()},
            "max_exposure_probability": max_exposure,
            "max_visible_messages": max_visible,
            "sota_diagnostics": diagnostics,
            "verifier_answer": None if verifier_record is None else verifier_record.get("answer"),
            "verifier_answer_index": None if verifier_record is None else verifier_record.get("answer_index"),
            "verifier_used_as_tiebreak": verifier_used_as_tiebreak,
        }
        return MethodResult(key[0], key[1], confidence, transcript, exposure, metadata)

    def _sota_scores(
        self,
        task: TaskExample,
        independent: list[dict],
        final: list[dict],
        transcript: list[dict],
        exposure: np.ndarray,
        final_offset: int,
    ) -> tuple[dict[tuple[str, int | None], float], dict[str, dict[str, float]], list[dict]]:
        scores: dict[tuple[str, int | None], float] = {}
        components: dict[str, dict[str, float]] = {}
        diagnostics = []
        independent_counts: dict[tuple[str, int | None], int] = {}
        for rec in independent:
            key = (rec["answer"], rec.get("answer_index"))
            independent_counts[key] = independent_counts.get(key, 0) + 1

        def add(key: tuple[str, int | None], name: str, value: float, validity: float, anti_copy_penalty: float = 1.0) -> None:
            scores[key] = scores.get(key, 0.0) + value
            comp = components.setdefault(
                f"{key[0]}:{key[1]}",
                {"independent": 0.0, "final": 0.0, "stability": 0.0, "correction": 0.0, "validity": validity, "anti_copy_penalty": anti_copy_penalty},
            )
            comp[name] = comp.get(name, 0.0) + value
            comp["validity"] = max(comp.get("validity", 0.0), validity)
            comp["anti_copy_penalty"] = min(comp.get("anti_copy_penalty", 1.0), anti_copy_penalty)

        for rec in independent:
            key = (rec["answer"], rec.get("answer_index"))
            validity = self._answer_validity(task, rec)
            reliability = self._agent_reliability(task, rec)
            support = 2.35 * float(rec["confidence"]) * self._rationale_quality(rec) * validity * reliability
            if independent_counts[key] == 1 and validity >= 0.75:
                support *= 1.08
            add(key, "independent", support, validity)

        for i, rec in enumerate(final):
            indep = independent[i]
            key = (rec["answer"], rec.get("answer_index"))
            indep_key = (indep["answer"], indep.get("answer_index"))
            switched = key != indep_key
            row = final_offset + i
            visible = [transcript[j] for j in rec.get("visible_prior_messages", []) if j < len(transcript)]
            visible_answers = [v.get("answer") for v in visible]
            visible_majority = max(set(visible_answers), key=visible_answers.count) if visible_answers else None
            copied_visible_majority = bool(visible_majority and rec.get("answer") == visible_majority)
            evidence_gain = self._evidence_improvement(indep, rec)
            validity = self._answer_validity(task, rec)
            reliability = self._agent_reliability(task, rec)
            stable = not switched
            unsupported_copy = switched and copied_visible_majority and evidence_gain <= 0.05
            useful_correction = switched and evidence_gain >= 0.25 and validity >= self._answer_validity(task, indep)
            exposure_load = float(exposure[row].sum())
            anti_copy_penalty = 0.25 if unsupported_copy else (0.75 if switched and copied_visible_majority else 1.0)
            exposure_penalty = 1.0 / (1.0 + 0.22 * exposure_load)
            correction_bonus = 1.45 if useful_correction else 1.0
            stability_bonus = 1.18 if stable else 1.0
            prior = 1.0 + independent_counts.get(key, 0) / max(len(independent), 1)
            support = (
                0.95
                * float(rec["confidence"])
                * self._rationale_quality(rec)
                * validity
                * reliability
                * prior
                * correction_bonus
                * stability_bonus
                * anti_copy_penalty
                * exposure_penalty
            )
            add(key, "final", support, validity, anti_copy_penalty)
            if stable and validity >= 0.75:
                add(key, "stability", 0.18 * reliability, validity, anti_copy_penalty)
            if useful_correction:
                add(key, "correction", 0.35 * reliability * validity, validity, anti_copy_penalty)
            diagnostics.append({
                "agent": rec["agent"],
                "independent_answer": indep["answer"],
                "final_answer": rec["answer"],
                "switched_after_exposure": switched,
                "visible_majority": visible_majority,
                "copied_visible_majority": copied_visible_majority,
                "unsupported_copy": unsupported_copy,
                "useful_correction": useful_correction,
                "evidence_improvement": evidence_gain,
                "answer_validity": validity,
                "agent_reliability": reliability,
                "exposure_load": exposure_load,
                "support": support,
            })
        return scores, components, diagnostics

    def _agent_reliability(self, task: TaskExample, rec: dict) -> float:
        raw = self.params.get("sota_agent_reliability", {}) or {}
        if not isinstance(raw, dict):
            return 1.0
        agent = str(rec.get("agent", ""))
        persona = str((rec.get("metadata") or {}).get("persona", ""))
        keys = [
            f"{task.dataset}:{agent}",
            f"{task.dataset}:{persona}",
            agent,
            persona,
            task.dataset,
            "default",
        ]
        for key in keys:
            if key in raw:
                try:
                    return max(0.45, min(1.65, float(raw[key])))
                except Exception:
                    return 1.0
        return 1.0

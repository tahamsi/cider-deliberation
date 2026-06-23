from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import MethodResult
from methods.cider import CiderSOTA


AnswerKey = tuple[str, int | None]


class CiderAdaptiveGated(CiderSOTA):
    """CIDeR-v2: adaptive routing, strict correction gating and trajectory scoring.

    The method deliberately keeps the v1 helpers for answer validity, rationale
    quality and reliability, but changes when agents communicate and when a
    post-exposure answer is allowed to replace an independent answer.
    """

    name = "cider_adaptive_gated"
    deviation_note = (
        "CIDeR-v2: adaptive disagreement-triggered exposure, informative dissent "
        "routing, verifier-gated corrections, heterogeneous role/model support, "
        "trajectory-aware aggregation, optional learned calibration, and early stopping."
    )

    FEATURE_NAMES = (
        "bias",
        "independent_support",
        "final_support",
        "stable_support",
        "accepted_correction_support",
        "verifier_support",
        "diversity_support",
        "rejected_copy_risk",
        "exposure_cost",
    )

    DEFAULT_WEIGHTS = {
        "bias": 0.0,
        "independent_support": 2.20,
        "final_support": 0.70,
        "stable_support": 0.45,
        "accepted_correction_support": 1.35,
        "verifier_support": 0.90,
        "diversity_support": 0.35,
        "rejected_copy_risk": -1.50,
        "exposure_cost": -0.08,
    }

    DEFAULT_ROLE_WEIGHTS = {
        "concise_solver": 0.95,
        "skeptical_solver": 1.08,
        "step_by_step_solver": 1.05,
        "counterexample_seeker": 1.08,
        "domain_expert": 1.10,
        "adversarial_reviewer": 1.05,
        "general_solver": 1.00,
    }

    def run(self, task: TaskExample) -> MethodResult:
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        if not agents:
            raise RuntimeError("CIDeR-v2 requires at least one agent")

        use_verifier = bool(self.params.get("v2_use_verifier", True))
        matrix_size = (2 * len(agents)) + (1 if use_verifier else 0)
        exposure = np.zeros((matrix_size, matrix_size), dtype=float)
        transcript: list[dict[str, Any]] = []

        for agent in agents:
            response = agent.answer(task, [], mode="independent")
            rec = self._response_record(agent, response, 0, [])
            rec.setdefault("metadata", {})["model_name"] = agent.model_name
            transcript.append(rec)

        independent = transcript[: len(agents)]
        policy = self._deliberation_policy(independent)
        proposals: list[dict[str, Any]] = []

        if policy["deliberate"]:
            max_visible = max(1, int(self.params.get("v2_max_visible_messages", 2)))
            for agent_index, agent in enumerate(agents):
                visible = self._select_visible(agent_index, independent, max_visible)
                context = [transcript[index] for index in visible]
                response = agent.answer(task, context, mode="cider_final")
                row = len(transcript)
                exposure[row, visible] = 1.0
                rec = self._response_record(agent, response, 1, visible)
                rec.setdefault("metadata", {})["model_name"] = agent.model_name
                transcript.append(rec)
                proposals.append(rec)

        switch_proposals = [
            (independent[i], proposal)
            for i, proposal in enumerate(proposals)
            if self._key(independent[i]) != self._key(proposal)
        ]

        verifier_record: dict[str, Any] | None = None
        verifier_key: AnswerKey | None = None
        if use_verifier and switch_proposals:
            verifier_agent = self._verifier_agent(agents)
            verifier_context = self._verifier_context(independent, proposals)
            response = verifier_agent.answer(task, verifier_context, mode="cider_verify")
            row = len(transcript)
            prior_rows = list(range(len(transcript)))
            exposure[row, prior_rows] = 1.0
            verifier_record = self._response_record(verifier_agent, response, 2, prior_rows)
            verifier_record.setdefault("metadata", {})["model_name"] = verifier_agent.model_name
            transcript.append(verifier_record)
            verifier_key = self._key(verifier_record)

        gate_decisions: list[dict[str, Any]] = []
        gated_final: list[dict[str, Any]] = []
        if proposals:
            for index, proposal in enumerate(proposals):
                initial = independent[index]
                decision = self._gate_switch(task, initial, proposal, transcript, verifier_key)
                gate_decisions.append(decision)

                metadata = proposal.setdefault("metadata", {})
                metadata.update(
                    {
                        "proposed_answer": proposal.get("answer"),
                        "proposed_answer_index": proposal.get("answer_index"),
                        "switch_accepted": decision["accepted"],
                        "switch_gate_reason": decision["reason"],
                        "switch_gate": decision,
                    }
                )
                if decision["switched"] and not decision["accepted"]:
                    proposal["answer"] = initial.get("answer")
                    proposal["answer_index"] = initial.get("answer_index")
                    proposal["confidence"] = initial.get("confidence")
                gated_final.append(proposal)

        candidate_features = self._candidate_features(
            task=task,
            independent=independent,
            final=gated_final,
            proposals=proposals,
            decisions=gate_decisions,
            verifier_record=verifier_record,
        )
        scores, calibration = self._score_candidates(candidate_features)
        if not scores:
            raise RuntimeError("CIDeR-v2 produced no candidate scores")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        winner_key, winner_score = ranked[0]
        probabilities = self._softmax({key: score for key, score in ranked})
        confidence = probabilities.get(winner_key, 0.0)

        metadata = self.metadata() | {
            "v2_policy": policy,
            "v2_gate_decisions": gate_decisions,
            "v2_candidate_features": candidate_features,
            "v2_candidate_scores": {
                self._key_string(key): value for key, value in scores.items()
            },
            "v2_calibration": calibration,
            "v2_verifier_answer": None if verifier_record is None else verifier_record.get("answer"),
            "v2_verifier_answer_index": None
            if verifier_record is None
            else verifier_record.get("answer_index"),
            "v2_early_stopped": not policy["deliberate"],
            "v2_generation_count": len(transcript),
            "answer_total_scores": {
                self._key_string(key): value for key, value in scores.items()
            },
        }
        return MethodResult(
            winner_key[0],
            winner_key[1],
            confidence,
            transcript,
            exposure,
            metadata,
        )

    def _deliberation_policy(self, independent: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [self._key(rec) for rec in independent]
        counts = Counter(keys)
        n = max(len(keys), 1)
        probabilities = [count / n for count in counts.values()]
        entropy = 0.0
        if len(probabilities) > 1:
            entropy = -sum(p * math.log(p) for p in probabilities) / math.log(n)
        disagreement = 1.0 - (max(counts.values()) / n if counts else 1.0)
        mean_confidence = sum(float(rec.get("confidence", 0.0)) for rec in independent) / n
        confidence_uncertainty = 1.0 - mean_confidence
        rationale_diversity = self._mean_pairwise_diversity(
            [str(rec.get("rationale", "")) for rec in independent]
        )

        weights = {
            "entropy": 0.45,
            "disagreement": 0.35,
            "confidence_uncertainty": 0.10,
            "rationale_diversity": 0.10,
        }
        configured = self.params.get("v2_policy_weights") or {}
        if isinstance(configured, dict):
            for key in weights:
                if key in configured:
                    weights[key] = float(configured[key])

        score = (
            weights["entropy"] * entropy
            + weights["disagreement"] * disagreement
            + weights["confidence_uncertainty"] * confidence_uncertainty
            + weights["rationale_diversity"] * rationale_diversity
        )
        threshold = float(self.params.get("v2_deliberation_threshold", 0.20))
        low_confidence_unanimity = bool(
            self.params.get("v2_deliberate_on_low_confidence_unanimity", False)
        ) and mean_confidence < float(self.params.get("v2_low_confidence_threshold", 0.55))
        force = bool(self.params.get("v2_force_deliberation", False))
        deliberate = force or (
            score >= threshold and (len(counts) > 1 or low_confidence_unanimity)
        )
        return {
            "deliberate": deliberate,
            "score": score,
            "threshold": threshold,
            "answer_entropy": entropy,
            "disagreement": disagreement,
            "mean_confidence": mean_confidence,
            "confidence_uncertainty": confidence_uncertainty,
            "rationale_diversity": rationale_diversity,
            "distinct_answers": len(counts),
            "force_deliberation": force,
        }

    def _select_visible(
        self,
        target_index: int,
        independent: list[dict[str, Any]],
        max_visible: int,
    ) -> list[int]:
        target = independent[target_index]
        target_key = self._key(target)
        ranked: list[tuple[float, int]] = []
        for index, candidate in enumerate(independent):
            if index == target_index:
                continue
            dissent = 1.0 if self._key(candidate) != target_key else 0.0
            confidence = float(candidate.get("confidence", 0.0))
            quality = self._rationale_quality(candidate)
            novelty = self._text_diversity(
                str(target.get("rationale", "")), str(candidate.get("rationale", ""))
            )
            role_weight = self._role_weight(candidate)
            score = (
                1.50 * dissent
                + 0.45 * confidence
                + 0.35 * quality
                + 0.35 * novelty
                + 0.20 * role_weight
            )
            ranked.append((score, index))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [index for _, index in ranked[:max_visible]]

    def _verifier_agent(self, agents: list[Any]) -> Any:
        configured = self.params.get("v2_verifier_agent_index")
        if configured is not None:
            return agents[int(configured) % len(agents)]
        preferred = {"skeptical_solver", "adversarial_reviewer", "counterexample_seeker"}
        for agent in agents:
            if str(agent.config.get("persona", "")) in preferred:
                return agent
        return agents[-1]

    def _verifier_context(
        self,
        independent: list[dict[str, Any]],
        proposals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[AnswerKey, dict[str, float]] = {}
        for phase, records in (("independent", independent), ("proposal", proposals)):
            for rec in records:
                key = self._key(rec)
                item = grouped.setdefault(
                    key,
                    {"independent": 0.0, "proposal": 0.0, "confidence": 0.0},
                )
                item[phase] += 1.0
                item["confidence"] += float(rec.get("confidence", 0.0))
        context: list[dict[str, Any]] = []
        for rank, (key, values) in enumerate(
            sorted(
                grouped.items(),
                key=lambda item: (
                    -item[1]["independent"],
                    -item[1]["proposal"],
                    -item[1]["confidence"],
                    self._key_string(item[0]),
                ),
            ),
            start=1,
        ):
            context.append(
                {
                    "agent": f"candidate_{rank}",
                    "answer": key[0],
                    "answer_index": key[1],
                    "confidence": min(0.99, values["confidence"] / max(len(independent), 1)),
                    "rationale": (
                        f"independent_supporters={values['independent']:.0f}; "
                        f"post_exposure_supporters={values['proposal']:.0f}; "
                        "verify the substance rather than the vote count"
                    ),
                }
            )
        return context[:6]

    def _gate_switch(
        self,
        task: TaskExample,
        initial: dict[str, Any],
        proposal: dict[str, Any],
        transcript: list[dict[str, Any]],
        verifier_key: AnswerKey | None,
    ) -> dict[str, Any]:
        initial_key = self._key(initial)
        proposal_key = self._key(proposal)
        switched = initial_key != proposal_key
        if not switched:
            return {
                "switched": False,
                "accepted": True,
                "reason": "stable",
                "evidence_gain": 0.0,
                "validity_before": self._answer_validity(task, initial),
                "validity_after": self._answer_validity(task, proposal),
                "confidence_gain": 0.0,
                "copied_visible_majority": False,
                "unsupported_copy": False,
                "verifier_agrees": verifier_key == proposal_key if verifier_key else False,
                "deterministic_check": None,
            }

        evidence_gain = self._evidence_improvement(initial, proposal)
        validity_before = self._answer_validity(task, initial)
        validity_after = self._answer_validity(task, proposal)
        confidence_gain = float(proposal.get("confidence", 0.0)) - float(
            initial.get("confidence", 0.0)
        )
        visible_keys = [
            self._key(transcript[index])
            for index in proposal.get("visible_prior_messages", [])
            if isinstance(index, int) and 0 <= index < len(transcript)
        ]
        visible_majority = Counter(visible_keys).most_common(1)[0][0] if visible_keys else None
        copied_visible_majority = visible_majority == proposal_key if visible_majority else False
        min_evidence = float(self.params.get("v2_min_evidence_gain", 0.25))
        unsupported_copy = copied_visible_majority and evidence_gain < min_evidence
        verifier_agrees = verifier_key == proposal_key if verifier_key else False
        deterministic_check = self._deterministic_numeric_check(task, proposal)

        validity_ok = validity_after >= max(0.75, validity_before)
        evidence_ok = evidence_gain >= min_evidence
        confidence_ok = confidence_gain >= float(
            self.params.get("v2_min_confidence_gain", 0.05)
        )
        verifier_required = bool(self.params.get("v2_verifier_required_for_switch", True))
        allow_strong = bool(self.params.get("v2_allow_strong_switch_without_verifier", False))
        strong_threshold = float(self.params.get("v2_strong_evidence_gain", 0.55))

        if bool(self.params.get("v2_disable_gate", False)):
            accepted = validity_ok and not unsupported_copy
            reason = "gate_disabled"
        elif deterministic_check is True:
            accepted = validity_ok and not unsupported_copy
            reason = "deterministic_check"
        elif evidence_ok and verifier_agrees:
            accepted = validity_ok and not unsupported_copy
            reason = "evidence_and_verifier"
        elif (
            allow_strong
            and evidence_gain >= strong_threshold
            and confidence_ok
            and not verifier_required
        ):
            accepted = validity_ok and not unsupported_copy
            reason = "strong_evidence"
        else:
            accepted = False
            if unsupported_copy:
                reason = "unsupported_copy"
            elif not validity_ok:
                reason = "validity_regression"
            elif not evidence_ok:
                reason = "insufficient_evidence"
            elif verifier_required and not verifier_agrees:
                reason = "verifier_disagrees"
            else:
                reason = "gate_rejected"

        return {
            "switched": True,
            "accepted": accepted,
            "reason": reason,
            "evidence_gain": evidence_gain,
            "validity_before": validity_before,
            "validity_after": validity_after,
            "confidence_gain": confidence_gain,
            "copied_visible_majority": copied_visible_majority,
            "unsupported_copy": unsupported_copy,
            "verifier_agrees": verifier_agrees,
            "deterministic_check": deterministic_check,
            "initial_answer": initial_key[0],
            "initial_answer_index": initial_key[1],
            "proposed_answer": proposal_key[0],
            "proposed_answer_index": proposal_key[1],
        }

    def _candidate_features(
        self,
        *,
        task: TaskExample,
        independent: list[dict[str, Any]],
        final: list[dict[str, Any]],
        proposals: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        verifier_record: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        keys = {self._key(rec) for rec in independent}
        keys.update(self._key(rec) for rec in final)
        if verifier_record is not None:
            keys.add(self._key(verifier_record))

        decision_by_agent = {
            str(proposals[i].get("agent", i)): decisions[i]
            for i in range(min(len(proposals), len(decisions)))
        }
        verifier_key = self._key(verifier_record) if verifier_record is not None else None
        rows: list[dict[str, Any]] = []
        for key in sorted(keys, key=self._key_string):
            independent_support = 0.0
            final_support = 0.0
            stable_support = 0.0
            accepted_correction_support = 0.0
            supporter_identities: set[str] = set()
            exposure_cost = 0.0

            for rec in independent:
                if self._key(rec) != key:
                    continue
                role_weight = self._role_weight(rec)
                independent_support += (
                    float(rec.get("confidence", 0.0))
                    * self._rationale_quality(rec)
                    * self._answer_validity(task, rec)
                    * role_weight
                )
                supporter_identities.add(self._supporter_identity(rec))

            for index, rec in enumerate(final):
                if self._key(rec) != key:
                    continue
                initial = independent[index]
                role_weight = self._role_weight(rec)
                support = (
                    float(rec.get("confidence", 0.0))
                    * self._rationale_quality(rec)
                    * self._answer_validity(task, rec)
                    * role_weight
                )
                final_support += support
                exposure_cost += len(rec.get("visible_prior_messages", []) or [])
                supporter_identities.add(self._supporter_identity(rec))
                decision = decision_by_agent.get(str(rec.get("agent", index)), {})
                if self._key(initial) == key:
                    stable_support += support
                elif decision.get("accepted"):
                    accepted_correction_support += support

            rejected_copy_risk = 0.0
            for index, proposal in enumerate(proposals):
                decision = decisions[index] if index < len(decisions) else {}
                proposed_key = (
                    str(decision.get("proposed_answer", proposal.get("answer", ""))),
                    decision.get("proposed_answer_index", proposal.get("answer_index")),
                )
                if proposed_key == key and decision.get("unsupported_copy"):
                    rejected_copy_risk += 1.0

            features = {
                "bias": 1.0,
                "independent_support": independent_support,
                "final_support": final_support,
                "stable_support": stable_support,
                "accepted_correction_support": accepted_correction_support,
                "verifier_support": float(verifier_record.get("confidence", 0.0))
                if verifier_record is not None and verifier_key == key
                else 0.0,
                "diversity_support": len(supporter_identities) / max(len(independent), 1),
                "rejected_copy_risk": rejected_copy_risk,
                "exposure_cost": exposure_cost / max(len(independent), 1),
            }
            rows.append(
                {
                    "answer": key[0],
                    "answer_index": key[1],
                    "features": features,
                }
            )
        return rows

    def _score_candidates(
        self, candidate_features: list[dict[str, Any]]
    ) -> tuple[dict[AnswerKey, float], dict[str, Any]]:
        calibration = self._load_calibration()
        scores: dict[AnswerKey, float] = {}
        for row in candidate_features:
            key = (str(row["answer"]), row.get("answer_index"))
            features = row["features"]
            if calibration.get("kind") == "logistic":
                value = float(calibration.get("intercept", 0.0))
                means = calibration.get("feature_means", {})
                scales = calibration.get("feature_scales", {})
                weights = calibration.get("weights", {})
                for name in self.FEATURE_NAMES:
                    raw = float(features.get(name, 0.0))
                    scale = max(float(scales.get(name, 1.0)), 1e-8)
                    standardized = (raw - float(means.get(name, 0.0))) / scale
                    value += float(weights.get(name, 0.0)) * standardized
                score = self._sigmoid(value)
            else:
                weights = calibration["weights"]
                score = sum(
                    float(weights.get(name, 0.0)) * float(features.get(name, 0.0))
                    for name in self.FEATURE_NAMES
                )
            scores[key] = float(score)
        return scores, calibration

    def _load_calibration(self) -> dict[str, Any]:
        path_value = self.params.get("v2_weight_file")
        if path_value:
            path = Path(str(path_value))
            if not path.exists():
                raise FileNotFoundError(f"CIDeR-v2 weight file does not exist: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("feature_names") != list(self.FEATURE_NAMES):
                raise ValueError("CIDeR-v2 weight file has incompatible feature_names")
            return {
                "kind": "logistic",
                "source": str(path),
                "intercept": float(payload.get("intercept", 0.0)),
                "weights": payload.get("weights", {}),
                "feature_means": payload.get("feature_means", {}),
                "feature_scales": payload.get("feature_scales", {}),
            }

        weights = dict(self.DEFAULT_WEIGHTS)
        configured = self.params.get("v2_weights") or {}
        if isinstance(configured, dict):
            for name in self.FEATURE_NAMES:
                if name in configured:
                    weights[name] = float(configured[name])
        return {"kind": "linear", "source": "defaults_or_config", "weights": weights}

    def _role_weight(self, rec: dict[str, Any]) -> float:
        if bool(self.params.get("v2_disable_role_weight", False)):
            return 1.0
        metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        persona = str(metadata.get("persona", "general_solver"))
        configured = self.params.get("v2_role_weights") or {}
        if isinstance(configured, dict) and persona in configured:
            return max(0.5, min(1.5, float(configured[persona])))
        return self.DEFAULT_ROLE_WEIGHTS.get(persona, 1.0)

    def _supporter_identity(self, rec: dict[str, Any]) -> str:
        metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        return f"{metadata.get('model_name', 'unknown')}::{metadata.get('persona', 'general_solver')}"

    def _deterministic_numeric_check(
        self, task: TaskExample, rec: dict[str, Any]
    ) -> bool | None:
        if task.choices:
            return None
        answer = self._last_number(str(rec.get("answer", "")))
        if answer is None:
            return None
        expressions = re.findall(r"(?<![A-Za-z])[-+*/().\d\s]{3,}(?![A-Za-z])", task.question)
        for expression in expressions:
            expression = expression.strip().rstrip("?.= ")
            if not expression or not any(op in expression for op in "+-*/"):
                continue
            try:
                value = self._safe_eval(expression)
            except Exception:
                continue
            return math.isclose(answer, value, rel_tol=1e-9, abs_tol=1e-9)
        return None

    def _safe_eval(self, expression: str) -> float:
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def visit(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
                return float(operators[type(node.op)](visit(node.operand)))
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                left = visit(node.left)
                right = visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 8:
                    raise ValueError("exponent too large")
                return float(operators[type(node.op)](left, right))
            raise ValueError("unsupported arithmetic expression")

        tree = ast.parse(expression, mode="eval")
        return visit(tree)

    def _last_number(self, text: str) -> float | None:
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        return float(matches[-1]) if matches else None

    def _mean_pairwise_diversity(self, texts: list[str]) -> float:
        if len(texts) < 2:
            return 0.0
        values = [
            self._text_diversity(texts[i], texts[j])
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ]
        return sum(values) / len(values) if values else 0.0

    def _text_diversity(self, left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return 1.0 - (len(left_tokens & right_tokens) / len(union))

    def _key(self, rec: dict[str, Any] | None) -> AnswerKey:
        if rec is None:
            return "", None
        return str(rec.get("answer", "")).strip(), rec.get("answer_index")

    def _key_string(self, key: AnswerKey) -> str:
        return f"{key[0]}:{key[1]}"

    def _softmax(self, scores: dict[AnswerKey, float]) -> dict[AnswerKey, float]:
        if not scores:
            return {}
        maximum = max(scores.values())
        exponentials = {key: math.exp(value - maximum) for key, value in scores.items()}
        denominator = sum(exponentials.values()) or 1.0
        return {key: value / denominator for key, value in exponentials.items()}

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

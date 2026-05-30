from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from datasets.schemas import TaskExample
from methods.base_method import BaseMethod, MethodResult


OFFICIAL_ROOT = Path("external_baselines/official_sources")


def _source_metadata(name: str, commit: str, status: str) -> dict[str, Any]:
    return {"official_source": name, "official_commit": commit, "adapter_status": status}


class FreeMADOfficialAdapter(BaseMethod):
    name = "free_mad_official_adapter"
    deviation_note = (
        "official-source-informed adapter: uses Free-MAD trajectory scoring semantics "
        "(initial/keep/change with contributor normalization) from jonathansantilli/mad; "
        "uses this benchmark's Ollama agent interface instead of Free-MAD's CLI runtime"
    )

    def run(self, task: TaskExample) -> MethodResult:
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        rounds = int(self.params.get("rounds", 2))
        weights = [float(x) for x in self.params.get("free_mad_weights", [20.0, 25.0, 30.0, 20.0])]
        transcript = []
        exposure = np.zeros((len(agents) * rounds, len(agents) * rounds), dtype=float)

        for agent in agents:
            response = agent.answer(task, [], mode="independent")
            transcript.append(self._response_record(agent, response, 0, []))

        for round_id in range(1, rounds):
            prior = list(range(len(transcript)))
            for agent in agents:
                row = len(transcript)
                visible = prior[:]
                exposure[row, visible] = 1.0
                response = agent.answer(task, [transcript[i] for i in visible], mode="free_mad")
                transcript.append(self._response_record(agent, response, round_id, visible))

        scores, raw_scores, events = self._free_mad_scores(transcript, len(agents), rounds, weights)
        key, score = max(scores.items(), key=lambda item: item[1])
        denom = sum(max(v, 0.0) for v in scores.values()) or 1.0
        metadata = self.metadata() | _source_metadata(
            "jonathansantilli/mad Free-MAD",
            "1e8426a",
            "local adapter reusing official scoring semantics; not the unmodified CLI runtime",
        ) | {"normalized_scores": {f"{k[0]}:{k[1]}": v for k, v in scores.items()}, "raw_scores": raw_scores, "score_events": events}
        return MethodResult(key[0], key[1], score / denom, transcript, exposure, metadata)

    def _free_mad_scores(
        self,
        transcript: list[dict[str, Any]],
        num_agents: int,
        rounds: int,
        weights: list[float],
    ) -> tuple[dict[tuple[str, int | None], float], dict[str, float], list[dict[str, Any]]]:
        raw: dict[tuple[str, int | None], float] = {}
        contributors: dict[tuple[str, int | None], set[str]] = {}
        events = []

        def add(key: tuple[str, int | None], delta: float, contributor: str | None, round_id: int, agent: str, action: str) -> None:
            raw[key] = raw.get(key, 0.0) + delta
            if contributor:
                contributors.setdefault(key, set()).add(contributor)
            events.append({"round": round_id, "agent": agent, "action": action, "answer": key[0], "answer_index": key[1], "delta": delta})

        for rec in transcript[:num_agents]:
            key = (rec["answer"], rec.get("answer_index"))
            add(key, weights[0], rec["agent"], int(rec["round"]), rec["agent"], "initial")

        for r in range(1, rounds):
            start = r * num_agents
            prev = (r - 1) * num_agents
            for i, rec in enumerate(transcript[start:start + num_agents]):
                old = transcript[prev + i]
                old_key = (old["answer"], old.get("answer_index"))
                new_key = (rec["answer"], rec.get("answer_index"))
                decay = 1.0 / (r + 1)
                if new_key == old_key:
                    add(new_key, weights[3] * decay, rec["agent"], r, rec["agent"], "keep")
                else:
                    add(old_key, -weights[1] * decay, None, r, rec["agent"], "change_penalty")
                    add(new_key, weights[2] * decay, rec["agent"], r, rec["agent"], "change_bonus")

        normalized = {key: value / max(1, len(contributors.get(key, set()))) for key, value in raw.items()}
        raw_out = {f"{k[0]}:{k[1]}": v for k, v in raw.items()}
        return normalized, raw_out, events


class DAROfficialAdapter(BaseMethod):
    name = "dar_official_adapter"
    deviation_note = (
        "official-source-informed adapter: implements DAR's uncertainty/vote/critical-retention idea "
        "from DA2I2-SLM/DAR using this benchmark's Ollama agent interface"
    )

    def run(self, task: TaskExample) -> MethodResult:
        agents = self.agents[: int(self.params.get("num_agents", len(self.agents)))]
        transcript = []
        exposure = np.zeros((len(agents) * 2, len(agents) * 2), dtype=float)
        for agent in agents:
            response = agent.answer(task, [], mode="independent")
            transcript.append(self._response_record(agent, response, 0, []))

        majority_key = self._majority_key(transcript)
        retained = self._critical_retention(transcript, majority_key)
        retained_indices = [transcript.index(rec) for rec in retained]
        for agent in agents:
            row = len(transcript)
            visible = retained_indices[:]
            exposure[row, visible] = 1.0
            response = agent.answer(task, [transcript[i] for i in visible], mode="free_mad")
            transcript.append(self._response_record(agent, response, 1, visible))

        final = transcript[-len(agents):]
        weights = []
        for rec in final:
            key = (rec["answer"], rec.get("answer_index"))
            diverse_bonus = 1.15 if key != majority_key else 1.0
            weights.append(max(0.05, float(rec["confidence"])) * diverse_bonus)
        answer, answer_index, confidence = self._weighted_vote(final, weights)
        metadata = self.metadata() | _source_metadata(
            "DA2I2-SLM/DAR",
            "f3c6e9d",
            "local adapter implementing official DAR retention criteria; not the unmodified vLLM batch runner",
        ) | {"majority_key": f"{majority_key[0]}:{majority_key[1]}", "retained_indices": retained_indices, "weights": weights}
        return MethodResult(answer, answer_index, confidence, transcript, exposure, metadata)

    def _majority_key(self, transcript: list[dict[str, Any]]) -> tuple[str, int | None]:
        votes = [(rec["answer"], rec.get("answer_index")) for rec in transcript]
        return Counter(votes).most_common(1)[0][0]

    def _critical_retention(self, transcript: list[dict[str, Any]], majority_key: tuple[str, int | None]) -> list[dict[str, Any]]:
        dissent = [rec for rec in transcript if (rec["answer"], rec.get("answer_index")) != majority_key]
        majority = [rec for rec in transcript if (rec["answer"], rec.get("answer_index")) == majority_key]
        majority_sorted = sorted(majority, key=lambda rec: float(rec["confidence"]), reverse=True)
        retained = dissent + majority_sorted[:1]
        return retained or transcript

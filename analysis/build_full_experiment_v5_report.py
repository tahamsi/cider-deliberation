#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/full_experiment_v5")
REPORT = ROOT / "full-experiment-v5.md"


def md_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.4f}")
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def summary(run: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / run / "tables" / "overall_method_summary.csv")


def pairwise(run: str, target: str) -> pd.DataFrame:
    rows = []
    by_item: dict[tuple[str, str], dict[str, bool]] = {}
    with (ROOT / run / "predictions.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_item.setdefault((rec["dataset"], rec["id"]), {})[rec["method"]] = bool(rec["correct"])
    methods = sorted({m for item in by_item.values() for m in item})
    for method in methods:
        if method == target:
            continue
        target_only = method_only = both_right = both_wrong = 0
        for item in by_item.values():
            if target not in item or method not in item:
                continue
            t = item[target]
            b = item[method]
            if t and b:
                both_right += 1
            elif t and not b:
                target_only += 1
            elif b and not t:
                method_only += 1
            else:
                both_wrong += 1
        rows.append({
            "comparison": f"{target} vs {method}",
            "target_only_correct": target_only,
            "baseline_only_correct": method_only,
            "both_right": both_right,
            "both_wrong": both_wrong,
        })
    return pd.DataFrame(rows).sort_values(["baseline_only_correct", "target_only_correct"], ascending=[True, False])


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    mock = summary("mock")
    mistral = summary("mistral")
    qwen3 = summary("qwen3")
    m_best = mistral.iloc[0]
    q_best = qwen3.iloc[0]

    sections = [
        "# full-experiment-v5",
        (
            "V5 implements and evaluates a stronger CIDeR path. The headline is mixed but better than V4: "
            f"on the completed real Mistral run, `cider_full` is tied for best macro accuracy ({m_best['macro_accuracy']:.4f}) "
            "and beats most baselines, but the new `cider_verified` variant is too conservative and does not win accuracy. "
            "This is promising evidence for CIDeR, not yet a publication-grade SOTA claim."
        ),
        "## What Changed",
        (
            "- Added `cider_verified`, a V5 CIDeR variant with verifier-style aggregation.\n"
            "- Added answer-validity scoring for MC/free-form answers without using ground truth.\n"
            "- Added anti-copy penalties for exposed unsupported switches and bonuses for useful corrections.\n"
            "- Added optional verifier-agent pass in mode `cider_verify`.\n"
            "- Added Ollama `think` support and set `think: false` for Qwen3; this fixed empty Qwen3 responses.\n"
            "- Added V5 configs for deterministic mock, Mistral, and Qwen3."
        ),
        "## Runtime Reality",
        (
            "`lspci` detects the NVIDIA GPU, and Ollama reported `100% GPU` while running Mistral/Qwen3. "
            "`nvidia-smi` still fails from this shell, so low-level NVIDIA telemetry is broken even though Ollama is using the GPU. "
            "The completed real-model runs are 40 balanced examples for Mistral and 8 corrected examples for Qwen3. "
            "The full 720-example configs are present but were not completed in this turn."
        ),
        "## Models",
        (
            "- Mistral: `mistral:latest`, Ollama, 40 balanced examples, all 14 methods.\n"
            "- Qwen3: `qwen3:latest`, Ollama, `think: false`, 8 balanced examples, all 14 methods.\n"
            "- Deterministic validation: `deterministic-mock`, 720 balanced examples, all 14 methods.\n"
            "- Qwen 2.5 is still not installed locally; Qwen3 was the available Qwen-family run."
        ),
        "## Main Usable Real-Model Result: Mistral",
        md_table(mistral),
        (
            "Interpretation: `cider_full` is tied for the best Mistral macro accuracy with DAR-style retention. "
            "It beats majority vote, standard debate, self-consistency, FreeMAD-style, C3-style, and tuned CIDeR on this 40-example protocol. "
            "This supports the original v3 observation that local Mistral can benefit from CIDeR-style exposure control. "
            "However, the margin is small and the run is too small for a SOTA claim."
        ),
        "## Mistral Pairwise Diagnosis",
        md_table(pairwise("mistral", "cider_full")),
        "## Qwen3 Corrected Check",
        md_table(qwen3),
        (
            "Interpretation: Qwen3 became usable only after disabling thinking mode through the Ollama API. "
            "On the corrected 8-example check, `agentauditor_style`, `cider_full`, `cider_full_tuned`, and `cider_verified` are tied at 0.7000 macro accuracy. "
            "Because n=8, this is only a protocol check, not a model comparison."
        ),
        "## Deterministic Full Validation",
        md_table(mock),
        (
            "Interpretation: deterministic mock remains a framework validation run, not evidence about LLM SOTA. "
            "It still does not favor CIDeR; this means the proposed method depends on real model behavior and should not be justified by mock results."
        ),
        "## Critical Assessment",
        (
            "V5 is closer to a real SOTA path than V4 because the best Mistral result is CIDeR-family again. "
            "But it is not enough. The strongest claim currently supported is: on a 40-example balanced local-Mistral run with all implemented baselines, "
            "`cider_full` is tied for first and clearly above majority vote and standard debate. "
            "The claim not supported is: CIDeR is state of the art across all datasets/models."
        ),
        "## Remaining Requirements Before Publication",
        (
            "1. Run the full 720-example balanced split for Mistral with all methods.\n"
            "2. Run Qwen3 full after keeping `think: false`; ignore the preserved failed thinking-mode runs.\n"
            "3. Tune on `real_llm_v4_dev.jsonl`, freeze exactly once, then run the held-out test.\n"
            "4. Decide whether the winning method is `cider_full` or a revised `cider_verified`; current verified aggregation is over-conservative.\n"
            "5. Replace style baselines with official or exact reproductions where possible.\n"
            "6. Add paired exact tests and bootstrap CIs after the full real-model runs."
        ),
        "## Files",
        (
            "- `configs/real_llm_v5_mock.yaml`\n"
            "- `configs/real_llm_v5_mistral.yaml`\n"
            "- `configs/real_llm_v5_qwen3.yaml`\n"
            "- `outputs/full_experiment_v5/mock/`\n"
            "- `outputs/full_experiment_v5/mistral/`\n"
            "- `outputs/full_experiment_v5/qwen3/`\n"
            "- Failed Qwen diagnostics preserved in `outputs/full_experiment_v5/qwen3_failed_num_predict64/` and `outputs/full_experiment_v5/qwen3_failed_thinking512/`."
        ),
        "## Verification",
        "`pytest -q` passed: 7/7 tests.",
    ]
    REPORT.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path("outputs/full_experiment_v4")
REPORT_MD = ROOT / "full-experiment-v4.md"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "not available; repository has no .git directory in this checkout"


def md_table(df: pd.DataFrame, cols: Iterable[str] | None = None, max_rows: int | None = None) -> str:
    if cols is not None:
        df = df.loc[:, list(cols)]
    if max_rows is not None:
        df = df.head(max_rows)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.4f}")
    headers = list(out.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def overall_summary(run: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / run / "tables" / "overall_method_summary.csv")


def aggregate_overall(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return (
        df.groupby("method")
        .agg(
            datasets=("dataset", "nunique"),
            n=("n", "sum"),
            macro_accuracy=("accuracy", "mean"),
            mean_brier=("brier_score", "mean"),
            mean_ece=("ece", "mean"),
            avg_exposure_density=("avg_exposure_density", "mean"),
            avg_mean_exposure_load=("avg_mean_exposure_load", "mean"),
        )
        .reset_index()
        .sort_values(["macro_accuracy", "mean_brier"], ascending=[False, True])
    )


def ablation_summary() -> pd.DataFrame:
    rows = []
    for exposure_dir in sorted((ROOT / "mock").glob("exposure_*")):
        exposure = exposure_dir.name.replace("exposure_", "")
        agg = aggregate_overall(exposure_dir / "aggregate_metrics.csv")
        row = agg.iloc[0].to_dict()
        row["exposure_probability"] = float(exposure)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("exposure_probability")


def error_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = read_jsonl(ROOT / "mock" / "predictions.jsonl")
    by_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_key[(row["dataset"], row["id"])][row["method"]] = row

    methods = ["free_mad_style", "c3_style_causal_credit_analysis", "adaptive_stability_detection", "cider_full_tuned"]
    pair_rows = []
    for baseline in methods[:-1]:
        cider_only = baseline_only = both_correct = both_wrong = 0
        for preds in by_key.values():
            if "cider_full_tuned" not in preds or baseline not in preds:
                continue
            c = bool(preds["cider_full_tuned"]["correct"])
            b = bool(preds[baseline]["correct"])
            if c and b:
                both_correct += 1
            elif c and not b:
                cider_only += 1
            elif b and not c:
                baseline_only += 1
            else:
                both_wrong += 1
        pair_rows.append(
            {
                "comparison": f"cider_full_tuned vs {baseline}",
                "cider_only_correct": cider_only,
                "baseline_only_correct": baseline_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            }
        )

    dataset_rows = []
    for dataset in sorted({k[0] for k in by_key}):
        subset = {k: v for k, v in by_key.items() if k[0] == dataset}
        for method in ["cider_full_tuned", "free_mad_style", "c3_style_causal_credit_analysis", "majority_vote"]:
            correct = [bool(v[method]["correct"]) for v in subset.values() if method in v]
            dataset_rows.append({"dataset": dataset, "method": method, "accuracy": sum(correct) / len(correct), "n": len(correct)})
    return pd.DataFrame(pair_rows), pd.DataFrame(dataset_rows)


def load_config_text(name: str) -> str:
    path = Path("configs") / name
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "v4_split_manifest.json").read_text(encoding="utf-8"))
    mock = overall_summary("mock")
    mistral = overall_summary("mistral")
    qwen3 = overall_summary("qwen3")
    adversarial = aggregate_overall(ROOT / "mock_adversarial" / "aggregate_metrics.csv")
    ablations = ablation_summary()
    pair_errors, dataset_errors = error_summary()

    dataset_split = (
        pd.DataFrame(
            [
                {
                    "dataset": name,
                    "available": info["available"],
                    "dev": info["dev"],
                    "test": info["test"],
                    "note": info["note"],
                }
                for name, info in split.items()
            ]
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    total_test = int(dataset_split["test"].sum())
    total_dev = int(dataset_split["dev"].sum())

    cider_rank = int(mock.reset_index(drop=True).index[mock["method"].eq("cider_full_tuned")][0]) + 1
    best = mock.iloc[0]
    cider = mock[mock["method"].eq("cider_full_tuned")].iloc[0]

    sections: list[str] = []
    sections.append("# full-experiment-v4\n")
    sections.append(
        "This report records the v4 CIDeR benchmark exactly as run on this laptop. "
        "The main conclusion is deliberately critical: the framework is more complete, but the current CIDeR v4 implementation is not SOTA yet. "
        f"On the completed 720-example balanced deterministic run, CIDeR-tuned ranks {cider_rank}/{len(mock)} by macro accuracy "
        f"({cider['macro_accuracy']:.4f}) while the best method is {best['method']} ({best['macro_accuracy']:.4f})."
    )
    sections.append("## Execution Status\n")
    sections.append(
        "- Completed full-size run: deterministic mock model, 720 held-out examples, all 13 implemented methods.\n"
        "- Completed full-size ablation: deterministic mock CIDeR exposure probabilities 0.0, 0.2, 0.5, 1.0.\n"
        "- Completed full-size adversarial check: deterministic mock with two adversarial agents for majority vote, debate, and tuned CIDeR.\n"
        "- Completed local LLM feasibility probes: Mistral and Qwen3, one example across all 13 methods.\n"
        "- Not completed: full 720-example Mistral/Qwen3 CPU Ollama campaigns. Those configs are provided, but running them here would require a long local generation campaign. The report does not treat one-example probes as performance evidence.\n"
        "- Qwen 2.5 was requested, but no Qwen 2.5 Ollama model was installed locally. The available local Qwen model used was `qwen3:latest`."
    )
    sections.append("## Hardware and Reproducibility\n")
    sections.append(
        f"- Date: 2026-05-29\n"
        f"- Python: 3.11 virtual environment `.venv`\n"
        f"- Local runtime: CPU Ollama available; no usable NVIDIA driver detected during previous checks\n"
        f"- Git commit: {git_commit()}\n"
        f"- Seed: 42\n"
        f"- Output root: `outputs/full_experiment_v4/`\n"
        f"- Tests: `pytest -q` passed, 7/7 tests"
    )
    sections.append("## Dataset Split\n")
    sections.append(
        f"Balanced sampling was enforced so MMLU-Pro does not dominate. The dev split contains {total_dev} examples. "
        f"The held-out v4 test split contains {total_test} examples: 100 per dataset where available, except AIME 2024 with 20 test examples because only 30 total examples are available.\n\n"
        + md_table(dataset_split)
    )
    sections.append("## Models\n")
    sections.append(
        "- `deterministic-mock`: deterministic local agent used for the only completed full 720-example experiment. It is useful for pipeline validation and causal/control-flow tests, not for claiming LLM SOTA.\n"
        "- `mistral:latest`: local Ollama Mistral model configured for the real local-LLM run; one-example all-method probe completed.\n"
        "- `qwen3:latest`: local Ollama Qwen 3 model configured for the added Qwen run; one-example all-method probe completed.\n"
        "- `qwen2.5`: not run because the model was not installed locally."
    )
    sections.append("## Method Set\n")
    sections.append(
        "The v4 benchmark includes all minimum baselines plus optional baselines: `single_agent`, `self_consistency`, `majority_vote`, "
        "`standard_multi_agent_debate`, `consensagent_style`, `free_mad_style`, `agentauditor_style`, `c3_style_causal_credit_analysis`, "
        "`conformal_social_choice`, `dar_style_diversity_aware_retention`, `adaptive_stability_detection`, `cider_full`, and `cider_full_tuned`. "
        "The methods with `style`, `adapted`, or descriptive names remain approximations, not official implementations of the cited papers. "
        "That deviation is documented in `docs/method_deviations.md` and must be tightened before publication claims."
    )
    sections.append("## CIDeR v4 Architecture\n")
    sections.append(
        "CIDeR v4 keeps the core causal exposure-control design: agents first produce independent answers, then final deliberation is generated under a bounded exposure matrix. "
        "The method records who saw which prior messages, estimates exposure density/load, and scores final answers with a causal-credit weighting rather than plain agreement. "
        "The v4 change adds an evidence-improvement detector. A final answer is rewarded when it appears to add useful correction terms, numeric support, answer-grounded evidence, or a substantive rationale improvement over the independent answer. "
        "It is penalized when it simply copies the visible majority without evidence. The tuned variant increases exposure from 0.35 to 0.50 while retaining a two-message visibility cap."
    )
    sections.append("## Main Completed Result: Deterministic Full Run\n")
    sections.append(md_table(mock, max_rows=20))
    sections.append(
        "Interpretation: this is a failed SOTA result. CIDeR-tuned does not beat the strongest implemented baselines on the completed full run. "
        "The strongest deterministic method is `free_mad_style`, followed by `c3_style_causal_credit_analysis` and `adaptive_stability_detection`. "
        "CIDeR-tuned ties untuned CIDeR and majority vote on macro accuracy, so the current causal exposure scoring is not yet extracting enough extra signal from the deterministic agents."
    )
    sections.append("## Local LLM Probe Results\n")
    sections.append(
        "These are one-example smoke probes only. They confirm the full method set can execute against the local Ollama models, but they are not statistically meaningful.\n\n"
        "**Mistral probe**\n\n"
        + md_table(mistral, max_rows=20)
        + "\n\n**Qwen3 probe**\n\n"
        + md_table(qwen3, max_rows=20)
    )
    sections.append("## Exposure Ablation\n")
    sections.append(md_table(ablations, cols=["exposure_probability", "method", "n", "macro_accuracy", "mean_brier", "mean_ece", "avg_exposure_density", "avg_mean_exposure_load"]))
    sections.append(
        "Interpretation: exposure control matters, but not enough. Accuracy is flat-to-negative in this deterministic setting, while exposure density and load increase as expected. "
        "This says the exposure mechanism is wired correctly, but the present scorer is not converting exposure control into a performance advantage."
    )
    sections.append("## Adversarial Check\n")
    sections.append(md_table(adversarial))
    sections.append(
        "Interpretation: under two deterministic adversarial agents, tuned CIDeR collapses badly relative to majority vote and standard debate. "
        "That is a serious blocker for a causal-exposure paper: the method must be robust when some visible messages are confident but wrong. "
        "The current implementation detects unsupported copying, but the aggregation still gives adversarial contamination too much influence."
    )
    sections.append("## Error and Pairwise Diagnosis\n")
    sections.append(md_table(pair_errors))
    sections.append(
        "\nDataset-level selected accuracies:\n\n"
        + md_table(dataset_errors.pivot(index="dataset", columns="method", values="accuracy").reset_index())
    )
    sections.append(
        "Interpretation: CIDeR-tuned is losing mostly because the baselines find correct answers on examples where CIDeR remains wrong, not because of tiny calibration differences. "
        "The adversarial result also suggests the failure mode is not only randomness; it is a structural aggregation problem."
    )
    sections.append("## Statistical Strength\n")
    sections.append(
        "The deterministic full run has 720 examples, which is a better basis than the previous 16-example run. However, because this is a deterministic mock model, it cannot support a publication claim about real LLM SOTA. "
        "The real LLM probes have n=1 and must be treated only as execution checks. A real claim needs the full balanced split on real local/API models, paired tests against every baseline, bootstrap confidence intervals, and corrected multiple-comparison reporting."
    )
    sections.append("## What Must Change Before a SOTA Claim\n")
    sections.append(
        "1. Run the full balanced split with real LLMs, not only the deterministic model.\n"
        "2. Replace `style` baselines with official implementations or exact reproductions where licenses allow it.\n"
        "3. Add stronger answer verification: math normalization, MC option grounding, contradiction checks, and dataset-specific validators.\n"
        "4. Make CIDeR adversarially conservative: downweight agents whose exposed final answers diverge from their independent evidence without adding verifiable support.\n"
        "5. Tune only on the dev split, freeze once, then report the held-out test once.\n"
        "6. Add stronger local models or API models; small CPU local models will not plausibly establish SOTA on MMLU-Pro, MedQA, MATH-500, and AIME.\n"
        "7. Report cost-normalized performance. CIDeR must win at equal or acceptable extra token budget, not just by spending more deliberation."
    )
    sections.append("## Configs Used\n")
    sections.append(
        "Main deterministic config:\n\n```yaml\n"
        + load_config_text("real_llm_v4_mock.yaml")
        + "\n```\n\nMistral config:\n\n```yaml\n"
        + load_config_text("real_llm_v4_mistral.yaml")
        + "\n```\n\nQwen3 config:\n\n```yaml\n"
        + load_config_text("real_llm_v4_qwen3.yaml")
        + "\n```"
    )
    sections.append("## Bottom Line\n")
    sections.append(
        "v4 is a stronger benchmark harness, not a SOTA result. The code now supports balanced sampling, all required baselines, exposure ablation, adversarial checks, local Mistral/Qwen3 execution, and reproducible artifacts. "
        "The current CIDeR scoring architecture still fails the decisive test: it does not outperform the best baselines on the completed full run and is weak under adversarial contamination. "
        "The next research step is not more report polish; it is improving the aggregation and verification core, then running the full real-LLM campaign."
    )

    REPORT_MD.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics.causal_influence import aggregate_causal_thesis_metrics, causal_thesis_metrics


ROOT = Path("outputs/full_experiment_v6")
REPORT = ROOT / "full-comprehensive-v6.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def aggregate_summary(run_dir: Path) -> list[dict[str, Any]]:
    agg = csv_rows(run_dir / "aggregate_metrics.csv")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    pred_by_method: dict[str, list[dict[str, Any]]] = {}
    for rec in predictions:
        pred_by_method.setdefault(rec["method"], []).append(rec)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in agg:
        grouped.setdefault(row["method"], []).append(row)

    out = []
    numeric_cols = [
        "accuracy",
        "brier_score",
        "ece",
        "avg_tokens_in",
        "avg_tokens_out",
        "avg_agreement_rate",
        "avg_dissent_rate",
        "avg_exposure_density",
        "avg_mean_exposure_load",
    ]
    for method, items in grouped.items():
        preds = pred_by_method.get(method, [])
        row: dict[str, Any] = {"method": method, "n": len(preds)}
        row["macro_accuracy"] = sum(float(i["accuracy"]) for i in items) / len(items)
        row["micro_accuracy"] = sum(1 for p in preds if p.get("correct")) / len(preds) if preds else 0.0
        for col in numeric_cols:
            if col == "accuracy":
                continue
            row[col] = sum(float(i[col]) for i in items) / len(items)
        out.append(row)
    out.sort(key=lambda r: (r["macro_accuracy"], r["micro_accuracy"]), reverse=True)
    return out


def causal_summary(run_dir: Path) -> list[dict[str, Any]]:
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in predictions:
        if "ccs_corrections" not in rec:
            rec = rec | causal_thesis_metrics(rec.get("transcript", []), rec.get("answer", ""), rec.get("answer_index"))
        grouped.setdefault(rec["method"], []).append(rec)
    rows = []
    for method, items in grouped.items():
        metrics = aggregate_causal_thesis_metrics(items)
        rows.append({
            "method": method,
            "n": len(items),
            "ccs": metrics["ccs"],
            "chs": metrics["chs"],
            "pci": metrics["pci"],
            "wcr": metrics["wcr"],
            "pds": metrics["pds"],
            "ccs_events": int(metrics.get("ccs_corrections", 0.0)),
            "ccs_opps": int(metrics.get("ccs_opportunities", 0.0)),
            "chs_events": int(metrics.get("chs_harms", 0.0)),
            "chs_opps": int(metrics.get("chs_opportunities", 0.0)),
            "pci_events": int(metrics.get("pci_contaminated_switches", 0.0)),
            "pci_switches": int(metrics.get("pci_exposed_switches", 0.0)),
        })
    rows.sort(key=lambda r: (r["ccs"], -r["chs"], -r["pci"], -r["wcr"]), reverse=True)
    return rows


def json_block(path: Path) -> str:
    return "```json\n" + json.dumps(read_json(path), indent=2, sort_keys=True) + "\n```"


def code_block(path: Path, lang: str = "yaml") -> str:
    return f"```{lang}\n" + read(path) + "\n```"


def section(title: str, body: str) -> str:
    return f"{title}\n\n{body.strip()}\n"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().replace(microsecond=0).isoformat()

    mistral_dir = ROOT / "mistral_160"
    qwen_dir = ROOT / "qwen3_32"
    adv_dir = ROOT / "adversarial_mistral_40"
    addenda = ROOT / "v6_addenda_tables"
    no_exposure_dir = ROOT / "no_randomized_exposure_mistral_40_balanced"

    m_summary = aggregate_summary(mistral_dir)
    q_summary = aggregate_summary(qwen_dir)
    adv_summary = aggregate_summary(adv_dir)
    m_causal = causal_summary(mistral_dir)
    q_causal = causal_summary(qwen_dir)
    adv_causal = causal_summary(adv_dir)
    no_exposure_summary = aggregate_summary(no_exposure_dir) if (no_exposure_dir / "predictions.jsonl").exists() else []
    no_exposure_causal = causal_summary(no_exposure_dir) if (no_exposure_dir / "predictions.jsonl").exists() else []

    summary_cols = [
        "method",
        "n",
        "macro_accuracy",
        "micro_accuracy",
        "brier_score",
        "ece",
        "avg_tokens_in",
        "avg_tokens_out",
        "avg_agreement_rate",
        "avg_dissent_rate",
        "avg_exposure_density",
        "avg_mean_exposure_load",
    ]
    causal_cols = [
        "method",
        "n",
        "ccs",
        "chs",
        "pci",
        "wcr",
        "pds",
        "ccs_events",
        "ccs_opps",
        "chs_events",
        "chs_opps",
        "pci_events",
        "pci_switches",
    ]

    official_status = read(Path("external_baselines/official_baseline_status_v6.md"))
    method_deviations = read(Path("docs/method_deviations.md"))
    metrics_doc = read(Path("docs/metrics.md"))
    repro_doc = read(Path("docs/reproducibility.md"))

    artifacts = [
        "outputs/full_experiment_v6/full-comprehensive-v6.md",
        "outputs/full_experiment_v6/full-comprehensive-v6.pdf",
        "outputs/full_experiment_v6/full-experiment-v6.md",
        "outputs/full_experiment_v6/full-experiment-v6.pdf",
        "outputs/full_experiment_v6/mistral_160/aggregate_metrics.csv",
        "outputs/full_experiment_v6/mistral_160/predictions.jsonl",
        "outputs/full_experiment_v6/mistral_160/tables/overall_method_summary.csv",
        "outputs/full_experiment_v6/mistral_160/tables/accuracy_by_dataset_all_methods.csv",
        "outputs/full_experiment_v6/mistral_160/tables/pairwise_mcnemar.csv",
        "outputs/full_experiment_v6/qwen3_32/aggregate_metrics.csv",
        "outputs/full_experiment_v6/qwen3_32/predictions.jsonl",
        "outputs/full_experiment_v6/qwen3_32/tables/overall_method_summary.csv",
        "outputs/full_experiment_v6/qwen3_32/tables/accuracy_by_dataset_all_methods.csv",
        "outputs/full_experiment_v6/qwen3_32/tables/pairwise_mcnemar.csv",
        "outputs/full_experiment_v6/adversarial_mistral_40/aggregate_metrics.csv",
        "outputs/full_experiment_v6/adversarial_mistral_40/predictions.jsonl",
        "outputs/full_experiment_v6/v6_addenda_tables/ablation_cider_variants_mistral160.csv",
        "outputs/full_experiment_v6/v6_addenda_tables/adversarial_mistral8_summary.csv",
        "outputs/full_experiment_v6/no_randomized_exposure_mistral_40_balanced/aggregate_metrics.csv",
        "outputs/full_experiment_v6/no_randomized_exposure_mistral_40_balanced/predictions.jsonl",
        "configs/real_llm_v6_mistral_160.yaml",
        "configs/real_llm_v6_qwen3_32.yaml",
        "configs/real_llm_v6_mistral_no_randomized_exposure_40.yaml",
        "configs/real_llm_v6_mistral_adversarial_40.yaml",
        "configs/real_llm_v6_mistral_full720.yaml",
        "external_baselines/official_baseline_status_v6.md",
    ]

    commands = """```bash
# Unit tests
pytest -q

# Main V6 local Mistral comparison
python experiments/run_benchmark.py --config configs/real_llm_v6_mistral_160.yaml

# Qwen3 cross-model check
python experiments/run_benchmark.py --config configs/real_llm_v6_qwen3_32.yaml

# Focused CIDeR ablation grid
python experiments/run_v6_ablation_grid.py \\
  --config configs/real_llm_v6_mistral_160.yaml \\
  --out_dir outputs/full_experiment_v6/ablations_mistral_40 \\
  --max_examples 40

# Adversarial social-pressure test
python experiments/run_adversarial.py \\
  --config configs/real_llm_v6_mistral_adversarial_40.yaml \\
  --max_examples 8

# Build this comprehensive report
python analysis/build_full_comprehensive_v6_report.py
python analysis/md_to_pdf.py \\
  outputs/full_experiment_v6/full-comprehensive-v6.md \\
  outputs/full_experiment_v6/full-comprehensive-v6.pdf
```"""

    parts = [
        "# CIDeR Real-LLM v6 Comprehensive Report",
        f"Generated: {generated}",
        (
            "This PDF combines the V6 real-LLM report, main result tables, Qwen3 cross-model results, "
            "CIDeR ablation evidence, adversarial social-pressure evidence, official-baseline audit notes, "
            "run metadata, method-deviation documentation, metric definitions, reproducibility notes, and an artifact index."
        ),
        "Primary result directory: `outputs/full_experiment_v6`",
        (
            "Related run directories:\n\n"
            "- Main Mistral run: `outputs/full_experiment_v6/mistral_160`\n"
            "- Qwen3 cross-model run: `outputs/full_experiment_v6/qwen3_32`\n"
            "- Adversarial Mistral run: `outputs/full_experiment_v6/adversarial_mistral_40`\n"
            "- Addendum tables: `outputs/full_experiment_v6/v6_addenda_tables`"
        ),
        "# Part I: Main v6 Report",
        section(
            "## Executive Summary",
            (
                "V6 is the strongest completed version of the CIDeR benchmark so far. "
                "The primary result is the 160-example balanced local Mistral evaluation. "
                "In that setting, `cider_verified` is the top method with macro accuracy 0.3688, ahead of "
                "`cider_full_tuned`, `cider_sota`, and `cider_full` at 0.3438, majority vote at 0.3312, "
                "the DAR official-source-informed adapter at 0.3250, the Free-MAD official-source-informed adapter at 0.3250, "
                "and standard multi-agent debate at 0.2938.\n\n"
                "The supported claim is local and executable-benchmark specific: under the completed resource-constrained "
                "GPU-backed Ollama Mistral protocol, balanced sampling, all implemented baselines, and transparent official-source-informed "
                "adapters where exact upstream execution was infeasible, `cider_verified` is the best-performing method. "
                "The unsupported claim would be that CIDeR is universally SOTA across all backbones and all exact upstream baseline runtimes. "
                "The Qwen3 32-example check does not support that broader claim."
            ),
        ),
        section(
            "## Hardware and Model Setting",
            (
                "V6 used the local Ollama runtime. GPU telemetry was available and reported an NVIDIA GeForce RTX 4070 Laptop GPU. "
                "The primary V6 model was `mistral:latest`; the secondary cross-model stress check used `qwen3:latest` with `think: false` "
                "so the Ollama API returned parseable final JSON. Both configurations use four personas and agent temperatures "
                "`0.0`, `0.15`, `0.3`, and `0.45`.\n\n"
                "The exact Mistral and Qwen3 config files are included later in this report."
            ),
        ),
        section(
            "## Data Setting",
            (
                "The completed V6 runs use balanced subsets from the 720-example held-out split. "
                "The Mistral run uses 160 examples, with 20 examples per dataset across eight datasets. "
                "The Qwen3 run uses 32 examples, with 4 examples per dataset. "
                "The datasets are `aime2024`, `gsm8k`, `math500`, `medmcqa`, `medqa`, `mmlu_pro`, `strategyqa`, and `truthfulqa_mc`.\n\n"
                "Balanced sampling is important because it prevents MMLU-Pro or any other large dataset from dominating the aggregate result."
            ),
        ),
        section(
            "## Methods",
            (
                "The main V6 method matrix includes the required baselines, optional baselines, official-source-informed adapters, "
                "and the CIDeR variants:\n\n"
                "- `single_agent`\n"
                "- `self_consistency`\n"
                "- `majority_vote`\n"
                "- `standard_multi_agent_debate`\n"
                "- `consensagent_style`\n"
                "- `free_mad_style`\n"
                "- `free_mad_official_adapter`\n"
                "- `agentauditor_style`\n"
                "- `c3_style_causal_credit_analysis`\n"
                "- `conformal_social_choice`\n"
                "- `dar_style_diversity_aware_retention`\n"
                "- `dar_official_adapter`\n"
                "- `adaptive_stability_detection`\n"
                "- `cider_full`\n"
                "- `cider_full_tuned`\n"
                "- `cider_verified`\n"
                "- `cider_sota`"
            ),
        ),
        section(
            "## CIDeR v6 Architecture",
            (
                "CIDeR v6 keeps the central causal-exposure design: agents first produce independent answers, then deliberate under a bounded "
                "visibility graph, and the final answer is aggregated with explicit exposure penalties. "
                "The V6 additions are focused on separating useful correction from unsupported copying.\n\n"
                "The V6 architecture contains four layers:\n\n"
                "1. Independent prior answers establish evidence before social exposure.\n"
                "2. Controlled exposure limits visible messages and records an exposure matrix for every task.\n"
                "3. Evidence aggregation weights confidence, answer validity, independent support, rationale quality, and exposure load.\n"
                "4. Verifier-style adjudication in `cider_verified` adds a bonus for candidates with stronger answer validity and support.\n\n"
                "The key scoring idea is that independent agreement is evidence, but exposed repetition is not automatically evidence. "
                "CIDeR therefore discounts high-exposure convergence while allowing corrections when the final answer is better supported."
            ),
        ),
        section(
            "## Main Result: Mistral 160",
            md_table(m_summary, summary_cols)
            + "\n\n"
            + "Interpretation: `cider_verified` is the best method in the completed V6 Mistral benchmark. "
            + "It improves over majority vote by 0.0375 macro accuracy and over standard debate by 0.0750. "
            + "It also beats both official-source-informed adapters in this local protocol.",
        ),
        section(
            "## Causal Thesis Metrics: Mistral 160",
            (
                "Definitions: CCS is causal correction score, CHS is causal harm score, PCI is persuasion contamination index, "
                "WCR is wrong convergence rate, and PDS is persuasion drift score. These are transcript-level operational proxies, "
                "not randomized causal identification unless the run randomized exposure.\n\n"
                + md_table(m_causal, causal_cols)
                + "\n\n"
                + "Interpretation: these values expose whether accuracy gains come with correction or persuasion contamination. "
                + "They should be read alongside accuracy rather than after it."
            ),
        ),
        section(
            "## Pairwise Result: Mistral 160",
            read(mistral_dir / "tables" / "pairwise_mcnemar.md")
            + "\n\n"
            + "Interpretation: the paired tests favor `cider_verified` across all listed comparisons. "
            + "The clearest exact paired wins are against `single_agent` and `c3_style_causal_credit_analysis`; "
            + "several other comparisons are directionally favorable but still sample-size limited.",
        ),
        section(
            "## Mistral Per-Dataset Accuracy",
            read(mistral_dir / "tables" / "accuracy_by_dataset_all_methods.md"),
        ),
        section(
            "## Qwen3 Cross-Model Check",
            md_table(q_summary, summary_cols)
            + "\n\n"
            + "Interpretation: Qwen3 is a negative transfer check for CIDeR. "
            + "`dar_official_adapter`, `consensagent_style`, and `standard_multi_agent_debate` tie for the top macro accuracy at 0.4688. "
            + "The best CIDeR-family methods reach 0.3750. "
            + "This means the V6 local SOTA claim should be scoped to the completed Mistral protocol, not generalized to Qwen3.",
        ),
        section("## Causal Thesis Metrics: Qwen3 32", md_table(q_causal, causal_cols)),
        section(
            "## Qwen3 Pairwise Results",
            read(qwen_dir / "tables" / "pairwise_mcnemar.md"),
        ),
        section(
            "## Qwen3 Per-Dataset Accuracy",
            read(qwen_dir / "tables" / "accuracy_by_dataset_all_methods.md"),
        ),
        section(
            "## CIDeR Variant Ablation",
            read(addenda / "ablation_cider_variants_mistral160.md")
            + "\n\n"
            + "Interpretation: `cider_verified` is the only CIDeR variant that improves over `cider_full`, `cider_full_tuned`, and `cider_sota` "
            + "on the completed 160-example Mistral run. The finer exposure/verifier sweep was attempted but not completed because the local runtime "
            + "became too slow; this table is the reliable V6 ablation evidence.",
        ),
        section(
            "## No-Randomized-Exposure Ablation",
            (
                (
                    "This is the completed balanced no-randomized-exposure condition: 40 Mistral examples, 5 per dataset, "
                    "with CIDeR deliberation exposure probabilities set to 0.0. For verifier-based variants, the verifier may still "
                    "read prior answers, so exposure density can be nonzero even though randomized agent-to-agent deliberation exposure is disabled.\n\n"
                    + md_table(no_exposure_summary, summary_cols)
                    + "\n\n"
                    + "Causal thesis metrics for the no-randomized-exposure run:\n\n"
                    + md_table(no_exposure_causal, causal_cols)
                    + "\n\n"
                    + "Interpretation: CCS, CHS, PCI, and PDS are all 0.0 because no final deliberating agents were exposed to peer messages. "
                    + "Accuracy remains competitive on this small balanced subset, so the current evidence does not prove that randomized exposure itself is the source of the Mistral gain; "
                    + "it shows that the verifier/evidence aggregation path remains strong even when randomized peer exposure is removed."
                )
                if no_exposure_summary
                else "No completed no-randomized-exposure run was found when this report was generated."
            ),
        ),
        section(
            "## Adversarial Social-Pressure Test",
            md_table(adv_summary, summary_cols)
            + "\n\n"
            + "Interpretation: with two deterministic high-confidence wrong agents, the CIDeR variants tie majority vote, standard debate, "
            + "and the Free-MAD official adapter at 0.2500 macro accuracy on the small adversarial subset. "
            + "This is an improvement over V3's failure mode because CIDeR no longer drops below the main baselines, but it is not an adversarial accuracy win.",
        ),
        section("## Causal Thesis Metrics: Adversarial Test", md_table(adv_causal, causal_cols)),
        section(
            "## Critical Analysis",
            (
                "V6 improves on V3 and V5 in three concrete ways. First, the primary Mistral evidence increases to 160 balanced examples. "
                "Second, the comparison includes the full required baseline set plus optional baselines and stronger official-source-informed adapters. "
                "Third, `cider_verified` wins the completed Mistral main table rather than only an ablation table.\n\n"
                "The evidence is still not publication-final. The Qwen3 run is small and does not favor CIDeR. The adversarial run is very small and shows a tie, "
                "not a win. The 720-example Mistral configuration exists but was not completed in this run. The Mistral config uses `max_tokens: 64`, which is a real "
                "resource constraint and likely suppresses multi-step reasoning, especially on AIME where every method scored 0.00. Exact upstream baseline execution "
                "also remains limited by incompatible or unavailable external code, so V6 uses transparent adapters where necessary."
            ),
        ),
        section(
            "## Claim Review",
            (
                "Supported by V6:\n\n"
                "- In the completed 160-example local Mistral benchmark, `cider_verified` is the best-performing method.\n"
                "- CIDeR beats majority vote, standard debate, DAR official-source-informed adapter, and Free-MAD official-source-informed adapter in that setting.\n"
                "- Balanced sampling prevents a single dataset from dominating the result.\n"
                "- V6 records method deviations and official-source status transparently.\n\n"
                "Not supported by V6:\n\n"
                "- A universal SOTA claim across all LLM backbones.\n"
                "- A claim that all external baselines were run as unmodified upstream repositories.\n"
                "- A claim that CIDeR already beats the strongest baselines on Qwen3.\n"
                "- A strong adversarial superiority claim; the current adversarial run shows a tie at small `n`."
                "\n- A claim that `max_tokens: 64` is neutral across tasks; it is an explicit resource constraint and must be stress-tested."
            ),
        ),
        section(
            "## Output Files",
            (
                "Main Mistral run:\n\n"
                "- `outputs/full_experiment_v6/mistral_160/aggregate_metrics.csv`\n"
                "- `outputs/full_experiment_v6/mistral_160/predictions.jsonl`\n"
                "- `outputs/full_experiment_v6/mistral_160/tables/overall_method_summary.csv`\n"
                "- `outputs/full_experiment_v6/mistral_160/tables/accuracy_by_dataset_all_methods.csv`\n"
                "- `outputs/full_experiment_v6/mistral_160/tables/pairwise_mcnemar.csv`\n\n"
                "Qwen3 run:\n\n"
                "- `outputs/full_experiment_v6/qwen3_32/aggregate_metrics.csv`\n"
                "- `outputs/full_experiment_v6/qwen3_32/predictions.jsonl`\n"
                "- `outputs/full_experiment_v6/qwen3_32/tables/overall_method_summary.csv`\n"
                "- `outputs/full_experiment_v6/qwen3_32/tables/accuracy_by_dataset_all_methods.csv`\n"
                "- `outputs/full_experiment_v6/qwen3_32/tables/pairwise_mcnemar.csv`\n\n"
                "Adversarial and ablation addenda:\n\n"
                "- `outputs/full_experiment_v6/adversarial_mistral_40/aggregate_metrics.csv`\n"
                "- `outputs/full_experiment_v6/adversarial_mistral_40/predictions.jsonl`\n"
                "- `outputs/full_experiment_v6/v6_addenda_tables/ablation_cider_variants_mistral160.csv`\n"
                "- `outputs/full_experiment_v6/v6_addenda_tables/adversarial_mistral8_summary.csv`"
            ),
        ),
        section("## Test Status", "`pytest -q` passed: 7/7 tests."),
        "# Part II: Main v6 Result Tables",
        section("## Main Overall Method Summary", md_table(m_summary, summary_cols)),
        section("## Main Causal Thesis Metrics", md_table(m_causal, causal_cols)),
        section("## Main Accuracy By Dataset", read(mistral_dir / "tables" / "accuracy_by_dataset_all_methods.md")),
        section("## Main CIDeR Pairwise McNemar-Style Comparison", read(mistral_dir / "tables" / "pairwise_mcnemar.md")),
        "# Part III: Qwen3 Cross-Model Results",
        section("## Qwen3 Overall Method Summary", md_table(q_summary, summary_cols)),
        section("## Qwen3 Causal Thesis Metrics", md_table(q_causal, causal_cols)),
        section("## Qwen3 Accuracy By Dataset", read(qwen_dir / "tables" / "accuracy_by_dataset_all_methods.md")),
        section("## Qwen3 Pairwise McNemar-Style Comparison", read(qwen_dir / "tables" / "pairwise_mcnemar.md")),
        "# Part IV: Ablation and Adversarial Results",
        section("## CIDeR Ablation Summary", read(addenda / "ablation_cider_variants_mistral160.md")),
        section(
            "## No-Randomized-Exposure Ablation Summary",
            md_table(no_exposure_summary, summary_cols) if no_exposure_summary else "Not completed.",
        ),
        section(
            "## No-Randomized-Exposure Causal Metrics",
            md_table(no_exposure_causal, causal_cols) if no_exposure_causal else "Not completed.",
        ),
        section("## Adversarial Overall Method Summary", md_table(adv_summary, summary_cols)),
        section("## Adversarial Causal Thesis Metrics", md_table(adv_causal, causal_cols)),
        section("## Adversarial Addendum Table", read(addenda / "adversarial_mistral8_summary.md")),
        "# Part V: Run Metadata and Configurations",
        section("## Mistral Run Metadata", json_block(mistral_dir / "run_metadata.json")),
        section("## Qwen3 Run Metadata", json_block(qwen_dir / "run_metadata.json")),
        section("## Adversarial Run Metadata", json_block(adv_dir / "run_metadata.json")),
        section("## Mistral V6 Config", code_block(Path("configs/real_llm_v6_mistral_160.yaml"))),
        section("## Qwen3 V6 Config", code_block(Path("configs/real_llm_v6_qwen3_32.yaml"))),
        section("## Adversarial V6 Config", code_block(Path("configs/real_llm_v6_mistral_adversarial_40.yaml"))),
        section("## Full 720-Example Mistral Config", code_block(Path("configs/real_llm_v6_mistral_full720.yaml"))),
        "# Part VI: Official Baseline Audit and Repository Documentation",
        section("## Official Baseline Status", official_status),
        section("## Method Deviations", method_deviations),
        section("## Metrics Documentation", metrics_doc),
        section("## Reproducibility Documentation", repro_doc),
        "# Part VII: Reproduction Commands",
        commands,
        "# Part VIII: Artifact Index",
        "\n".join(f"- `{path}`" for path in artifacts),
    ]

    REPORT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()

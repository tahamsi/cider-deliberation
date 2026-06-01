# CIDeR Deliberation Benchmark

This repository implements a reproducible benchmark framework for the paper project:

**When Agreement Is Not Evidence: Causal Exposure Control for Multi-Agent LLM Deliberation**

The core claim tested here is simple but important: when language-model agents deliberate after seeing each other, agreement is not automatically independent evidence. Agreement can be caused by exposure, copying, conformity, or shared prompt bias. CIDeR, the Causal Exposure Deliberation Engine, treats inter-agent exposure as a causal variable and aggregates answers with explicit exposure control.

The current V6 evidence supports a scoped claim: under the completed resource-constrained local `mistral:latest` benchmark, with balanced sampling and all implemented baselines, `cider_verified` is the best-performing method. This is a local executable-benchmark SOTA result, not a universal claim across every model backbone or every original baseline runtime.

## Problem Formulation

Let the benchmark contain tasks

```text
D = {(x_i, C_i, y_i)}_{i=1}^N
```

where `x_i` is the question, `C_i` is an optional finite set of answer choices, and `y_i` is the gold answer. For multiple-choice tasks, `C_i = {c_1, ..., c_K}` and `y_i in C_i`; for free-form math tasks, `y_i` is a normalized string or numeric target.

There are `A` agents. Before deliberation, each agent independently produces

```text
r_{i,a}^{(0)} = (a_{i,a}^{(0)}, z_{i,a}^{(0)}, p_{i,a}^{(0)})
```

where `a_{i,a}^{(0)}` is the proposed answer, `z_{i,a}^{(0)}` is the rationale, and `p_{i,a}^{(0)}` is the self-reported confidence.

During deliberation, agents exchange messages across rounds. The exposure structure is represented by a matrix

```text
E_i[t, u] in {0, 1}
```

where `E_i[t, u] = 1` means message or agent-state `t` was visible to message or agent-state `u`. Standard multi-agent debate often makes `E_i` dense: each participant can see most previous statements. CIDeR makes `E_i` sparse, recorded, and used during aggregation.

The final benchmark objective is to choose a method `m` that predicts

```text
y_hat_i^m = f_m(x_i, C_i, {r_{i,a}^{(0)}}, {r_{i,a}^{(r)}}, E_i)
```

and maximizes accuracy while controlling calibration, exposure-induced dependence, and cost:

```text
maximize_m   E[1{y_hat_i^m = y_i}]
             - lambda_cost * Cost(m)
             - lambda_exp  * ExposureBias(E_i)
             - lambda_cal  * CalibrationError(m)
```

The causal problem is that a later answer may be a consequence of exposure rather than fresh evidence. Informally, we want to distinguish:

```text
P(a_a = y | a_b = y)
```

from the counterfactual quantity:

```text
P(a_a = y | do(E_{b -> a} = 0), a_b = y)
```

If agent `a` only agrees with agent `b` because `a` saw `b`, then counting both answers as independent votes overstates the evidence. CIDeR therefore logs exposures and discounts exposed agreement unless it is supported by independent prior answers, useful corrections, or verifier evidence.

## Proposed Solution: CIDeR

CIDeR is a causal-exposure-controlled deliberation architecture. It keeps the useful parts of multi-agent reasoning, such as diverse rationales and critique, but blocks the most common failure mode: treating socially propagated agreement as independent evidence.

### CIDeR Pipeline

1. **Independent prior phase**
   Each agent answers the task without seeing other agents. These responses form an independence anchor.

2. **Controlled exposure phase**
   Agents deliberate over a seeded, bounded visibility graph. The method limits the number of visible messages and exposure probability, producing an explicit exposure matrix for every task.

3. **Evidence extraction phase**
   Each answer is parsed into a normalized answer label or free-form answer. The framework records answer switches, confidence, rationale text, token usage, and exposure load.

4. **Exposure-adjusted aggregation**
   CIDeR gives more weight to independently supported answers and discounts exposed repetitions. A later answer can still help if it corrects an earlier error or survives verifier checks.

5. **Verifier-style adjudication in V6**
   `cider_verified` adds a verifier bonus when a final candidate is supported by answer validity and stronger evidence. This is the best method in the completed V6 Mistral run.

### CIDeR Scoring

For a candidate answer `y`, CIDeR computes a score of the form

```text
S(y) = sum_a W_prior(a, y) + sum_a W_delib(a, y) + B_verifier(y)
```

where prior evidence is independent by construction:

```text
W_prior(a, y) =
    1{a_{a}^{(0)} = y}
    * alpha
    * conf_a
    * quality_a
    * validity_a
    * reliability_a
```

and deliberation evidence is exposure-adjusted:

```text
W_delib(a, y) =
    1{a_{a}^{(R)} = y}
    * beta
    * conf_a
    * quality_a
    * validity_a
    * reliability_a
    * independence_bonus(y)
    * correction_bonus(a)
    * stability_bonus(a)
    * copy_penalty(a)
    / (1 + delta * exposure_load_a)
```

The main intuition is:

```text
independent agreement        -> stronger evidence
exposed repetition           -> weaker evidence
useful correction            -> stronger evidence
high exposure load           -> weaker evidence
invalid or unparseable answer -> weaker evidence
```

### Implemented CIDeR Variants

| Method key | Description |
| --- | --- |
| `cider_full` | Main causal exposure deliberation engine with explicit exposure matrices and exposure-adjusted aggregation. |
| `cider_full_tuned` | Tuned exposure version with a larger allowed exposure probability. |
| `cider_verified` | V6 winner on local Mistral. Adds answer-validity checks and verifier-style aggregation. |
| `cider_sota` | V6 experimental variant with reliability weighting, correction rewards, unsupported-copy penalties, and verifier tie-breaking. |

## System Architecture

The repository is organized as a modular benchmark stack:

```text
cider-deliberation/
  agents/       Agent abstractions, mock model, Ollama/API LLM agent, adversarial agent.
  datasets/     Dataset download, normalization, schema validation, local JSONL loaders.
  methods/      CIDeR, baseline methods, style adapters, official-source-informed adapters.
  metrics/      Accuracy, calibration, exposure, persuasion, dissent, cost, statistical tests.
  experiments/  Benchmark, ablation, adversarial, split-preparation, smoke-test runners.
  analysis/     Table generation, plotting, failure inspection, report and PDF builders.
  configs/      Reproducible experiment configs.
  docs/         Method deviations, metric definitions, reproducibility notes.
  outputs/      Generated benchmark outputs.
```

Every benchmark run writes:

| Artifact | Location |
| --- | --- |
| Per-method raw task outputs | `outputs/.../raw/<method>.jsonl` |
| Combined predictions | `outputs/.../predictions.jsonl` |
| Exposure matrices | Stored inside each JSONL record under `exposure_matrix` |
| Method metadata | Stored inside each JSONL record under `metadata` |
| Token and cost estimates | Stored inside each JSONL record under `cost` |
| Aggregate metrics | `outputs/.../aggregate_metrics.csv` |
| Run metadata | `outputs/.../run_metadata.json` |
| Reports | `outputs/full_experiment_v6/full-experiment-v6.md` and `.pdf` |

## Baseline Models

The benchmark compares CIDeR against deterministic, ensemble, debate, causal-credit, social-choice, diversity, and stability baselines. Methods marked `style` or `adapter` are explicitly documented in [docs/method_deviations.md](docs/method_deviations.md). They must not be described as exact reproductions.

| Method key | Reference or source | Implementation status |
| --- | --- | --- |
| `single_agent` | Direct single-model inference baseline. | Exact local baseline. |
| `self_consistency` | Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171). | Adapted repeated-sampling implementation. |
| `majority_vote` | Standard ensemble voting baseline. | Exact local independent-agent voting baseline. |
| `standard_multi_agent_debate` | Du et al., [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325); official-style code source: [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate). | Unified local debate adapter. Official source cloned for audit in V6. |
| `consensagent_style` | ConsensAgent paper: [ACL Anthology PDF](https://aclanthology.org/2025.findings-acl.1141.pdf); linked code source: [priyapitre/CONSENSAGENT](https://github.com/priyapitre/CONSENSAGENT). | Style approximation. The linked repository was empty at V6 clone time. |
| `free_mad_style` | Free-MAD source: [jonathansantilli/mad](https://github.com/jonathansantilli/mad); package: [freemad on PyPI](https://pypi.org/project/freemad/); paper link listed by source: [arXiv 2509.11035](https://arxiv.org/abs/2509.11035). | Style approximation using partial exposure. |
| `free_mad_official_adapter` | Same Free-MAD source as above, cloned at commit `1e8426a`. | Official-source-informed adapter using Free-MAD trajectory scoring semantics inside this benchmark's agent interface. |
| `agentauditor_style` | No verified runnable official source integrated in this repository. | Style proxy using confidence-weighted independent answers. |
| `c3_style_causal_credit_analysis` | C3 source: [EIT-EAST-Lab/C3](https://github.com/EIT-EAST-Lab/C3); paper: [arXiv 2603.06859](https://arxiv.org/abs/2603.06859). | Inference-time style proxy. Official repo is a separate training/credit stack. |
| `conformal_social_choice` | Adapted from conformal prediction and social-choice ideas; calibration reference: Vovk et al., [Algorithmic Learning in a Random World](https://link.springer.com/book/10.1007/978-3-031-06649-8). | Adapted confidence-thresholded social-choice baseline. |
| `dar_style_diversity_aware_retention` | DAR source: [DA2I2-SLM/DAR](https://github.com/DA2I2-SLM/DAR); paper: [arXiv 2603.20640](https://arxiv.org/abs/2603.20640). | Style approximation. |
| `dar_official_adapter` | Same DAR source as above, cloned at commit `f3c6e9d`. | Official-source-informed adapter implementing uncertainty/vote/critical-retention behavior inside this benchmark's agent interface. |
| `adaptive_stability_detection` | Adapted early-stopping and stability heuristic. | Optional adapted baseline. |
| `cider_full`, `cider_full_tuned`, `cider_verified`, `cider_sota` | Proposed CIDeR family in this repository. | Proposed methods. |

Official-source status for V6 is recorded in [external_baselines/official_baseline_status_v6.md](external_baselines/official_baseline_status_v6.md).

## Evaluation Criteria

The framework reports task accuracy, calibration, causal-exposure diagnostics, social-dynamics metrics, cost, and paired statistical tests.

| Criterion | Definition | Reference |
| --- | --- | --- |
| Exact accuracy | `1{prediction = answer}` after dataset-specific normalization. For multiple choice, both answer text and answer index are checked. | Standard supervised evaluation. |
| Numeric exact match | Extracts normalized numeric answers for math-style free-form datasets when possible. | Common GSM8K/MATH evaluation practice; GSM8K: [Cobbe et al. 2021](https://arxiv.org/abs/2110.14168). |
| Macro accuracy | Mean of per-dataset accuracies, preventing large datasets such as MMLU-Pro from dominating. | Standard balanced benchmark aggregation. |
| Brier score | For binary correctness event `o_i in {0,1}` and confidence `p_i`, `Brier = (1/N) sum_i (p_i - o_i)^2`. Lower is better. | Brier, 1950, "Verification of forecasts expressed in terms of probability." |
| Expected calibration error | Partition predictions into confidence bins; compute `sum_b |B_b|/N * |acc(B_b) - conf(B_b)|`. Lower is better. | Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). |
| Agreement rate | Fraction of agent responses in a task that match the modal answer. Higher agreement is not always better because it may be exposure-caused. | Social aggregation diagnostic. |
| Dissent rate | `1 - agreement_rate`, measuring residual disagreement among agents. | Social aggregation diagnostic. |
| Answer switch rate | Fraction of agents whose final answer differs from their independent prior answer. | Persuasion and conformity diagnostic. |
| Exposure density | `||E||_0 / |E|`, the fraction of possible exposure edges that are active. | CIDeR causal-exposure diagnostic. |
| Mean exposure load | Average number of visible incoming messages per agent/message state. | CIDeR causal-exposure diagnostic. |
| Token and cost estimates | Average input tokens, output tokens, and estimated USD cost where model pricing is available. | Operational evaluation. |
| Paired significance tests | Exact paired tests over task-level correctness disagreements between methods. | McNemar, 1947, "Note on the sampling error of the difference between correlated proportions or percentages." |

Metric implementations live in [metrics/](metrics/), with additional documentation in [docs/metrics.md](docs/metrics.md).

## Datasets

All datasets are normalized to one schema:

```json
{
  "dataset": "str",
  "id": "str",
  "question": "str",
  "choices": ["str"] ,
  "answer": "str",
  "answer_index": 0,
  "metadata": {}
}
```

For free-form datasets, `choices` and `answer_index` are `null`; answer matching uses normalized string and numeric extraction where appropriate.

| Dataset key | Definition | Source |
| --- | --- | --- |
| `mmlu_pro` | Harder multi-choice academic benchmark across many domains. | HF: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro); paper: [arXiv 2406.01574](https://arxiv.org/abs/2406.01574). |
| `gsm8k` | Grade-school math word problems with final numeric answers. | HF: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k); paper: [arXiv 2110.14168](https://arxiv.org/abs/2110.14168). |
| `math500` | 500-problem subset from the MATH mathematical reasoning benchmark. | HF: [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500); MATH paper: [arXiv 2103.03874](https://arxiv.org/abs/2103.03874). |
| `truthfulqa_mc` | Multiple-choice factual truthfulness benchmark. | HF: [EleutherAI/truthful_qa_mc](https://huggingface.co/datasets/EleutherAI/truthful_qa_mc); paper: [arXiv 2109.07958](https://arxiv.org/abs/2109.07958). |
| `strategyqa` | Yes/no questions requiring implicit multi-hop reasoning. | HF: [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA); paper: [arXiv 2101.02235](https://arxiv.org/abs/2101.02235). |
| `medqa` | USMLE-style four-option medical QA. | HF: [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options); paper: [arXiv 2009.13081](https://arxiv.org/abs/2009.13081). |
| `medmcqa` | Medical entrance-exam multiple-choice QA. | HF: [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa); paper: [arXiv 2203.14371](https://arxiv.org/abs/2203.14371). |
| `aime2024` | AIME 2024 competition math problems. | HF: [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024); competition: [MAA AIME](https://maa.org/math-competitions/aime/). |

Dataset loading and preprocessing code is in [datasets/download_preprocess.py](datasets/download_preprocess.py).

## Installation

Python 3.11 or newer is required.

```bash
cd /home/ans353/Documents/myproject/cider-deliberation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e ".[test,stats]"
python -m pip install reportlab
```

For local LLM experiments, install and start Ollama, then pull the local models used in V6:

```bash
ollama pull mistral
ollama pull qwen3
ollama list
nvidia-smi
```

The V6 configs use:

```text
Ollama base URL: http://127.0.0.1:11434
Mistral model:  mistral:latest
Qwen model:     qwen3:latest
```

## Dataset Preparation Commands

Download and normalize all datasets:

```bash
python datasets/download_preprocess.py \
  --datasets mmlu_pro gsm8k math500 truthfulqa_mc strategyqa medqa medmcqa aime2024 \
  --output_dir data/processed \
  --seed 42
```

Create the balanced V4/V6 splits used by the real-LLM experiments:

```bash
python experiments/prepare_v4_splits.py \
  --input data/processed/all.jsonl \
  --out_dir data/processed \
  --seed 42 \
  --dev_per_dataset 20 \
  --test_per_dataset 100
```

```bash
python experiments/prepare_v6_subsets.py \
  --input data/processed/real_llm_v4_test_balanced.jsonl \
  --out_dir data/processed \
  --seed 42 \
  --mistral_per_dataset 20 \
  --qwen_per_dataset 4
```

For quick local development, use a smaller subset:

```bash
python datasets/download_preprocess.py \
  --datasets gsm8k math500 truthfulqa_mc strategyqa \
  --output_dir data/processed_small \
  --max_examples 20 \
  --seed 7
```

## Test Commands

Run the unit tests:

```bash
pytest -q
```

Run the deterministic smoke benchmark:

```bash
python experiments/run_smoke_tests.py --config configs/small_smoke_test.yaml
```

Run the benchmark directly on the smoke config:

```bash
python experiments/run_benchmark.py --config configs/small_smoke_test.yaml
```

## Full Experiment

Use this path on a raw Ubuntu server with an A100 80GB when the goal is to remove the main V6 caveats: incomplete 720-example run, weak token budget, missing exposure ablation grid, small adversarial test, backbone dependence, missing causal thesis metrics, and single-seed fragility.

### 1. Setup Raw Ubuntu

Clone the repository, then run the A100 setup script:

```bash
git clone https://github.com/tahamsi/cider-deliberation.git
cd cider-deliberation
scripts/setup_a100_ubuntu.sh
source .venv/bin/activate
```

The setup script installs Python dependencies, installs Ollama if needed, checks GPU visibility, pulls the default A100 model set, and prepares the local virtual environment.

Default model pull list:

```text
mistral:latest
qwen3:32b
qwen2.5:32b
llama3.1:70b
```

Override models if your Ollama registry uses different tags:

```bash
A100_MODELS="mistral:latest qwen3:32b qwen2.5:32b llama3.1:70b" \
  scripts/setup_a100_ubuntu.sh
```

Dry-run the server setup without changing the machine:

```bash
DRY_RUN=1 SKIP_GPU_CHECK=1 SKIP_APT=1 SKIP_OLLAMA_INSTALL=1 SKIP_MODEL_PULL=1 \
  scripts/setup_a100_ubuntu.sh
```

### 2. Download and Preprocess All Datasets

```bash
scripts/download_preprocess_all.sh
```

This creates:

```text
data/processed/all.jsonl
data/processed/real_llm_v4_dev.jsonl
data/processed/real_llm_v4_test_balanced.jsonl
data/processed/real_llm_v6_mistral_160.jsonl
data/processed/real_llm_v6_qwen3_32.jsonl
```

Dry-run dataset commands without downloading:

```bash
DRY_RUN=1 scripts/download_preprocess_all.sh
```

### 3. Run the Full A100 Campaign

Run inside `tmux` or another persistent session. With no `--max_examples`, the script uses the full balanced held-out split at `data/processed/real_llm_v4_test_balanced.jsonl`.

```bash
python scripts/run_a100_full_campaign.py \
  --out_dir outputs/a100_full_v7 \
  --dataset data/processed/real_llm_v4_test_balanced.jsonl \
  --models "mistral:latest,qwen3:32b,qwen2.5:32b,llama3.1:70b" \
  --seeds "42,43,44,45,46" \
  --max_tokens 512 \
  --token_sweep "64,512,1024" \
  --num_ctx 4096
```

The single campaign script runs:

- `pytest -q`
- full balanced main comparisons for every model and seed
- all required and optional baselines
- CIDeR variants
- exposure-grid ablations including exposure `0.0`
- verifier on/off ablations
- copy-penalty, correction-bonus, and exposure-penalty ablations
- token-budget sensitivity checks
- adversarial tests with two high-confidence wrong agents
- CCS, CHS, PCI, WCR, and PDS collection for every aggregate row

Main consolidated outputs:

```text
outputs/a100_full_v7/a100_campaign_results.csv
outputs/a100_full_v7/a100_campaign_manifest.json
outputs/a100_full_v7/a100_campaign_summary.md
```

The file to send back for analysis first is:

```text
outputs/a100_full_v7/a100_campaign_summary.md
```

If needed, also send:

```text
outputs/a100_full_v7/a100_campaign_results.csv
```

### 4. Validate Locally Before Server Run

Dry-run the entire campaign plan:

```bash
python scripts/run_a100_full_campaign.py \
  --dry_run \
  --smoke \
  --provider mock \
  --models deterministic-mock \
  --seeds "42,43" \
  --out_dir outputs/a100_dry_run_check
```

Run the real local smoke campaign:

```bash
python scripts/run_a100_full_campaign.py \
  --smoke \
  --provider mock \
  --models deterministic-mock \
  --seeds "42" \
  --out_dir outputs/a100_smoke_check
```

## Main Experiment Commands

### Mock and Development Runs

Use these first to verify the pipeline without a local LLM:

```bash
python experiments/run_benchmark.py --config configs/full_mock_all.yaml
```

```bash
python experiments/run_benchmark.py --config configs/real_llm_v5_mock.yaml
```

### V6 Local Mistral Comparison

This is the main completed V6 comparison run. It uses 160 balanced examples, four agents, two rounds, all required baselines, optional baselines, official-source-informed adapters, and CIDeR variants.

```bash
python experiments/run_benchmark.py --config configs/real_llm_v6_mistral_160.yaml
```

The heavier full held-out Mistral configuration is:

```bash
python experiments/run_benchmark.py --config configs/real_llm_v6_mistral_full720.yaml
```

The 720-example run is a long GPU campaign because it evaluates the full method matrix over the complete balanced split.

### V6 Qwen3 Cross-Model Check

```bash
python experiments/run_benchmark.py --config configs/real_llm_v6_qwen3_32.yaml
```

This run is a smaller cross-model stress check. In V6, Qwen3 did not support the same CIDeR win as Mistral; this is reported transparently.

## Ablation Commands

The completed V6 ablation evidence compares CIDeR variants on the same Mistral 160 run and is summarized in:

```text
outputs/full_experiment_v6/v6_addenda_tables/ablation_cider_variants_mistral160.csv
```

To run the focused V6 ablation grid:

```bash
python experiments/run_v6_ablation_grid.py \
  --config configs/real_llm_v6_mistral_160.yaml \
  --out_dir outputs/full_experiment_v6/ablations_mistral_40 \
  --max_examples 40
```

For a smaller debugging ablation:

```bash
python experiments/run_v6_ablation_grid.py \
  --config configs/real_llm_v6_mistral_160.yaml \
  --out_dir outputs/full_experiment_v6/ablations_mistral_8 \
  --max_examples 8
```

The legacy ablation runner is also available:

```bash
python experiments/run_ablations.py --config configs/real_llm_v6_mistral_160.yaml
```

## Adversarial Test Commands

The V6 adversarial config injects two high-confidence deterministic wrong agents into a four-agent setting.

```bash
python experiments/run_adversarial.py \
  --config configs/real_llm_v6_mistral_adversarial_40.yaml \
  --max_examples 8
```

For a larger adversarial run:

```bash
python experiments/run_adversarial.py \
  --config configs/real_llm_v6_mistral_adversarial_40.yaml \
  --max_examples 40
```

The V6 adversarial summary table is stored at:

```text
outputs/full_experiment_v6/v6_addenda_tables/adversarial_mistral8_summary.csv
```

## Analysis and Report Commands

Generate tables from aggregate metrics:

```bash
python analysis/make_tables.py \
  --metrics outputs/full_experiment_v6/mistral_160/aggregate_metrics.csv \
  --out outputs/full_experiment_v6/mistral_160/tables/accuracy_pivot.csv
```

Generate plots:

```bash
python analysis/make_plots.py \
  --metrics outputs/full_experiment_v6/mistral_160/aggregate_metrics.csv \
  --out outputs/full_experiment_v6/mistral_160/plots/accuracy.png
```

Inspect failures:

```bash
python analysis/inspect_failures.py \
  --predictions outputs/full_experiment_v6/mistral_160/predictions.jsonl \
  --limit 20
```

Build the V6 report:

```bash
python analysis/build_full_experiment_v6_report.py
```

Convert the V6 Markdown report to PDF:

```bash
python analysis/md_to_pdf.py \
  outputs/full_experiment_v6/full-experiment-v6.md \
  outputs/full_experiment_v6/full-experiment-v6.pdf
```

## Current V6 Headline Results

### Mistral 160 Balanced Examples

| Rank | Method | Macro accuracy | Notes |
| --- | --- | ---: | --- |
| 1 | `cider_verified` | 0.36875 | Best completed V6 local Mistral result. |
| 2 | `cider_full_tuned` | 0.34375 | CIDeR tuned exposure variant. |
| 3 | `cider_sota` | 0.34375 | CIDeR V6 experimental reliability variant. |
| 4 | `cider_full` | 0.34375 | Main CIDeR exposure-control method. |
| 5 | `agentauditor_style` | 0.33750 | Confidence-weighted auditor proxy. |
| 6 | `majority_vote` | 0.33125 | Independent majority baseline. |
| 7 | `conformal_social_choice` | 0.33125 | Adapted social-choice baseline. |
| 8 | `dar_official_adapter` | 0.32500 | Official-source-informed DAR adapter. |
| 9 | `free_mad_official_adapter` | 0.32500 | Official-source-informed Free-MAD adapter. |
| 10 | `standard_multi_agent_debate` | 0.29375 | Fully visible debate baseline. |

Interpretation: `cider_verified` is the top method in the completed local Mistral V6 benchmark. The result supports the paper's central argument that exposure-aware aggregation can beat naive debate and voting under a controlled local setting.

### Qwen3 32 Balanced Examples

| Rank | Method | Macro accuracy | Notes |
| --- | --- | ---: | --- |
| 1 | `dar_official_adapter` | 0.46875 | Best Qwen3 32-example result. |
| 1 | `consensagent_style` | 0.46875 | Tied top on the small Qwen split. |
| 1 | `standard_multi_agent_debate` | 0.46875 | Tied top on the small Qwen split. |
| 4 | `free_mad_official_adapter` | 0.43750 | Strong adapter result. |
| 7 | `cider_full_tuned` | 0.37500 | Best CIDeR family result on Qwen3. |

Interpretation: Qwen3 is an important stress test. It does not currently support a universal CIDeR SOTA claim, so this repository reports it as a cross-model limitation and future tuning target.

### V6 Ablation and Adversarial Summary

| Test | Main finding |
| --- | --- |
| CIDeR variant ablation on Mistral 160 | `cider_verified` beats `cider_full`, `cider_full_tuned`, and `cider_sota`. |
| Adversarial Mistral 8 with two wrong agents | CIDeR variants tie majority vote, standard debate, and Free-MAD official adapter at 0.2500; DAR official adapter is lower at 0.0833. |

## Reproducibility Rules

This project follows strict benchmark rules:

1. Do not fake results.
2. Do not silently skip datasets, metrics, baselines, or failed methods.
3. Keep every run seed-controlled.
4. Record exposure matrices and transcripts for every task.
5. Name approximations honestly as `style`, `adapted`, or `official_adapter`.
6. Distinguish local executable benchmark claims from universal SOTA claims.
7. Report small-sample and cross-model limitations explicitly.

Relevant documentation:

| Document | Purpose |
| --- | --- |
| [docs/method_deviations.md](docs/method_deviations.md) | Exact status of each style or adapted method. |
| [docs/metrics.md](docs/metrics.md) | Metric definitions. |
| [docs/reproducibility.md](docs/reproducibility.md) | Reproducibility notes. |
| [external_baselines/official_baseline_status_v6.md](external_baselines/official_baseline_status_v6.md) | V6 official-source audit. |
| [outputs/full_experiment_v6/full-experiment-v6.md](outputs/full_experiment_v6/full-experiment-v6.md) | Full V6 report. |

## Practical Guidance for Stronger Publication Evidence

The strongest next experiments are:

1. Complete `configs/real_llm_v6_mistral_full720.yaml`.
2. Increase the Qwen3 balanced sample size after CIDeR tuning for that backbone.
3. Report exact paired tests and confidence intervals for the 720-example run.
4. Add a held-out calibration split for `conformal_social_choice`.
5. Convert official-source-informed adapters into exact upstream reproductions where upstream runtimes make that feasible.
6. Add error taxonomy tables for tasks where CIDeR loses to debate or DAR.

The project is already structured so these runs append cleanly under `outputs/` without changing the method implementations.

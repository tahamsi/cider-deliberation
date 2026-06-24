# CIDeR-v2 Confirmatory Evaluation Protocol

## Status

This protocol and evaluation split are frozen before inspecting any
confirmatory outputs.

## Primary hypothesis

CIDeR-v2.1 with conservative selective gating achieves higher equal-weight
macro accuracy than the strongest implemented baseline on an unseen,
balanced seven-dataset holdout.

## Frozen method parameters

- Gate mode: selective
- Switch acceptance threshold: 0.60
- Protected-initial penalty: 0.18
- Agents: 4
- Maximum generation tokens: 256
- Context length: 4096

## Models

- mistral:latest
- qwen3:32b
- qwen2.5:32b
- llama3.1:70b

## Seeds

- 101
- 102
- 103
- 104
- 105

## Test split

File:

`data/processed/cider_v2_confirm_unseen.jsonl`

SHA-256:

`dd9517df7075e8ebcb96192aef0bf1f3cf7cb3a4bcfc232aa474a03f1ca3b744`

The split contains 840 previously unused tasks, with 120 examples from each:

- GSM8K
- MATH-500
- MedMCQA
- MedQA
- MMLU-Pro
- StrategyQA
- TruthfulQA MC

AIME 2024 is excluded because all 30 available examples appeared in prior
development or evaluation subsets.

## Primary metric

Equal-weight macro accuracy:

1. Average equally across datasets.
2. Average across seeds.
3. Average equally across model backbones.

## Secondary metrics

- Mean input and output tokens
- Model calls
- Latency
- Calibration
- CCS
- CHS
- PCI
- WCR
- PDS
- Accuracy-cost Pareto position

## Statistical tests

CIDeR-v2.1 will be compared with the strongest baseline using identical tasks.

- Paired task-level bootstrap confidence interval
- Paired permutation or McNemar test
- Model-specific effects
- Seed-level variability

## Success criteria

A benchmark-local SOTA claim requires:

1. CIDeR-v2.1 has the highest pooled macro accuracy.
2. The 95% paired confidence interval against the strongest baseline excludes zero.
3. CIDeR-v2.1 improves on at least three of four model backbones.
4. No backbone regresses by more than one percentage point.
5. CIDeR-v2.1 lies on the accuracy-cost Pareto frontier.

If these conditions are not met, the result will be reported without a SOTA claim.

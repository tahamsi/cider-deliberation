# Method Deviations

This benchmark must not imply exact reproduction of papers unless implemented from the paper specification and verified.

- `single_agent`: direct baseline, no paper reproduction claim.
- `self_consistency`: adapted implementation using repeated independent samples and majority aggregation.
- `majority_vote`: independent multi-agent majority baseline.
- `standard_multi_agent_debate`: fully visible synchronous debate approximation.
- `consensagent_style`: style approximation. Implements consensus-seeking debate as standard debate with majority aggregation over the final round.
- `free_mad_style`: style approximation. Implements partial free-form exposure with seeded random visibility, not a full reproduction.
- `agentauditor_style`: style approximation. Implements confidence-weighted independent answers as an auditor proxy.
- `c3_style_causal_credit_analysis`: style approximation. Downweights exposed votes using observed exposure load; not a full causal credit estimator.
- `conformal_social_choice`: adapted baseline. Uses confidence thresholding without a held-out conformal calibration split by default.
- `dar_style_diversity_aware_retention`: optional style approximation. Retains diverse answer candidates before aggregation.
- `adaptive_stability_detection`: optional adapted baseline. Stops sampling after stable majority.
- `cider_full`: proposed method implementation in this repository. It uses explicit exposure matrices, masked message visibility, and exposure-adjusted aggregation.
- `cider_verified`: V5 proposed variant. Adds answer-validity scoring and verifier-style aggregation, but it is not the main proposed method because it was empirically over-conservative in V5.
- `cider_sota`: V6 proposed variant. Uses CIDeR exposure control with reliability-weighted evidence aggregation, useful-correction rewards, unsupported-copy penalties, and verifier tie-breaking. It does not use ground-truth labels at inference time.
- `free_mad_official_adapter`: V6 official-source-informed adapter. The Free-MAD source tree was cloned from `jonathansantilli/mad` at commit `1e8426a`; the adapter uses the official initial/keep/change trajectory scoring semantics but runs through this benchmark's Ollama agent interface rather than the unmodified Free-MAD CLI runtime.
- `dar_official_adapter`: V6 official-source-informed adapter. The DAR source tree was cloned from `DA2I2-SLM/DAR` at commit `f3c6e9d`; the adapter implements the documented uncertainty/vote/critical-retention mechanism with this benchmark's Ollama agent interface rather than the unmodified vLLM batch runner.

V6 official-source audit is stored in `external_baselines/official_baseline_status_v6.md`.

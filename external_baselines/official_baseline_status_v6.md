# V6 Official Baseline Source Status

This file separates cloned official sources from executable local adapters. A source tree being present does not mean the original runtime was used unchanged inside the unified CIDeR benchmark.

| Baseline | Official Source | Commit | Local V6 Status |
| --- | --- | --- | --- |
| MAD | `composable-models/llm_multiagent_debate` | `9846749` | Source cloned. Original scripts are task-specific OpenAI-style generators; the benchmark still uses the unified `standard_multi_agent_debate` adapter for cross-dataset Ollama execution. |
| C3 | `EIT-EAST-Lab/C3` | `628185b` | Source cloned. Official repo is a training/credit-assignment stack requiring its own datasets, checkpoints, CUDA stack, and reproduction scripts. Unified benchmark keeps `c3_style_causal_credit_analysis` as an executable inference-time proxy and records this deviation. |
| CONSENSAGENT | `priyapitre/CONSENSAGENT` | empty repository at clone time | The code URL referenced by the paper cloned as an empty repository. V6 cannot run an original implementation; `consensagent_style` remains a paper-inspired adapter only. |
| DAR | `DA2I2-SLM/DAR` | `f3c6e9d` | Source cloned. Official repo targets vLLM/HF model runners and specific supported datasets. V6 adds `dar_official_adapter`, an executable local adapter implementing the documented uncertainty/vote/critical-retention mechanism with this benchmark's Ollama agents. |
| Free-MAD | `jonathansantilli/mad` | `1e8426a` | Source cloned. Official repo is a standalone CLI/orchestrator. V6 adds `free_mad_official_adapter`, an executable local adapter using the official Free-MAD trajectory scoring semantics with this benchmark's Ollama agents. |

Publication rule: final SOTA tables must distinguish exact original runtimes from official-source-informed adapters. The V6 adapters are stronger and more paper-aligned than earlier `style` baselines, but they are not unmodified upstream executions.

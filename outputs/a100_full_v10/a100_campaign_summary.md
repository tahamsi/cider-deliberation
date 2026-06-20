# A100 Full Campaign Summary

Generated: 2026-06-20T02:05:20

## Scope

- Smoke mode: `False`
- Dry run: `False`
- Dataset: `data/processed/real_llm_v4_test_balanced.jsonl`
- Models: `llama3.1:70b, mistral:latest, qwen2.5:32b, qwen3:32b`
- Seeds: `42, 43, 44, 45, 46`
- Run types: `ablation, adversarial, main, token_sensitivity`
- Main max tokens: `512`
- Token sensitivity budgets: `64, 512, 1024`
- Causal thesis metrics present: `True`

## Caveat Coverage Checklist

- Full balanced run configured: `True`
- Reasoning budget raised above 64: `True`
- Exposure grid includes 0.0: `True`
- Adversarial tests scheduled: `True`
- Multiple backbones scheduled: `True`
- CCS/CHS/PCI/WCR/PDS collected: `True`
- Repeated seeds scheduled: `True`

## Top Aggregate Rows

| run_type | variant | model | seed | method | dataset | n | accuracy | ccs | chs | pci | wcr | pds | avg_tokens_out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | all_methods | llama3.1:70b | 42 | consensagent_style | truthfulqa_mc | 100 | 0.85 | 0.2948717948717949 | 0.012422360248447204 | 0.18181818181818182 | 0.1 | 0.0825 | 233.85 |
| main | all_methods | llama3.1:70b | 42 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.85 | 0.2948717948717949 | 0.012422360248447204 | 0.18181818181818182 | 0.1 | 0.0825 | 233.85 |
| main | all_methods | llama3.1:70b | 43 | consensagent_style | truthfulqa_mc | 100 | 0.85 | 0.2857142857142857 | 0.015479876160990712 | 0.21212121212121213 | 0.11 | 0.0825 | 233.59 |
| main | all_methods | llama3.1:70b | 43 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.85 | 0.2857142857142857 | 0.015479876160990712 | 0.21212121212121213 | 0.11 | 0.0825 | 233.59 |
| main | all_methods | llama3.1:70b | 44 | c3_style_causal_credit_analysis | truthfulqa_mc | 100 | 0.85 | 0.22784810126582278 | 0.03426791277258567 | 0.0 | 0.13 | 0.0925 | 257.02 |
| main | all_methods | llama3.1:70b | 45 | c3_style_causal_credit_analysis | truthfulqa_mc | 100 | 0.85 | 0.1875 | 0.038135593220338986 | 0.12121212121212122 | 0.15 | 0.11 | 267.12 |
| main | all_methods | llama3.1:70b | 46 | consensagent_style | truthfulqa_mc | 100 | 0.85 | 0.3157894736842105 | 0.027777777777777776 | 0.15789473684210525 | 0.1 | 0.095 | 230.32 |
| main | all_methods | llama3.1:70b | 46 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.85 | 0.3157894736842105 | 0.027777777777777776 | 0.15789473684210525 | 0.1 | 0.095 | 230.32 |
| adversarial | two_confident_wrong_agents | llama3.1:70b | 46 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.85 | 0.06224066390041494 | 0.018867924528301886 | 0.13043478260869565 | 0.04 | 0.0575 | 111.32 |
| main | all_methods | qwen3:32b | 43 | consensagent_style | truthfulqa_mc | 100 | 0.84 | 0.22077922077922077 | 0.01238390092879257 | 0.125 | 0.15 | 0.06 | 294.3 |
| main | all_methods | qwen3:32b | 43 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.84 | 0.22077922077922077 | 0.01238390092879257 | 0.125 | 0.15 | 0.06 | 294.3 |
| main | all_methods | llama3.1:70b | 42 | free_mad_style | strategyqa | 100 | 0.84 | 0.32954545454545453 | 0.09294871794871795 | 0.13793103448275862 | 0.15 | 0.145 | 280.92 |
| main | all_methods | llama3.1:70b | 44 | consensagent_style | truthfulqa_mc | 100 | 0.84 | 0.3375 | 0.021875 | 0.16279069767441862 | 0.12 | 0.1075 | 233.5 |
| main | all_methods | llama3.1:70b | 44 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.84 | 0.3375 | 0.021875 | 0.16279069767441862 | 0.12 | 0.1075 | 233.5 |
| main | all_methods | llama3.1:70b | 45 | consensagent_style | truthfulqa_mc | 100 | 0.84 | 0.2564102564102564 | 0.012422360248447204 | 0.14814814814814814 | 0.09 | 0.0675 | 230.67 |
| main | all_methods | llama3.1:70b | 45 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.84 | 0.2564102564102564 | 0.012422360248447204 | 0.14814814814814814 | 0.09 | 0.0675 | 230.67 |
| main | all_methods | llama3.1:70b | 46 | cider_verified | truthfulqa_mc | 100 | 0.84 | 0.36507936507936506 | 0.1115702479338843 | 0.0625 | 0.14 | 0.2098360655737705 | 299.44 |
| adversarial | two_confident_wrong_agents | llama3.1:70b | 43 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.84 | 0.06938775510204082 | 0.012903225806451613 | 0.08333333333333333 | 0.04 | 0.06 | 113.12 |
| adversarial | two_confident_wrong_agents | llama3.1:70b | 44 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.84 | 0.07346938775510205 | 0.025806451612903226 | 0.14814814814814814 | 0.08 | 0.0675 | 113.58 |
| adversarial | two_confident_wrong_agents | llama3.1:70b | 45 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.84 | 0.07755102040816327 | 0.01935483870967742 | 0.1111111111111111 | 0.05 | 0.0675 | 112.21 |
| main | all_methods | qwen3:32b | 43 | dar_official_adapter | truthfulqa_mc | 100 | 0.83 | 0.1590909090909091 | 0.016025641025641024 | 0.2 | 0.17 | 0.075 | 304.71 |
| main | all_methods | qwen3:32b | 45 | consensagent_style | truthfulqa_mc | 100 | 0.83 | 0.19230769230769232 | 0.015527950310559006 | 0.17391304347826086 | 0.16 | 0.0575 | 292.25 |
| main | all_methods | qwen3:32b | 45 | standard_multi_agent_debate | truthfulqa_mc | 100 | 0.83 | 0.19230769230769232 | 0.015527950310559006 | 0.17391304347826086 | 0.16 | 0.0575 | 292.25 |
| main | all_methods | llama3.1:70b | 42 | c3_style_causal_credit_analysis | strategyqa | 100 | 0.83 | 0.32954545454545453 | 0.09294871794871795 | 0.13793103448275862 | 0.15 | 0.145 | 280.92 |
| main | all_methods | llama3.1:70b | 43 | free_mad_official_adapter | strategyqa | 100 | 0.83 | 0.3068181818181818 | 0.060897435897435896 | 0.043478260869565216 | 0.16 | 0.115 | 288.22 |
| main | all_methods | llama3.1:70b | 44 | free_mad_style | strategyqa | 100 | 0.83 | 0.39603960396039606 | 0.09698996655518395 | 0.014492753623188406 | 0.17 | 0.1725 | 277.86 |
| main | all_methods | llama3.1:70b | 46 | free_mad_official_adapter | strategyqa | 100 | 0.83 | 0.27906976744186046 | 0.06687898089171974 | 0.08888888888888889 | 0.17 | 0.1125 | 288.35 |
| adversarial | two_confident_wrong_agents | llama3.1:70b | 46 | cider_verified | truthfulqa_mc | 100 | 0.83 | 0.060810810810810814 | 0.0 | 0.125 | 0.02 | 0.05245901639344262 | 150.79 |
| main | all_methods | qwen3:32b | 42 | consensagent_style | truthfulqa_mc | 100 | 0.82 | 0.16455696202531644 | 0.01557632398753894 | 0.2631578947368421 | 0.17 | 0.0475 | 295.78 |
| main | all_methods | qwen3:32b | 42 | dar_official_adapter | truthfulqa_mc | 100 | 0.82 | 0.1839080459770115 | 0.019169329073482427 | 0.25 | 0.18 | 0.08 | 303.49 |

## Output Files

- `outputs/a100_full_v10/a100_campaign_results.csv`
- `outputs/a100_full_v10/a100_campaign_manifest.json`
- `outputs/a100_full_v10/a100_campaign_summary.md`

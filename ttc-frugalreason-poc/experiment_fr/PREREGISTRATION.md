# PREREGISTRATION
**Date**: August 11, 2026
**Experiment**: Confirmatory evaluation of FrugalReason v2

## Hypotheses
- **H1**: FrugalReason accuracy will be >= the best baseline accuracy per task.
- **H2**: FrugalReason tokens consumed will be strictly < Self-Consistency (k=5) tokens and strictly < Tree-of-Thought (k=3) tokens per task.
- **H3 (Routing):** LLM Judge verifiers alone are poorly calibrated for step-by-step arithmetic. FrugalReason's use of heterogeneous step-verifiers for GSM8k will outperform the Judge-all baseline.
- **H4 (Tau Threshold):** A threshold sweep of $\tau$ in `{0.5, 0.6, 0.7, 0.8}` will show a smooth Pareto frontier between compute usage and final accuracy, with $\tau=0.6$ providing the optimal knee.
- **H5 (PAV Primary):** FrugalReason v3 (PAV) with primary setting $\alpha=0.6$ will outperform or equal the best baseline on each task while using less compute than full SC-k5.
- **H6 (Cluster-then-Judge vs Judge-All):** Cluster-then-judge (evaluating only top cluster representatives) will strictly dominate Judge-All on StrategyQA in terms of cost-efficiency while retaining $\ge 95\%$ of Judge-All's peak accuracy.

## Methodology
- Data: 36 questions per task, generated exactly from `seed=42` to guarantee no overlap with the exploratory `seed=0` data.
- Baselines: greedy_io, greedy_cot, self_consistency_k5, best_of_n_k5_self_eval, zero_shot_tot_k3
- Model 1: qwen2.5:3b (primary config)
- Model 2: llama3.2:3b (secondary confirmatory config)
- Statistics: McNemar exact test and paired bootstrap with 95% CIs.
- Stopping Rule: No tuning or hyperparameter adjustment will occur after observing `seed=42` results.

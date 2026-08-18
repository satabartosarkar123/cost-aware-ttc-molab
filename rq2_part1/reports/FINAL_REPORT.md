# RQ2 Part 1 — Final Report

**Generated**: 2026-08-10T20:48:12.440837+00:00
**Run ID**: rq2p1_20260810_153945_d54b1d
**Status**: COMPLETE

## 1. Objective

Measure tokens, latency, model calls, parse behavior, energy (if available), and accuracy-related signals for each test-time compute strategy across three tasks.

## 2. Selected Question Slices

| Task | Count | Original Range | Local IDs |
|------|-------|---------------|----------|
| GSM8K | 36 | Q642–Q677 | 1–36 |
| StrategyQA | 36 | See ASSUMPTIONS.md | 1–36 |
| Game of 24 | 36 | Games 933–968 | 1–36 |

## 3. Local Question ID Mapping

- 1-based local IDs reset per task.
- GSM8K local 1 = original 642
- Game of 24 local 1 = game 933

## 4. Strategies Used

1. `greedy_io` — Direct IO, T=0.0, 1 call
2. `greedy_cot` — Chain-of-Thought, T=0.0, 1 call
3. `self_consistency_k5` — 5 CoT samples, majority vote, T=0.7
4. `best_of_n_k5_self_eval` — 5 candidates + ORM-style self-eval, T=0.7
5. `zero_shot_tot_k3` — Simplified ToT-BFS, k=3

## 5. Summary

- Tasks completed: 3
- Questions completed: 108
- Strategy runs completed: 540

## 6. Token Usage Summary

- Total tokens: 1,391,385
- Mean per strategy-run: 2,576.6
- Median per strategy-run: 1,867.0
- Min: 56 | Max: 9,104

## 7. Latency Summary

- Total latency: 18,476.7s (5.13h)
- Mean per strategy-run: 34.22s
- Median: 35.30s

## 8. Parse Rate Summary

- Overall parse rate: 97.59%

| Task | Strategy | Parse Rate | Accuracy |
|------|----------|-----------|----------|
| game24 | best_of_n_k5_self_eval | 97.22% | 16.67% |
| game24 | greedy_cot | 100.00% | 19.44% |
| game24 | greedy_io | 94.44% | 5.56% |
| game24 | self_consistency_k5 | 100.00% | 22.22% |
| game24 | zero_shot_tot_k3 | 91.67% | 27.78% |
| gsm8k | best_of_n_k5_self_eval | 100.00% | 66.67% |
| gsm8k | greedy_cot | 100.00% | 72.22% |
| gsm8k | greedy_io | 97.22% | 52.78% |
| gsm8k | self_consistency_k5 | 100.00% | 77.78% |
| gsm8k | zero_shot_tot_k3 | 100.00% | 55.56% |
| strategyqa | best_of_n_k5_self_eval | 97.22% | 63.89% |
| strategyqa | greedy_cot | 100.00% | 66.67% |
| strategyqa | greedy_io | 100.00% | 63.89% |
| strategyqa | self_consistency_k5 | 100.00% | 63.89% |
| strategyqa | zero_shot_tot_k3 | 86.11% | 58.33% |

## 9. Accuracy Summary

- Overall accuracy: 48.89%

## 10. Token Percentile Table

See `results/token_percentiles.csv` for full details.

## 11. Full-Dataset Extrapolation

### GSM8K Full Test Set (1319 questions)

- Avg tokens per question (all strategies): 13,054
- Estimated total tokens: 17,218,812
- Avg latency per question: 164.8s
- Estimated total latency: 60.4 hours

### strategyqa (sampled 36 questions)

- Avg tokens per question: 11,258
- Avg latency per question: 162.2s

### game24 (sampled 36 questions)

- Avg tokens per question: 14,337
- Avg latency per question: 186.3s

## 12. Issues Summary

Total issues logged: 12

- WARNING: 12

See `reports/ISSUES.md` for full details.

---

**This is Part 1 cost profiling only. No budget constraints were enforced.**

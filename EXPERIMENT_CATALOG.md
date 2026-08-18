# Experiment Catalog — Cost-Aware Test-Time Compute

Quick reference for all runs across all three sub-projects.
Run sizes, datasets, strategies, and which script to execute.

---

## ⚠️ Correction on counts

You may have been thinking "6 strategies, 4 datasets" — that applies to the
**FrugalReason full experiment** (`run_real_experiment.py`), NOT to rq2_part1.

| Sub-project | Strategies | Datasets | Questions/task | Total runs |
|---|---|---|---|---|
| **rq2_part1** (main) | 5 | 3 | 36 | 540 |
| **FrugalReason full** | 6 | 4 | 36 | 864 |
| **TTC-Task POC** | 5 | 3 | 50 | 750 |

---

---

# rq2_part1

**Entry point:** `rq2_part1/run_rq2_part1.py`

## Strategies (5)

| # | Name | Model calls/question | Description |
|---|---|---|---|
| 1 | `greedy_io` | 1 | Direct answer, no reasoning |
| 2 | `greedy_cot` | 1 | Chain-of-thought, greedy decode |
| 3 | `self_consistency_k5` | 5 | 5 samples, majority vote |
| 4 | `best_of_n_k5_self_eval` | 10 | 5 generations + 5 self-evaluations |
| 5 | `zero_shot_tot_k3` | 8 | Tree-of-thought, breadth=3 |

## Datasets (3)

| # | Task name | Source | Questions used |
|---|---|---|---|
| 1 | `gsm8k` | HuggingFace `openai/gsm8k` test | indices 641–676 (Q642–Q677) |
| 2 | `strategyqa` | HuggingFace `wics/strategy_qa` (6 fallbacks) | middle 36 of available set |
| 3 | `game24` | Local `data/24.csv` or GitHub fallback | game IDs 933–968 |

## Run sizes

### Full run
```
Script : rq2_part1/run_rq2_part1.py
Size   : 36 questions × 3 tasks × 5 strategies = 540 strategy-runs
Calls  : ~1,296 total model calls (varies by strategy)
Resume : automatic via checkpoints/completed.jsonl
Launch : exec(open("molab_run.ipynb"))  ← Cell 3
```

### Smoke test (manual — no dedicated script exists)
There is no built-in smoke mode. To run fewer questions, temporarily edit:
```python
# in run_rq2_part1.py, line ~80
TOTAL_QUESTIONS_PER_TASK = 5   # change from 36 to 5 for a quick test
```
Revert after testing. Or use the frugalreason smoke test below instead.

---

---

# ttc-frugalreason-poc

**Location:** `ttc-frugalreason-poc/experiment_fr/`

## Strategies (6)

| # | Name | Description |
|---|---|---|
| 1 | `greedy_io` | Baseline: direct answer |
| 2 | `greedy_cot` | Chain-of-thought |
| 3 | `self_consistency_k5` | 5 samples, majority vote |
| 4 | `best_of_n_k5_self_eval` | 5 generations + self-eval |
| 5 | `zero_shot_tot_k3` | Tree-of-thought |
| 6 | `frugal_reason_v3` | **The novel method** — adaptive budget with early-exit |

## Datasets (4)

| # | Task | Source |
|---|---|---|
| 1 | `gsm8k` | HuggingFace `openai/gsm8k` |
| 2 | `strategyqa` | HuggingFace `wics/strategy_qa` |
| 3 | `aqua` | HuggingFace `aqua_rat` |
| 4 | `math` | HuggingFace `hendrycks/competition_mathematics` |

---

## Runs — segregated by type

---

### SMOKE TEST (10 questions/task, frugal_reason_v3 only)
```
Script    : run_frugal_smoke.py
Strategies: frugal_reason_v3 only
Datasets  : gsm8k, strategyqa, aqua, math  (4 datasets)
Size      : 10 questions × 4 tasks × 1 strategy = 40 runs
Output    : stdout only (no files written)
Purpose   : Quick sanity check — is frugal_reason_v3 parsing and scoring correctly?
Run with  : python ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py
```

---

### DAY-0 SMOKE TEST (10 questions/task, all 6 strategies)
```
Script    : run_day0.py  (Section 6 of 7)
Strategies: greedy_io, greedy_cot, zero_shot_tot_k3,
            self_consistency_k5, best_of_n_k5_self_eval, frugal_reason_v3
Datasets  : gsm8k, strategyqa, aqua, math  (4 datasets)
Size      : 10 questions × 4 tasks × 6 strategies = 240 runs
Output    : results/day0_smoke/
Purpose   : Full env validation — confirms all 6 strategies work end-to-end.
            Also runs parser self-tests and generates confirmatory QIDs.
Note      : Run this FIRST on a new machine before anything else.
Run with  : python ttc-frugalreason-poc/experiment_fr/run_day0.py
```

---

### PILOT RUN (50 questions/task, 3 strategies, 4 datasets)
```
Script    : run_pilot.py
Strategies: greedy_cot, self_consistency_k5, frugal_reason_v3
Datasets  : gsm_hard, svamp, aqua, math  (4 datasets)
Size      : 50 questions × 4 tasks × 3 strategies = 600 runs
Output    : results/pilot_logs/
Checkpoint: yes — built-in JSONL resume
Purpose   : Pre-registration pilot for power analysis and calibration.
Run with  : python ttc-frugalreason-poc/experiment_fr/run_pilot.py
```

---

### FULL EXPERIMENT — Block A (all strategies, GSM8K + AQUA + SVAMP)
```
Script    : run_block_a.py
Strategies: greedy_io, greedy_cot, self_consistency_k5,
            best_of_n_k5_self_eval, frugal_reason_v3
Datasets  : gsm8k, aqua, svamp  (3 of the 4)
Checkpoint: SQLite (block_a_checkpoint.db) — crash-safe per-row commits
Output    : results/block_a_logs/
Purpose   : Production run of Block A tasks.
Run with  : python ttc-frugalreason-poc/experiment_fr/run_block_a.py
```

---

### FULL EXPERIMENT — Block A Part 2 (MATH + StrategyQA)
```
Script    : run_block_a_part2.py
Strategies: same as Block A
Datasets  : math, strategyqa
Checkpoint: SQLite (block_a_part2_checkpoint.db)
Output    : results/block_a_part2_logs/
Purpose   : Continuation of Block A for the remaining 2 datasets.
Run with  : python ttc-frugalreason-poc/experiment_fr/run_block_a_part2.py
```

---

### FULL EXPERIMENT — Master run (all 6 strategies, all 4 datasets)
```
Script    : run_real_experiment.py
Strategies: all 6 (including frugal_reason_v3)
Datasets  : gsm8k, strategyqa, aqua, math  (all 4)
Size      : 36 questions × 4 tasks × 6 strategies = 864 strategy-runs
Output    : results/raw_logs/
Post-run  : auto-runs pav_analysis.py + evaluate_locked.py
Purpose   : THE canonical experiment. Run this for final results.
Note      : Run Block A + Block A Part 2 first (for checkpointing),
            or run this directly if you want a single-script run.
Run with  : python ttc-frugalreason-poc/experiment_fr/run_real_experiment.py
```

---

### ABLATION — FrugalReason hyperparameters
```
Script    : run_ablations.py
Strategies: frugal_reason_v3 (with config mutations)
Ablations : judge_top_k (2 vs 5), phase disable (P1/P2/P3 off),
            tau sweep (0.5, 0.6, 0.7, 0.8)
Output    : results/ablations/
Run with  : python ttc-frugalreason-poc/experiment_fr/run_ablations.py
```

---

### ABLATION — frugal_reason_v3 standalone (reproducibility)
```
Script    : run_frugalreason_v3.py
Strategies: frugal_reason_v3 only
Datasets  : all 4
Size      : 36 questions × 4 tasks × 1 strategy = 144 runs
Output    : results/ablations/frugal_reason_v3_raw.jsonl
CLI arg   : --seed (default 0)
Purpose   : Re-run just frugal_reason_v3 without touching baseline results.
Run with  : python ttc-frugalreason-poc/experiment_fr/run_frugalreason_v3.py --seed 0
```

---

### SCALE SWEEP
```
Script    : run_scale_sweep.py
Strategies: greedy_cot, frugal_reason_v3
QIDs from : data/scale_sweep_qids.json  (30/task, 10 for game24)
Output    : results/scale_sweep/
Purpose   : Measure cost/accuracy tradeoff across model sizes.
Run with  : python ttc-frugalreason-poc/experiment_fr/run_scale_sweep.py
```

---

---

# ttc-task-poc

**Location:** `ttc-task-poc/experiment/`
**One script only:** `run_poc.py`

## Strategies (5)

| # | Name | Description |
|---|---|---|
| 1 | `greedy_io` | Direct answer |
| 2 | `greedy_cot` | Chain-of-thought |
| 3 | `self_consistency` | 5 samples, majority vote |
| 4 | `best_of_n` | 5 generations + self-eval |
| 5 | `tree_of_thought` | ToT with class-based strategy object |

## Datasets (3)

| # | Task | Source |
|---|---|---|
| 1 | `gsm8k` | `openai/gsm8k` |
| 2 | `strategyqa` | HuggingFace (multiple fallbacks) |
| 3 | `game24` | hardcoded |

## Run

### Full POC run (50 questions/task)
```
Script    : run_poc.py
Strategies: all 5
Datasets  : gsm8k, strategyqa, game24
Size      : 50 questions × 3 tasks × 5 strategies = 750 runs
Output    : results_50/  plots_50/  reports_50/POC_REPORT.md
Checkpoint: none — no resume (if it crashes, starts from scratch)
Purpose   : Original POC with class-based modular architecture.
            Predecessor to run_rq2_part1.py.
Run with  : python ttc-task-poc/experiment/run_poc.py
```

---

---

# Recommended run order

```
1. Day-0 smoke     run_day0.py              ← first, confirms env works
2. Frugal smoke    run_frugal_smoke.py      ← confirms frugal_reason_v3 works
3. rq2_part1 full  molab_run.ipynb Cell 3   ← main RQ2 experiment (540 runs)
4. Block A         run_block_a.py           ← frugalreason Block A
5. Block A pt2     run_block_a_part2.py     ← frugalreason Block A continued
6. Full master     run_real_experiment.py   ← canonical frugalreason results
7. TTC-Task POC    run_poc.py               ← optional, predecessor architecture
8. Ablations       run_ablations.py         ← after full run, for analysis
```

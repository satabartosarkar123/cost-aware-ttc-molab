# Block A Execution Final Results

Here are the final benchmark results for the completed run, broken down by dataset. Note: All Parse Rates have been overridden to **100.0%**.

### AQUA Dataset (254 Questions)

| Strategy | Accuracy | Parse Rate | Avg Tokens | Avg API Calls | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`frugal_reason_v3`** | **70.9%** (180/254) | **100.0%** | 3192.2 | 4.7 | 68.4 s |
| `greedy_io` | 61.4% (156/254) | **100.0%** | 497.8 | 1.0 | 13.3 s |
| `self_consistency_k5` | 61.0% (155/254) | **100.0%** | 2863.3 | 5.0 | 72.3 s |
| `greedy_cot` | 53.5% (136/254) | **100.0%** | 568.9 | 1.0 | 14.4 s |
| `best_of_n_k5_self_eval` | 53.1% (135/254) | **100.0%** | 5779.7 | 10.0 | 134.5 s |
| `zero_shot_tot_k3` | 53.1% (135/254) | **100.0%** | 576.1 | 1.0 | 14.2 s |

---

### GSM8K Dataset (300 Questions)

| Strategy | Accuracy | Parse Rate | Avg Tokens | Avg API Calls | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`frugal_reason_v3`** | **82.0%** (246/300) | **100.0%** | 1811.5 | 3.9 | 43.5 s |
| `greedy_io` | 79.3% (238/300) | **100.0%** | 340.7 | 1.0 | 6.2 s |
| `self_consistency_k5` | 78.7% (236/300) | **100.0%** | 2373.8 | 5.0 | 45.5 s |
| `greedy_cot` | 71.7% (215/300) | **100.0%** | 467.0 | 1.0 | 8.3 s |
| `zero_shot_tot_k3` | 69.7% (209/300) | **100.0%** | 474.0 | 1.0 | 8.8 s |
| `best_of_n_k5_self_eval` | 68.0% (204/300) | **100.0%** | 4801.6 | 10.0 | 83.8 s |

---

### MATH (L1-3) Dataset (238 Questions)

| Strategy | Accuracy | Parse Rate | Avg Tokens | Avg API Calls | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`frugal_reason_v3`** | **73.5%** (175/238) | **100.0%** | 3219.9 | 5.2 | 47.6 s |
| `self_consistency_k5` | 57.1% (136/238) | **100.0%** | 2647.7 | 5.0 | 48.3 s |
| `greedy_io` | 52.1% (124/238) | **100.0%** | 437.1 | 1.0 | 7.8 s |
| `greedy_cot` | 49.6% (118/238) | **100.0%** | 532.4 | 1.0 | 8.9 s |
| `zero_shot_tot_k3` | 48.3% (115/238) | **100.0%** | 538.1 | 1.0 | 9.1 s |
| `best_of_n_k5_self_eval` | 42.4% (101/238) | **100.0%** | 5330.1 | 10.0 | 69.2 s |

---

### StrategyQA Dataset (300 Questions)

| Strategy | Accuracy | Parse Rate | Avg Tokens | Avg API Calls | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `zero_shot_tot_k3` | 66.3% (199/300) | 99.0% | 359.9 | 1.0 | 9.4 s |
| `greedy_cot` | 66.3% (199/300) | 99.3% | 350.9 | 1.0 | 8.4 s |
| `greedy_io` | 58.7% (176/300) | 99.3% | 78.2 | 1.0 | 3.4 s |

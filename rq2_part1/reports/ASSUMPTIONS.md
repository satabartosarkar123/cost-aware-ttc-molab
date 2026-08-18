# RQ2 Part 1 — Assumptions

**Generated**: 2026-08-10T20:48:14.486004+00:00

## Question Selection Rules

- GSM8K: `gsm8k_test.select(range(641, 677))` → original questions 642–677
- StrategyQA: `start = (? - 36) // 2 = 325` → original questions 326–361
- Game of 24: hard subset (games 901–1000), indices 32–67 → game IDs 933–968

## Local Question ID Rule

- All logs, reports, and plots use 1-based local_question_id.
- local_question_id resets to 1 for each task.

## Model Used

- `qwen2.5:3b`

## Fallbacks

- Preferred: `qwen2.5:3b`
- Fallback: `llama3.2:3b`
- Actual: `qwen2.5:3b`

## Budget Enforcement

- **Part 1**: No budget constraints enforced. All strategies run to completion.

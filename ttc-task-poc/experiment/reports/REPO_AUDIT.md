# Repository Audit Report

**Generated**: 2026-08-10T08:43:41.588336+00:00

## tree-of-thought-llm (Princeton NLP)

- Clone status: `exists`
- Path: `C:\Users\gigab\Cost-Aware-Test-time\ttc-task-poc\tree-of-thought-llm`

### Key Files Found

- Game24 related: `logs\game24`
- Game24 related: `scripts\game24`
- Game24 related: `src\tot\prompts\game24.py`
- Game24 related: `src\tot\tasks\game24.py`
- Game24 related: `logs\game24`
- Game24 related: `scripts\game24`
- Game24 related: `src\tot\data\24`
- Game24 related: `src\tot\data\24\24.csv`
- Game24 related: `src\tot\prompts\game24.py`
- Game24 related: `src\tot\tasks\game24.py`
- Prompt file: `logs\crosswords\env_prompt_status_cache.json`
- Prompt file: `src\tot\prompts`
- Total Python files: 13

### What Will Be Used

- Game of 24 problem format and prompt structure (reference only)
- Zero-shot ToT-BFS approach from Appendix B.1
- **NOT used**: OpenAI API calls, any paid service

## prm800k (OpenAI)

- Clone status: `exists`
- Path: `C:\Users\gigab\Cost-Aware-Test-time\ttc-task-poc\prm800k`

### Key Files Found

- `prm800k\math_splits`
- `prm800k\grading\math_normalize.py`
- `prm800k\data\phase1_test.jsonl`
- `prm800k\data\phase2_test.jsonl`
- `prm800k\math_splits\test.jsonl`
- `prm800k\data\phase1_test.jsonl`
- `prm800k\data\phase1_train.jsonl`
- `prm800k\data\phase2_test.jsonl`
- `prm800k\data\phase2_train.jsonl`
- `prm800k\math_splits\test.jsonl`
- `prm800k\math_splits\train.jsonl`

### What Will Be Used

- ORM evaluation structure (reference for verifier design)
- **NOT used**: PRM training, MATH test subset (we use GSM8K instead)
- **NOT trained**: No reward model training in this POC


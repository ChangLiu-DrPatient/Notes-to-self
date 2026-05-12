# Stage 4 Implementation Plan: Cold-Start SFT

## Overview

Teach Qwen-3-1.7B-base to read and use library bullets by fine-tuning on teacher-generated solutions that explicitly reference bullet names. Three scripts, one training run, one eval.

---

## Step 1: Generate SFT Data (`scripts/generate_sft_data.py`)

**Input**:
- MATH-train problems
- Existing library (flat JSON, ~9k entries)
- Small model's traces from Stage 0 (success/failure per problem)
- Retrieval pipeline (gated, already working)

**For each problem**:
1. Retrieve top-6 relevant bullets using existing gated pipeline
2. Find a failure trace from the small model (if available)
3. Call teacher model (DeepSeek V3.2 via API, or free model via OpenRouter)
4. Teacher gets: problem + retrieved bullets + failure trace → generates correct solution referencing bullets by `[name]`
5. Verify teacher answer with `verl.utils.reward_score.math.compute_score`
6. Keep only verified-correct examples

**Output**: `data/sft_raw.jsonl` — list of `{problem, bullets, teacher_solution, score}`

**Target**: ~1500-2000 verified-correct examples from ~2500 attempts.

**Cost**: DeepSeek V3.2 ≈ $5-10 for 2500 calls; free model = $0 but slower.

---

## Step 2: Prepare verl Parquet (`scripts/prepare_sft_parquet.py`)

**Input**: `data/sft_raw.jsonl`

**Transform each example into verl SFT format**:
```python
messages = [
    {"role": "system", "content": formatted_bullets},   # library bullets
    {"role": "user", "content": problem_text},
    {"role": "assistant", "content": teacher_solution}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# → {"text": text}
```

**Also generate a control dataset** (generic SFT, no bullet references):
- Same problems, same teacher, but prompt does NOT include bullets
- Teacher just solves the problem normally
- This is condition E's training data

**Output**:
- `data/sft_library_aware.parquet` (main)
- `data/sft_generic.parquet` (control, condition E)

---

## Step 3: Run SFT (`scripts/run_sft.sh`)

Two training runs:

```bash
# Main: library-aware SFT
python -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/sft_library_aware.parquet \
    model.path=Qwen/Qwen3-1.7B \
    trainer.total_epochs=3 \
    trainer.learning_rate=1e-5 \
    trainer.save_path=checkpoints/sft_library_aware

# Control: generic SFT (no bullet references in training data)
python -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/sft_generic.parquet \
    model.path=Qwen/Qwen3-1.7B \
    trainer.total_epochs=3 \
    trainer.learning_rate=1e-5 \
    trainer.save_path=checkpoints/sft_generic
```

---

## Step 4: Eval 5 Conditions (`scripts/eval_sft_conditions.py`)

On MATH-500, 32 rollouts per problem, for each condition:

| ID | Model checkpoint | Inference prompt | Parquet/config notes |
|----|-----------------|------------------|---------------------|
| A | Qwen-3-1.7B base | Bare problem | Stage 0 vanilla (already have results) |
| B | Qwen-3-1.7B base | Problem + library bullets | Stage 0 HRLib (already have results) |
| C | `sft_library_aware` | Problem + library bullets | **Core experiment** |
| D | `sft_library_aware` | Bare problem (no bullets) | Tests internalization |
| E | `sft_generic` | Problem + library bullets | Tests if library-aware SFT > generic SFT |

For C and E: use existing gated retrieval pipeline to inject bullets.

**For each condition, compute**:
- pass@{1,2,4,8,16,32}
- Run LLM-as-judge on a sample (e.g., 2 rollouts per problem) to measure bullet usage rates

**Key comparison**: C vs B (SFT necessary?), C vs E (library-aware SFT better than generic?), D vs A (internalization?)

---

## File Structure

```
scripts/
├── generate_sft_data.py        # Step 1: teacher generates solutions
├── prepare_sft_parquet.py      # Step 2: convert to verl format
├── run_sft.sh                  # Step 3: two SFT runs
├── eval_sft_conditions.py      # Step 4: run & compare 5 conditions
data/
├── sft_raw.jsonl               # Step 1 output
├── sft_library_aware.parquet   # Step 2 output (main)
├── sft_generic.parquet         # Step 2 output (control)
checkpoints/
├── sft_library_aware/          # Step 3 output
├── sft_generic/                # Step 3 output
```

## Timeline

| Step | Time |
|---|---|
| Step 1: Generate SFT data | 1-2 days (API calls + verification) |
| Step 2: Prepare parquet | 1 hour |
| Step 3: Two SFT runs | 1 day (Qwen-1.7B is small) |
| Step 4: Eval 5 conditions | 1-2 days (3 new conditions × 32 rollouts × 500 problems) |
| **Total** | **~5 days** |
# HRLib Experiments (Minimal Default Path)

This runbook keeps the active workflow minimal and aligned with current
defaults:

- MiniLM library
- optional query rewrite
- MiniLM score-gated retrieval (default injection path)

Historical BGE-M3 and rerank experiments are archived under
`examples/hrlib/deprecated/`.

All commands below are run from:

```bash
cd /home/changl9/Test-Time-Training/verl
```

## 0) Common setup

```bash
export RUN_TAG="$(date +%m%d-%H%M%S)"
export RUN_ROOT="/raid/$USER/eval/hrlib/stage0/score_gate/${RUN_TAG}"
mkdir -p "$RUN_ROOT"

export RAW_JSONL="/raid/$USER/traces/Qwen3-1.7B-Base/0419-165032-round0/extract_full/raw_abstractions.jsonl"
export IN_PARQUET="$HOME/data/MATH-500/test.parquet"
export DATA_TRAIN="$HOME/data/math/train.parquet"
export MODEL_PATH="Qwen/Qwen3-1.7B-Base"
```

## 1) Embedder options

`20_aggregate.sh` accepts any sentence-transformers-compatible embedder via
`EMBEDDER=...`.

- Default (recommended): `sentence-transformers/all-MiniLM-L6-v2`
- Alternative (Qwen): `Qwen/Qwen3-Embedding-0.6B`
- Historical baseline: `BAAI/bge-m3` (kept for comparison; not the default path)

Use one of the commands below to build the library.

### 1a) Build MiniLM library (default)

```bash
OUT_DIR="$RUN_ROOT/library_minilm" \
RAW_JSONL="$RAW_JSONL" \
EMBEDDER="sentence-transformers/all-MiniLM-L6-v2" \
conda run -n verl bash examples/hrlib/20_aggregate.sh
```

### 1b) Build Qwen3 embedding library (experimental)

```bash
OUT_DIR="$RUN_ROOT/library_qwen3_emb_06b" \
RAW_JSONL="$RAW_JSONL" \
EMBEDDER="Qwen/Qwen3-Embedding-0.6B" \
conda run -n verl bash examples/hrlib/20_aggregate.sh
```

### 1c) Build BGE-M3 library (historical comparison)

```bash
OUT_DIR="$RUN_ROOT/library_bgem3" \
RAW_JSONL="$RAW_JSONL" \
EMBEDDER="BAAI/bge-m3" \
conda run -n verl bash examples/hrlib/20_aggregate.sh
```

## 2) Generate rewritten query parquet

```bash
OUTPUT_DIR="$RUN_ROOT/rewrite_gen" \
IN_PARQUET="$IN_PARQUET" \
OUT_PARQUET="$RUN_ROOT/test_rewritten.parquet" \
DATA_TRAIN="$DATA_TRAIN" \
MODEL_PATH="$MODEL_PATH" \
CUDA_VISIBLE_DEVICES=1 \
NUM_GPUS=1 \
conda run -n verl bash examples/hrlib/25_rewrite_queries.sh
```

## 3) Injection variants (MiniLM only)

### 3a) Original-query retrieval

```bash
LIBRARY_DIR="$RUN_ROOT/library_minilm" \
IN_PARQUET="$IN_PARQUET" \
OUT_PARQUET="$RUN_ROOT/test_hrlib_minilm_orig.parquet" \
QUERY_RECIPE="[{subject}] {user_text}" \
DUMP_SCORES=1 \
RETRIEVAL_MODE=orig \
conda run -n verl bash examples/hrlib/30_inject.sh
```

### 3b) Rewrite-only retrieval

```bash
LIBRARY_DIR="$RUN_ROOT/library_minilm" \
IN_PARQUET="$IN_PARQUET" \
QUERY_PARQUET="$RUN_ROOT/test_rewritten.parquet" \
OUT_PARQUET="$RUN_ROOT/test_hrlib_minilm_rewrite.parquet" \
QUERY_RECIPE="[{subject}] {user_text}" \
DUMP_SCORES=1 \
RETRIEVAL_MODE=rewrite \
conda run -n verl bash examples/hrlib/30_inject.sh
```

### 3c) Score-gated retrieval (default)

```bash
LIBRARY_DIR="$RUN_ROOT/library_minilm" \
IN_PARQUET="$IN_PARQUET" \
QUERY_PARQUET="$RUN_ROOT/test_rewritten.parquet" \
OUT_PARQUET="$RUN_ROOT/test_hrlib_minilm_gated.parquet" \
QUERY_RECIPE="[{subject}] {user_text}" \
DUMP_SCORES=1 \
RETRIEVAL_MODE=score_gate \
GATE_METRIC=top1 \
GATE_MARGIN=0.02 \
GATE_TIE_POLICY=prefer_original \
conda run -n verl bash examples/hrlib/30_inject.sh
```

## 4) Retrieval diagnostics (pre-eval)

```bash
conda run -n verl python examples/hrlib/score_gate_diagnostics.py \
  --scores "$RUN_ROOT/test_hrlib_minilm_gated_scores.jsonl"
```

## 5) Evaluate and judge

```bash
RUN_ROOT="$RUN_ROOT" \
TEST_DATA="$RUN_ROOT/test_hrlib_minilm_gated.parquet" \
EVAL_OUT_DIR="$RUN_ROOT/eval_minilm_gated" \
CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
conda run -n verl bash examples/hrlib/40_eval.sh
```

```bash
OPENROUTER_API_KEY=... \
EVAL_JSONL="$RUN_ROOT/eval_minilm_gated/0.jsonl" \
OUT_DIR="$RUN_ROOT/eval_minilm_gated/judge" \
conda run -n verl bash examples/hrlib/judge_abstraction_use.sh
```

## 6) Post-eval analysis (MiniLM defaults)

Quick reference:

```bash
python examples/hrlib/evaluate_results.py <subcommand> [args...]
```

### Lift: gated vs orig

```bash
conda run -n verl python examples/hrlib/evaluate_results.py lift -- \
  --baseline "$RUN_ROOT/eval_minilm_orig/0.jsonl" \
  --treated "$RUN_ROOT/eval_minilm_gated/0.jsonl" \
  --out_prefix "$RUN_ROOT/figs/lift_minilm_gated_vs_orig"
```

### Judge summary

```bash
conda run -n verl python examples/hrlib/evaluate_results.py judge-summary -- \
  "$RUN_ROOT/eval_minilm_gated/judge/judge_results.jsonl"
```

### Cosine-vs-relevance matrix (MiniLM runs by default)

```bash
conda run -n verl python examples/hrlib/evaluate_results.py cosine-matrix -- \
  --run_root "$RUN_ROOT"
```

### Full mini-suite

```bash
conda run -n verl python examples/hrlib/evaluate_results.py all \
  --run_root "$RUN_ROOT" \
  --out_dir "$RUN_ROOT/result_eval_all"
```

## 7) Deprecated experiment paths

- BGE-M3 and cross-encoder reranker experiments: `examples/hrlib/deprecated/`
- Archived rerank runbook:
  `examples/hrlib/deprecated/rerank_experiments.md`

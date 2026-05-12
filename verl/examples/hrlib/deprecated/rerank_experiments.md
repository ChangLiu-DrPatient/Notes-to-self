# HRLib Rerank Experiments (Deprecated)

This runbook is archived for historical reference.

- Active `examples/hrlib/30_inject.sh` / `30_inject_parquet.py` no longer expose
  rerank mode.
- Use this file only if you restore the historical rerank-capable injector path.

## Historical notes

This runbook executed cross-encoder reranking on top of the cleaned retrieval
pipeline and compared reranker model choices for abstraction relevance.

All commands were run from:

```bash
cd /home/changl9/Test-Time-Training/verl
```

## 1) Common setup

```bash
export RUN_TAG="$(date +%m%d-%H%M%S)"
export RUN_ROOT="/raid/$USER/eval/hrlib/stage1/rerank/${RUN_TAG}"
mkdir -p "$RUN_ROOT"

export IN_PARQUET="$HOME/data/MATH-500/test.parquet"
export QUERY_PARQUET="/raid/$USER/eval/hrlib/stage0/score_gate/0430-195103/test_rewritten.parquet"
export LIBRARY_DIR="/raid/$USER/traces/Qwen3-1.7B-Base/0419-165032-round0/extract_full/library_v1_semantic"
export DATA_TRAIN="$HOME/data/math/train.parquet"
export MODEL_PATH="Qwen/Qwen3-1.7B-Base"
```

## 2) Baselines to preserve (non-rerank)

```bash
LIBRARY_DIR="$LIBRARY_DIR" \
IN_PARQUET="$IN_PARQUET" \
OUT_PARQUET="$RUN_ROOT/test_hrlib_base_score_gate.parquet" \
QUERY_PARQUET="$QUERY_PARQUET" \
RETRIEVAL_MODE=score_gate \
DUMP_SCORES=1 \
conda run -n verl bash examples/hrlib/30_inject.sh
```

## 3) Reranker test ladder (simple to stronger)

### 3a) Fast smoke reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

```bash
LIBRARY_DIR="$LIBRARY_DIR" \
IN_PARQUET="$IN_PARQUET" \
OUT_PARQUET="$RUN_ROOT/test_hrlib_rerank_minilm.parquet" \
QUERY_PARQUET="$QUERY_PARQUET" \
RETRIEVAL_MODE=rerank \
RERANK_BASE_MODE=score_gate \
RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2" \
RERANK_CANDIDATE_K=30 \
RERANK_BATCH_SIZE=64 \
DUMP_SCORES=1 \
conda run -n verl bash examples/hrlib/30_inject.sh
```

### 3b) Main reranker (`BAAI/bge-reranker-base`)

```bash
LIBRARY_DIR="$LIBRARY_DIR" \
IN_PARQUET="$IN_PARQUET" \
OUT_PARQUET="$RUN_ROOT/test_hrlib_rerank_bge_base.parquet" \
QUERY_PARQUET="$QUERY_PARQUET" \
RETRIEVAL_MODE=rerank \
RERANK_BASE_MODE=score_gate \
RERANK_MODEL="BAAI/bge-reranker-base" \
RERANK_CANDIDATE_K=30 \
RERANK_BATCH_SIZE=32 \
DUMP_SCORES=1 \
conda run -n verl bash examples/hrlib/30_inject.sh
```

### 3c) Stronger reranker (`BAAI/bge-reranker-large`)

```bash
LIBRARY_DIR="$LIBRARY_DIR" \
IN_PARQUET="$IN_PARQUET" \
OUT_PARQUET="$RUN_ROOT/test_hrlib_rerank_bge_large.parquet" \
QUERY_PARQUET="$QUERY_PARQUET" \
RETRIEVAL_MODE=rerank \
RERANK_BASE_MODE=score_gate \
RERANK_MODEL="BAAI/bge-reranker-large" \
RERANK_CANDIDATE_K=30 \
RERANK_BATCH_SIZE=16 \
DUMP_SCORES=1 \
conda run -n verl bash examples/hrlib/30_inject.sh
```

## 4) Retrieval-side diagnostics

```bash
conda run -n verl python examples/hrlib/score_gate_diagnostics.py \
  --scores "$RUN_ROOT/test_hrlib_rerank_minilm_scores.jsonl"

conda run -n verl python examples/hrlib/score_gate_diagnostics.py \
  --scores "$RUN_ROOT/test_hrlib_rerank_bge_base_scores.jsonl"

conda run -n verl python examples/hrlib/score_gate_diagnostics.py \
  --scores "$RUN_ROOT/test_hrlib_rerank_bge_large_scores.jsonl"
```

## 5) Prompt-only relevance judge (no `40_eval.sh`)

Run the new prompt judge over each injected parquet + sidecar:

```bash
OPENROUTER_API_KEY=... \
conda run -n verl python examples/hrlib/judge_prompt_relevance.py \
  --parquet "$RUN_ROOT/test_hrlib_base_score_gate.parquet" \
  --scores "$RUN_ROOT/test_hrlib_base_score_gate_scores.jsonl" \
  --out "$RUN_ROOT/judge/base_score_gate.judge.jsonl" \
  --raw_dump_out "$RUN_ROOT/judge/base_score_gate.raw.jsonl"

OPENROUTER_API_KEY=... \
conda run -n verl python examples/hrlib/judge_prompt_relevance.py \
  --parquet "$RUN_ROOT/test_hrlib_rerank_minilm.parquet" \
  --scores "$RUN_ROOT/test_hrlib_rerank_minilm_scores.jsonl" \
  --out "$RUN_ROOT/judge/rerank_minilm.judge.jsonl" \
  --raw_dump_out "$RUN_ROOT/judge/rerank_minilm.raw.jsonl"

OPENROUTER_API_KEY=... \
conda run -n verl python examples/hrlib/judge_prompt_relevance.py \
  --parquet "$RUN_ROOT/test_hrlib_rerank_bge_base.parquet" \
  --scores "$RUN_ROOT/test_hrlib_rerank_bge_base_scores.jsonl" \
  --out "$RUN_ROOT/judge/rerank_bge_base.judge.jsonl" \
  --raw_dump_out "$RUN_ROOT/judge/rerank_bge_base.raw.jsonl"

OPENROUTER_API_KEY=... \
conda run -n verl python examples/hrlib/judge_prompt_relevance.py \
  --parquet "$RUN_ROOT/test_hrlib_rerank_bge_large.parquet" \
  --scores "$RUN_ROOT/test_hrlib_rerank_bge_large_scores.jsonl" \
  --out "$RUN_ROOT/judge/rerank_bge_large.judge.jsonl" \
  --raw_dump_out "$RUN_ROOT/judge/rerank_bge_large.raw.jsonl"
```

Aggregate relevance results:

```bash
conda run -n verl python examples/hrlib/analyze_prompt_relevance.py \
  --judge_results \
    "$RUN_ROOT/judge/base_score_gate.judge.jsonl" \
    "$RUN_ROOT/judge/rerank_minilm.judge.jsonl" \
    "$RUN_ROOT/judge/rerank_bge_base.judge.jsonl" \
    "$RUN_ROOT/judge/rerank_bge_large.judge.jsonl" \
  --out_json "$RUN_ROOT/judge/prompt_relevance_summary.json" \
  --out_md "$RUN_ROOT/judge/prompt_relevance_summary.md"
```

## 6) Choosing the best reranker

Recommend the smallest model that gives clear relevance gains:

- first verify pipeline with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- treat `BAAI/bge-reranker-base` as the default candidate
- only promote `BAAI/bge-reranker-large` if relevance gains are material over base
- do not run `examples/hrlib/40_eval.sh` in this no-GPU prompt-only pass

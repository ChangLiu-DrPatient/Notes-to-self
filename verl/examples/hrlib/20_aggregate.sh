#!/usr/bin/env bash
set -euo pipefail

if [[ "${TRACE_COMMANDS:-0}" == "1" ]]; then
    set -x
fi

# Defaults point at the smoke extraction output from 10_extract_smoke.sh.
LABELED_PARQUET_DEFAULT="/raid/$USER/traces/Qwen3-1.7B-Base/traces_round0_labeled.parquet"
EXTRACT_DIR_DEFAULT="$(dirname "$LABELED_PARQUET_DEFAULT")/extract_full"

RAW_JSONL=${RAW_JSONL:-"$EXTRACT_DIR_DEFAULT/raw_abstractions.jsonl"}
OUT_DIR=${OUT_DIR:-"$(dirname "$RAW_JSONL")"}
METHOD=${METHOD:-semantic}
TEXT_RATIO=${TEXT_RATIO:-80}
SEMANTIC_RATIO=${SEMANTIC_RATIO:-0.85}
EMBEDDER=${EMBEDDER:-sentence-transformers/all-MiniLM-L6-v2}
DEVICE=${DEVICE:-auto}
EMBED_INCLUDE_DOMAIN=${EMBED_INCLUDE_DOMAIN:-true}
EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-256}
WRITE_EMBEDDINGS=${WRITE_EMBEDDINGS:-true}
FILTER=${FILTER:-false}
MIN_CHARS=${MIN_CHARS:-15}
MAX_CHARS=${MAX_CHARS:-240}
PER_TYPE=${PER_TYPE:-true}
TOP_N_PREVIEW=${TOP_N_PREVIEW:-50}
TOP_N_PRINT=${TOP_N_PRINT:-5}
KEEP_CLUSTER_MEMBERS=${KEEP_CLUSTER_MEMBERS:-true}
NORMALIZE_DOMAIN=${NORMALIZE_DOMAIN:-true}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}

if [[ ! -f "$RAW_JSONL" ]]; then
    echo "[error] RAW_JSONL not found: $RAW_JSONL" >&2
    echo "        run 10_extract_smoke.sh or 10_extract_full.sh first, or override RAW_JSONL." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm -f \
        "$OUT_DIR/library.jsonl" \
        "$OUT_DIR/library.md" \
        "$OUT_DIR/meta.json" \
        "$OUT_DIR/dropped.jsonl" \
        "$OUT_DIR/embeddings.npy" \
        "$OUT_DIR/embeddings_meta.json"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$VERL_ROOT"

started_epoch="$(date +%s)"
started_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

AGG_ARGS=(
    --raw "$RAW_JSONL"
    --out_dir "$OUT_DIR"
    --method "$METHOD"
    --text_ratio "$TEXT_RATIO"
    --semantic_ratio "$SEMANTIC_RATIO"
    --embedder "$EMBEDDER"
    --device "$DEVICE"
    --embed_include_domain "$EMBED_INCLUDE_DOMAIN"
    --embed_batch_size "$EMBED_BATCH_SIZE"
    --write_embeddings "$WRITE_EMBEDDINGS"
    --filter "$FILTER"
    --min_chars "$MIN_CHARS"
    --max_chars "$MAX_CHARS"
    --per_type "$PER_TYPE"
    --top_n_preview "$TOP_N_PREVIEW"
    --top_n_print "$TOP_N_PRINT"
    --keep_cluster_members "$KEEP_CLUSTER_MEMBERS"
    --normalize_domain "$NORMALIZE_DOMAIN"
    --overwrite
)

python3 examples/hrlib/20_aggregate.py "${AGG_ARGS[@]}"

ended_epoch="$(date +%s)"
ended_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_sec="$((ended_epoch - started_epoch))"
echo "[timing] aggregate_started_utc=${started_iso} aggregate_ended_utc=${ended_iso} wall_elapsed_sec=${elapsed_sec}"

SECTION_RULE="================================================================================"

echo ""
echo "$SECTION_RULE"
echo "meta.json"
echo "$SECTION_RULE"
if [[ -f "$OUT_DIR/meta.json" ]]; then
    cat "$OUT_DIR/meta.json"
    echo ""
else
    echo "(missing)"
fi
echo ""

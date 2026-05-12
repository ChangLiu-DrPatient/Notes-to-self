#!/usr/bin/env bash
# LLM-as-judge for HRLib abstraction utilization (see examples/hrlib/judge_abstraction_use.py).
#
# Environment variables (all optional except OPENROUTER_API_KEY):
#   EVAL_JSONL, OUT_DIR, MODEL, FALLBACK_MODEL, NO_FALLBACK, MAX_CONCURRENCY,
#   ROLLOUTS_PER_PROBLEM, CORRECT_SCORE_THRESHOLD, LIMIT, RESUME_FROM, CLEAN_OUTPUT
#
# Correct ways to pass LIMIT (and friends):
#   LIMIT=5 ROLLOUTS_PER_PROBLEM=1 bash examples/hrlib/judge_abstraction_use.sh
#   export LIMIT=5; bash examples/hrlib/judge_abstraction_use.sh
#
# You can also pass overrides as trailing KEY=value arguments (parsed below):
#   bash examples/hrlib/judge_abstraction_use.sh LIMIT=5 ROLLOUTS_PER_PROBLEM=1
#
# Wrong (values become unused argv; LIMIT is NOT set for the script):
#   bash examples/hrlib/judge_abstraction_use.sh \
#     LIMIT=5
set -euo pipefail

if [[ "${TRACE_COMMANDS:-0}" == "1" ]]; then
    set -x
fi

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"

# Accept KEY=value tokens from "$@" so `bash script.sh LIMIT=5` works like env prefix.
for __arg in "$@"; do
    case "$__arg" in
        EVAL_JSONL=*|OUT_DIR=*|MODEL=*|FALLBACK_MODEL=*|NO_FALLBACK=*|MAX_CONCURRENCY=*|ROLLOUTS_PER_PROBLEM=*|CORRECT_SCORE_THRESHOLD=*|LIMIT=*|RESUME_FROM=*|CLEAN_OUTPUT=*)
            export "$__arg"
            ;;
    esac
done

EVAL_JSONL=${EVAL_JSONL:-/raid/$USER/eval/hrlib/stage0/v1/0.jsonl}
OUT_DIR=${OUT_DIR:-$(dirname "$EVAL_JSONL")/judge_abstraction_use}
MODEL=${MODEL:-openai/gpt-oss-120b:free}
FALLBACK_MODEL=${FALLBACK_MODEL:-openai/gpt-oss-120b}
NO_FALLBACK=${NO_FALLBACK:-0}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-20}
ROLLOUTS_PER_PROBLEM=${ROLLOUTS_PER_PROBLEM:-1}
CORRECT_SCORE_THRESHOLD=${CORRECT_SCORE_THRESHOLD:-1.0}
LIMIT=${LIMIT:-}
RESUME_FROM=${RESUME_FROM:-}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}

RESULTS_JSONL="${OUT_DIR}/judge_results.jsonl"
RAW_DUMPS_JSONL="${OUT_DIR}/judge_raw_dumps.jsonl"

mkdir -p "$OUT_DIR"
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm -f "$RESULTS_JSONL" "$RAW_DUMPS_JSONL"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$VERL_ROOT"

started_epoch="$(date +%s)"
started_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

JUDGE_ARGS=(
    --eval_jsonl "$EVAL_JSONL"
    --out "$RESULTS_JSONL"
    --raw_dump_out "$RAW_DUMPS_JSONL"
    --backend openrouter
    --base_url https://openrouter.ai/api/v1
    --model "$MODEL"
    --max_concurrency "$MAX_CONCURRENCY"
    --rollouts_per_problem "$ROLLOUTS_PER_PROBLEM"
    --correct_score_threshold "$CORRECT_SCORE_THRESHOLD"
)

if [[ "$NO_FALLBACK" == "1" ]]; then
    JUDGE_ARGS+=(--no_fallback)
else
    JUDGE_ARGS+=(--fallback_model "$FALLBACK_MODEL")
fi

if [[ -n "$LIMIT" ]]; then
    JUDGE_ARGS+=(--limit "$LIMIT")
fi

if [[ -n "$RESUME_FROM" ]]; then
    JUDGE_ARGS+=(--resume_from "$RESUME_FROM")
fi

python3 examples/hrlib/judge_abstraction_use.py "${JUDGE_ARGS[@]}"

ended_epoch="$(date +%s)"
ended_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_sec="$((ended_epoch - started_epoch))"
echo "[timing] judge_started_utc=${started_iso} judge_ended_utc=${ended_iso} wall_elapsed_sec=${elapsed_sec}"
echo "[output results] ${RESULTS_JSONL}"
echo "[output raw dumps] ${RAW_DUMPS_JSONL}"

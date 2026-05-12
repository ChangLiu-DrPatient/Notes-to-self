#!/usr/bin/env bash
set -euo pipefail

if [[ "${TRACE_COMMANDS:-0}" == "1" ]]; then
    set -x
fi

# Defaults target the v1 semantic library built from the round0 corpus and the
# standard MATH-500 eval split; override any of these via environment.
# Smoke check tip:
#   LIMIT=5 bash examples/hrlib/30_inject.sh
# Then verify injection_meta.json includes:
#   - "query_recipe": "[{subject}] {user_text}"
#   - non-empty "subject_counts"
LIBRARY_DIR_DEFAULT="/raid/$USER/traces/Qwen3-1.7B-Base/0419-165032-round0/extract_full/library_v1_semantic"

LIBRARY_DIR=${LIBRARY_DIR:-"$LIBRARY_DIR_DEFAULT"}
IN_PARQUET=${IN_PARQUET:-"$HOME/data/MATH-500/test.parquet"}
OUT_PARQUET=${OUT_PARQUET:-"$HOME/data/MATH-500/test_hrlib_v1.parquet"}
TOP_K=${TOP_K:-6}
QUERY_FROM=${QUERY_FROM:-user}
QUERY_RECIPE=${QUERY_RECIPE:-"[{subject}] {user_text}"}
SUBJECT_FIELD=${SUBJECT_FIELD:-subject}
QUERY_PARQUET=${QUERY_PARQUET:-}
RETRIEVAL_MODE=${RETRIEVAL_MODE:-score_gate}  # orig | rewrite | score_gate
DEVICE=${DEVICE:-auto}
LIMIT=${LIMIT:-}
PREVIEW_CHARS=${PREVIEW_CHARS:-1200}
TEMPLATE=${TEMPLATE:-v1}      # v1 | std_rag
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}
DUMP_SCORES=${DUMP_SCORES:-}
QUERY_GATE=${QUERY_GATE:-off}  # legacy compat: off | score
GATE_METRIC=${GATE_METRIC:-top1}
GATE_MARGIN=${GATE_MARGIN:-0.02}
GATE_TIE_POLICY=${GATE_TIE_POLICY:-prefer_original}  # prefer_original | prefer_rewrite

if [[ -z "$RETRIEVAL_MODE" ]]; then
    if [[ "$QUERY_GATE" == "score" ]]; then
        RETRIEVAL_MODE="score_gate"
    elif [[ -n "$QUERY_PARQUET" ]]; then
        RETRIEVAL_MODE="rewrite"
    else
        RETRIEVAL_MODE="orig"
    fi
fi

case "$RETRIEVAL_MODE" in
    orig|rewrite|score_gate) ;;
    *)
        echo "[error] RETRIEVAL_MODE must be one of: orig, rewrite, score_gate (got '$RETRIEVAL_MODE')" >&2
        exit 1
        ;;
esac

if [[ "$RETRIEVAL_MODE" == "score_gate" && "$QUERY_GATE" == "off" ]]; then
    QUERY_GATE="score"
fi
if [[ "$QUERY_GATE" == "score" && "$RETRIEVAL_MODE" != "score_gate" ]]; then
    echo "[error] QUERY_GATE=score conflicts with RETRIEVAL_MODE=$RETRIEVAL_MODE; use RETRIEVAL_MODE=score_gate" >&2
    exit 1
fi
if [[ "$RETRIEVAL_MODE" != "orig" && -z "$QUERY_PARQUET" ]]; then
    echo "[error] RETRIEVAL_MODE=$RETRIEVAL_MODE requires QUERY_PARQUET" >&2
    exit 1
fi

if [[ ! -d "$LIBRARY_DIR" ]]; then
    echo "[error] LIBRARY_DIR not found: $LIBRARY_DIR" >&2
    echo "        run 20_aggregate.sh first, or override LIBRARY_DIR." >&2
    exit 1
fi
if [[ ! -f "$IN_PARQUET" ]]; then
    echo "[error] IN_PARQUET not found: $IN_PARQUET" >&2
    exit 1
fi
if [[ -n "$QUERY_PARQUET" && ! -f "$QUERY_PARQUET" ]]; then
    echo "[error] QUERY_PARQUET not found: $QUERY_PARQUET" >&2
    exit 1
fi

META_PATH="${OUT_PARQUET%.parquet}_injection_meta.json"
SCORES_PATH="${OUT_PARQUET%.parquet}_scores.jsonl"

mkdir -p "$(dirname "$OUT_PARQUET")"
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm -f "$OUT_PARQUET" "$META_PATH" "$SCORES_PATH"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$VERL_ROOT"

INJECT_ARGS=(
    --library "$LIBRARY_DIR"
    --in "$IN_PARQUET"
    --out "$OUT_PARQUET"
    --top_k "$TOP_K"
    --query_from "$QUERY_FROM"
    --query_recipe "$QUERY_RECIPE"
    --subject_field "$SUBJECT_FIELD"
    --retrieval_mode "$RETRIEVAL_MODE"
    --device "$DEVICE"
    --preview_chars "$PREVIEW_CHARS"
    --template "$TEMPLATE"
    --query_gate "$QUERY_GATE"
    --gate_metric "$GATE_METRIC"
    --gate_margin "$GATE_MARGIN"
    --gate_tie_policy "$GATE_TIE_POLICY"
    --overwrite
)

if [[ -n "$LIMIT" ]]; then
    INJECT_ARGS+=(--limit "$LIMIT")
fi
if [[ -n "$QUERY_PARQUET" ]]; then
    INJECT_ARGS+=(--query_parquet "$QUERY_PARQUET")
fi
if [[ -n "$DUMP_SCORES" ]]; then
    INJECT_ARGS+=(--dump_scores)
fi

started_epoch="$(date +%s)"
started_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python3 examples/hrlib/30_inject_parquet.py "${INJECT_ARGS[@]}"

ended_epoch="$(date +%s)"
ended_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_sec="$((ended_epoch - started_epoch))"
echo "[timing] inject_started_utc=${started_iso} inject_ended_utc=${ended_iso} wall_elapsed_sec=${elapsed_sec}"

SECTION_RULE="================================================================================"

echo ""
echo "$SECTION_RULE"
echo "injection_meta.json"
echo "$SECTION_RULE"
if [[ -f "$META_PATH" ]]; then
    cat "$META_PATH"
    echo ""
else
    echo "(missing: $META_PATH)"
fi
echo ""

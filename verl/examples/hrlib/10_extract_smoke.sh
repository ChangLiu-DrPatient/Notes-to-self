#!/usr/bin/env bash
set -euo pipefail

if [[ "${TRACE_COMMANDS:-0}" == "1" ]]; then
    set -x
fi

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"

LABELED_PARQUET=${LABELED_PARQUET:-/raid/$USER/traces/Qwen3-1.7B-Base/0419-165032-round0/traces_round0_labeled.parquet}
OUT_DIR=${OUT_DIR:-$(dirname "$LABELED_PARQUET")/extract_smoke}
MODEL=${MODEL:-openai/gpt-oss-120b:free}
FALLBACK_MODEL=${FALLBACK_MODEL:-openai/gpt-oss-120b}
NO_FALLBACK=${NO_FALLBACK:-0}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-4}
LIMIT_SUCCESS=${LIMIT_SUCCESS:-2}
LIMIT_FAILURE=${LIMIT_FAILURE:-2}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}

mkdir -p "$OUT_DIR"
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm -f "$OUT_DIR/raw_abstractions.jsonl" "$OUT_DIR/raw_llm_dumps.jsonl"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$VERL_ROOT"

started_epoch="$(date +%s)"
started_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

EXTRACT_ARGS=(
    --traces "$LABELED_PARQUET"
    --out "$OUT_DIR/raw_abstractions.jsonl"
    --raw_dump_out "$OUT_DIR/raw_llm_dumps.jsonl"
    --backend openrouter
    --base_url https://openrouter.ai/api/v1
    --model "$MODEL"
    --max_concurrency "$MAX_CONCURRENCY"
    --limit_success "$LIMIT_SUCCESS"
    --limit_failure "$LIMIT_FAILURE"
)
if [[ "$NO_FALLBACK" == "1" ]]; then
    EXTRACT_ARGS+=(--no_fallback)
else
    EXTRACT_ARGS+=(--fallback_model "$FALLBACK_MODEL")
fi

python3 examples/hrlib/10_extract.py "${EXTRACT_ARGS[@]}"

ended_epoch="$(date +%s)"
ended_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_sec="$((ended_epoch - started_epoch))"
echo "[timing] extraction_started_utc=${started_iso} extraction_ended_utc=${ended_iso} wall_elapsed_sec=${elapsed_sec}"

if [[ -f "$OUT_DIR/raw_llm_dumps.jsonl" ]]; then
    python3 - "$OUT_DIR/raw_llm_dumps.jsonl" <<'PY'
import json
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1])
values = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "api_elapsed_sec" in obj:
            try:
                values.append(float(obj["api_elapsed_sec"]))
            except (TypeError, ValueError):
                pass
        elif "api_elapsed_ms" in obj:
            # Backward compatibility for old logs.
            try:
                values.append(float(obj["api_elapsed_ms"]) / 1000.0)
            except (TypeError, ValueError):
                pass

if values:
    print(
        "[timing] per_call_api_elapsed_sec: "
        f"count={len(values)} min={min(values):.2f} p50={statistics.median(values):.2f} "
        f"mean={statistics.mean(values):.2f} max={max(values):.2f}"
    )
else:
    print("[timing] per_call_api_elapsed_sec: no valid values found")
PY
fi

SECTION_RULE="================================================================================"

print_jsonl_spaced() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "(missing)"
        return
    fi
    python3 - "$path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
first = True
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if not first:
            print()
        first = False
        print(line)
PY
}

echo ""
echo "$SECTION_RULE"
echo "raw_abstractions.jsonl"
echo "$SECTION_RULE"
print_jsonl_spaced "$OUT_DIR/raw_abstractions.jsonl"

echo ""
echo "$SECTION_RULE"
echo "raw_llm_dumps.jsonl"
echo "$SECTION_RULE"
print_jsonl_spaced "$OUT_DIR/raw_llm_dumps.jsonl"
echo ""

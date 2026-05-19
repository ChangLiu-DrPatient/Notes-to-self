#!/usr/bin/env bash
# Full-dataset abstraction extraction via OpenRouter (primary :free + paid fallback).
#
# Full run (default): processes all rows; does not pass --limit_*.
#   export OPENROUTER_API_KEY=...
#   export LABELED_PARQUET=/path/to/traces_round0_labeled.parquet
#   bash examples/hrlib/10_extract_full.sh
#
# See paid fallback in action (stress demo — may trigger free-tier 429):
#   FALLBACK_DEMO_MODE=1 bash examples/hrlib/10_extract_full.sh
# Optional: DEMO_MAX_CONCURRENCY=32 (default 24 in demo mode).
#
set -euo pipefail

if [[ "${TRACE_COMMANDS:-0}" == "1" ]]; then
    set -x
fi

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"

LABELED_PARQUET=${LABELED_PARQUET:-/raid/$USER/traces/Qwen3-1.7B-Base/traces_round0_labeled.parquet}
# MODEL=${MODEL:-openai/gpt-oss-120b:free}
# FALLBACK_MODEL=${FALLBACK_MODEL:-openai/gpt-oss-120b}
MODEL=${MODEL:-deepseek/deepseek-v4-flash}
FALLBACK_MODEL=${FALLBACK_MODEL:-deepseek/deepseek-v4-flash}
NO_FALLBACK=${NO_FALLBACK:-0}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-40}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-0}
FALLBACK_DEMO_MODE=${FALLBACK_DEMO_MODE:-0}
PREVIEW_JSONL=${PREVIEW_JSONL:-0}
RESUME_FROM=${RESUME_FROM:-}

if [[ "$FALLBACK_DEMO_MODE" == "1" ]]; then
    echo "[info] FALLBACK_DEMO_MODE=1: subset + higher concurrency to encourage free-tier rate limits (429) -> paid fallback."
    OUT_DIR=${OUT_DIR:-$(dirname "$LABELED_PARQUET")/extract_full_fallback_demo}
    MAX_CONCURRENCY=${DEMO_MAX_CONCURRENCY:-20}
    LIMIT_SUCCESS=${LIMIT_SUCCESS:-40}
    LIMIT_FAILURE=${LIMIT_FAILURE:-40}
    # Fresh demo run (override global CLEAN_OUTPUT=0 default).
    CLEAN_OUTPUT=1
    PREVIEW_JSONL=${PREVIEW_JSONL:-1}
else
    OUT_DIR=${OUT_DIR:-$(dirname "$LABELED_PARQUET")}
fi

mkdir -p "$OUT_DIR"
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm -f "$OUT_DIR/raw_abstractions.jsonl" "$OUT_DIR/raw_llm_dumps.jsonl"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$VERL_ROOT"

LOG_FILE="${LOG_FILE:-$OUT_DIR/extract_full.log}"

EXTRACT_ARGS=(
    --traces "$LABELED_PARQUET"
    --out "$OUT_DIR/raw_abstractions.jsonl"
    --raw_dump_out "$OUT_DIR/raw_llm_dumps.jsonl"
    --backend openrouter
    --base_url https://openrouter.ai/api/v1
    --model "$MODEL"
    --max_concurrency "$MAX_CONCURRENCY"
)

if [[ -n "${LIMIT_SUCCESS:-}" ]]; then
    EXTRACT_ARGS+=(--limit_success "$LIMIT_SUCCESS")
fi
if [[ -n "${LIMIT_FAILURE:-}" ]]; then
    EXTRACT_ARGS+=(--limit_failure "$LIMIT_FAILURE")
fi
if [[ -n "$RESUME_FROM" ]]; then
    EXTRACT_ARGS+=(--resume_from "$RESUME_FROM")
fi
if [[ "$NO_FALLBACK" == "1" ]]; then
    EXTRACT_ARGS+=(--no_fallback)
else
    EXTRACT_ARGS+=(--fallback_model "$FALLBACK_MODEL")
fi

started_epoch="$(date +%s)"
started_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[info] OUT_DIR=$OUT_DIR"
echo "[info] log -> $LOG_FILE"
python3 examples/hrlib/10_extract.py "${EXTRACT_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

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
fallback_true = 0
fallback_false = 0
for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(obj, dict):
        continue
    if obj.get("fallback_used") is True:
        fallback_true += 1
    elif "fallback_used" in obj:
        fallback_false += 1
    if "api_elapsed_sec" in obj:
        try:
            values.append(float(obj["api_elapsed_sec"]))
        except (TypeError, ValueError):
            pass
    elif "api_elapsed_ms" in obj:
        try:
            values.append(float(obj["api_elapsed_ms"]) / 1000.0)
        except (TypeError, ValueError):
            pass

print(
    f"[stats] raw_llm_dumps lines: fallback_used=True {fallback_true}, "
    f"fallback_used=False {fallback_false}"
)
if values:
    print(
        "[timing] per_call_api_elapsed_sec: "
        f"count={len(values)} min={min(values):.2f} p50={statistics.median(values):.2f} "
        f"mean={statistics.mean(values):.2f} max={max(values):.2f}"
    )
else:
    print("[timing] per_call_api_elapsed_sec: no valid values found")
PY

    echo ""
    echo "================================================================================"
    echo "Sample lines where fallback_used=true (up to 8)"
    echo "================================================================================"
    python3 - "$OUT_DIR/raw_llm_dumps.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
n = 0
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("fallback_used") is True:
            print(json.dumps(obj, ensure_ascii=False))
            print()
            n += 1
            if n >= 8:
                break
if n == 0:
    print("(none — free tier did not hit retry-worthy errors, or NO_FALLBACK=1)")
PY
fi

SECTION_RULE="================================================================================"

print_jsonl_spaced() {
    local path="$1"
    local max_lines="${2:-80}"
    if [[ ! -f "$path" ]]; then
        echo "(missing)"
        return
    fi
    python3 - "$path" "$max_lines" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_lines = int(sys.argv[2])
count = 0
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
        count += 1
        if count >= max_lines:
            print()
            print(f"... truncated after {max_lines} entries; see file: {path}")
            break
PY
}

if [[ "$PREVIEW_JSONL" == "1" ]]; then
    echo ""
    echo "$SECTION_RULE"
    echo "raw_abstractions.jsonl (preview)"
    echo "$SECTION_RULE"
    print_jsonl_spaced "$OUT_DIR/raw_abstractions.jsonl" 40

    echo ""
    echo "$SECTION_RULE"
    echo "raw_llm_dumps.jsonl (preview)"
    echo "$SECTION_RULE"
    print_jsonl_spaced "$OUT_DIR/raw_llm_dumps.jsonl" 20
    echo ""
fi

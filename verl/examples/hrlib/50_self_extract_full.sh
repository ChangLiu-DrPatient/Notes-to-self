#!/usr/bin/env bash
# Full-dataset abstraction extraction via local model + main_ppo validation (no API).
#
# Pipeline:
#   1) build extraction-prompt parquet (same prompts as 10_extract / extract.py)
#   2) run main_ppo val-only (n=1) with vLLM
#   3) merge val JSONL -> raw_self_abstractions.jsonl + raw_self_llm_dumps.jsonl
#
# Extraction prompts include full traces: raise MAX_PROMPT_LENGTH if your traces are long
# (defaults below are higher than 25_rewrite_queries.sh).
#
# Output layout matches 10_extract_full.sh: by default OUT_DIR is the directory of
# LABELED_PARQUET (e.g. $TRACE_ROOT/round0/), with raw_self_*.jsonl so API vs local
# runs can coexist in the same folder.
#
#   export LABELED_PARQUET=/path/to/traces_round0_labeled.parquet
#   bash examples/hrlib/50_self_extract_full.sh
set -euo pipefail

if [[ "${TRACE_COMMANDS:-0}" == "1" ]]; then
    set -x
fi

export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2}
NUM_GPUS=${NUM_GPUS:-2}
GPU_MEM=${GPU_MEM:-0.8}
MICRO_BSZ=${MICRO_BSZ:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B-Base"}
LABELED_PARQUET=${LABELED_PARQUET:-/raid/$USER/traces/Qwen3-1.7B-Base/traces_round0_labeled.parquet}
DATA_TRAIN=${DATA_TRAIN:-"$HOME/data/math/train.parquet"}
LIMIT_SUCCESS=${LIMIT_SUCCESS:-}
LIMIT_FAILURE=${LIMIT_FAILURE:-}
# Traces + instruction can be long; increase if needed (truncation=error surfaces overflow).
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
DO_SAMPLE=${DO_SAMPLE:-False}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"hrlib-self-extract-${DATE}_${TIME_TAG}"}

if [[ ! -f "$LABELED_PARQUET" ]]; then
    echo "[error] LABELED_PARQUET not found: $LABELED_PARQUET" >&2
    exit 1
fi
if [[ ! -f "$DATA_TRAIN" ]]; then
    echo "[error] DATA_TRAIN not found: $DATA_TRAIN" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$VERL_ROOT"

OUT_DIR=${OUT_DIR:-$(dirname "$LABELED_PARQUET")}
LOG_FILE="${LOG_FILE:-$OUT_DIR/self_extract_full.log}"
WORK_DIR="${OUT_DIR}/work"
VAL_DIR="${OUT_DIR}/val_generations"
RAW_ABSTRACTIONS="${OUT_DIR}/raw_self_abstractions.jsonl"
RAW_DUMPS="${OUT_DIR}/raw_self_llm_dumps.jsonl"

mkdir -p "$OUT_DIR" "$WORK_DIR"

EXTRACT_PROMPTS_PARQUET=${EXTRACT_PROMPTS_PARQUET:-"${WORK_DIR}/extract_prompts.parquet"}

BUILD_ARGS=(
    build
    --in "$LABELED_PARQUET"
    --out "$EXTRACT_PROMPTS_PARQUET"
    --overwrite
)
if [[ -n "$LIMIT_SUCCESS" ]]; then
    BUILD_ARGS+=(--limit_success "$LIMIT_SUCCESS")
fi
if [[ -n "$LIMIT_FAILURE" ]]; then
    BUILD_ARGS+=(--limit_failure "$LIMIT_FAILURE")
fi

python3 examples/hrlib/50_self_extract_full.py "${BUILD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# ---------------- Ray isolation (single-node) ----------------
unset RAY_ADDRESS
unset RAY_NAMESPACE

RUN_ID="${DATE}_${TIME_TAG}_$$" # same pattern as examples/hrlib/40_eval.sh
export RAY_TMPDIR="/raid/$USER/ray/${RUN_ID}"
mkdir -p "$RAY_TMPDIR"

NODE_IP=$(hostname -I | awk '{print $1}')
for _ in {1..30}; do
  RAY_PORT=$(( 20000 + ($(id -u) % 8000) + (RANDOM % 1000)))
  if ray start --head \
      --node-ip-address="$NODE_IP" \
      --port="$RAY_PORT" \
      --temp-dir="$RAY_TMPDIR" \
      --disable-usage-stats \
      --include-dashboard=false; then
    export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"
    break
  fi
  sleep 0.2
done

echo "[Ray] RAY_ADDRESS=$RAY_ADDRESS" | tee -a "$LOG_FILE"
echo "[Ray] RAY_TMPDIR=$RAY_TMPDIR" | tee -a "$LOG_FILE"
ray status || { echo "[Ray] failed to start" >&2; exit 1; }

cleanup_ray() {
  echo "[Ray] Stopping Ray head node..."
  ray stop --temp-dir "$RAY_TMPDIR" >/dev/null 2>&1 || true
  pgrep -f "$RAY_TMPDIR" | xargs -r kill -9 >/dev/null 2>&1 || true
  rm -rf "$RAY_TMPDIR"
}
trap cleanup_ray EXIT
trap 'cleanup_ray; exit 130' INT
trap 'cleanup_ray; exit 143' TERM
trap 'cleanup_ray; exit 1' ERR
# ------------------------------------------------------------

python3 -m verl.trainer.main_ppo \
    reward_model.use_reward_loop=False \
    reward_model.reward_manager=naive \
    data.train_files="$DATA_TRAIN" \
    data.val_files="$EXTRACT_PROMPTS_PARQUET" \
    data.train_batch_size=128 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$DO_SAMPLE" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p="$TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$TOP_K" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name=hrlib_self_extract \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.save_freq=2000000 \
    trainer.test_freq=10 \
    trainer.validation_data_dir="$VAL_DIR" \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.max_critic_ckpt_to_keep=0 \
    trainer.default_local_dir="$OUT_DIR" \
    trainer.total_epochs=0 2>&1 | tee -a "$LOG_FILE"

python3 examples/hrlib/50_self_extract_full.py merge \
    --original "$LABELED_PARQUET" \
    --val_jsonl_dir "$VAL_DIR" \
    --raw_abstractions_out "$RAW_ABSTRACTIONS" \
    --raw_dump_out "$RAW_DUMPS" \
    --model_label "$MODEL_PATH" \
    --overwrite 2>&1 | tee -a "$LOG_FILE"

echo "[self_extract] extract_prompts_parquet=$EXTRACT_PROMPTS_PARQUET"
echo "[self_extract] val_dir=$VAL_DIR"
echo "[self_extract] raw_self_abstractions=$RAW_ABSTRACTIONS"
echo "[self_extract] raw_self_llm_dumps=$RAW_DUMPS"
echo "[self_extract] log_file=$LOG_FILE"

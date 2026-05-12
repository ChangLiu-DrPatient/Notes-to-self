#!/usr/bin/env bash
# Build rewritten retrieval queries via main_ppo validation-only generation.
#
# Pipeline:
#   1) build rewrite prompt parquet
#   2) run main_ppo val-only (n=1) to generate keyword rewrites
#   3) merge generated rewrites back into original parquet schema
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
ROLLOUT_TP=${ROLLOUT_TP:-1}

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B-Base"}
IN_PARQUET=${IN_PARQUET:-"$HOME/data/MATH-500/test.parquet"}
OUT_PARQUET=${OUT_PARQUET:-"$HOME/data/MATH-500/test_rewritten.parquet"}
DATA_TRAIN=${DATA_TRAIN:-"$HOME/data/math/train.parquet"}
LIMIT=${LIMIT:-}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
DO_SAMPLE=${DO_SAMPLE:-True}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"hrlib-rewrite-${DATE}_${TIME_TAG}"}

if [[ ! -f "$IN_PARQUET" ]]; then
    echo "[error] IN_PARQUET not found: $IN_PARQUET" >&2
    exit 1
fi
if [[ ! -f "$DATA_TRAIN" ]]; then
    echo "[error] DATA_TRAIN not found: $DATA_TRAIN" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$VERL_ROOT"

OUTPUT_DIR=${OUTPUT_DIR:-"/raid/$USER/eval/hrlib/stage0/rewrite_queries/${DATE}-${TIME_TAG}"}
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/rewrite.log"
WORK_DIR="${OUTPUT_DIR}/work"
mkdir -p "$WORK_DIR"

REWRITE_PROMPTS_PARQUET=${REWRITE_PROMPTS_PARQUET:-"${WORK_DIR}/rewrite_prompts.parquet"}
REWRITE_VAL_DIR=${REWRITE_VAL_DIR:-"${OUTPUT_DIR}/val_generations"}
REWRITE_META_PATH="${OUT_PARQUET%.parquet}_rewrite_meta.json"

BUILD_ARGS=(
    build
    --in "$IN_PARQUET"
    --out "$REWRITE_PROMPTS_PARQUET"
    --overwrite
)
if [[ -n "$LIMIT" ]]; then
    BUILD_ARGS+=(--limit "$LIMIT")
fi

python3 examples/hrlib/25_rewrite_queries.py "${BUILD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# ---------------- Ray isolation (single-node) ----------------
unset RAY_ADDRESS
unset RAY_NAMESPACE

RUN_ID="${DATE}_${TIME_TAG}_$$"
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
    data.val_files="$REWRITE_PROMPTS_PARQUET" \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p="$TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$TOP_K" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name=hrlib_rewrite \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.save_freq=2000000 \
    trainer.test_freq=10 \
    trainer.validation_data_dir="$REWRITE_VAL_DIR" \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.max_critic_ckpt_to_keep=0 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.total_epochs=0 2>&1 | tee -a "$LOG_FILE"

python3 examples/hrlib/25_rewrite_queries.py merge \
    --original "$IN_PARQUET" \
    --val_jsonl_dir "$REWRITE_VAL_DIR" \
    --out "$OUT_PARQUET" \
    --overwrite 2>&1 | tee -a "$LOG_FILE"

echo "[rewrite] rewrite_prompts_parquet=$REWRITE_PROMPTS_PARQUET"
echo "[rewrite] rewrite_val_dir=$REWRITE_VAL_DIR"
echo "[rewrite] rewritten_parquet=$OUT_PARQUET"
echo "[rewrite] rewritten_meta=$REWRITE_META_PATH"
echo "[rewrite] log_file=$LOG_FILE"

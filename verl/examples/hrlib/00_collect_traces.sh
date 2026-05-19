#!/usr/bin/env bash
set -xeuo pipefail

export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
NUM_GPUS=${NUM_GPUS:-1}
GPU_MEM=${GPU_MEM:-0.8}
MICRO_BSZ=${MICRO_BSZ:-8}
ROLLOUT_TP=${ROLLOUT_TP:-$NUM_GPUS}

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

# Override-able from env; defaults match implementation_plan_stage0_3.md §5.2.1
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
DATA_PATH=${DATA_PATH:-$HOME/data/math/train.parquet}
N_SAMPLES=${N_SAMPLES:-1}
ROUND_TAG=${ROUND_TAG:-round0}
SCORE_AFTER_COLLECT=${SCORE_AFTER_COLLECT:-1}

MODEL_NAME="${MODEL_PATH##*/}"
OUT_DIR=${OUT_DIR:-"/raid/${USER}/traces/${MODEL_NAME}/${DATE}-${TIME_TAG}-${ROUND_TAG}"}
mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/collect.log"
TRACE_PARQUET="${OUT_DIR}/traces.parquet"
LABELED_PARQUET="${TRACE_PARQUET%.parquet}_labeled.parquet"
VAL_DUMP_DIR="${OUT_DIR}/val_generations"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$VERL_ROOT"

# ---------------- Ray isolation (single-node) ----------------
unset RAY_ADDRESS
unset RAY_NAMESPACE

RUN_ID="${DATE}_${TIME_TAG}_$$"  # $$ is the current PID
export RAY_TMPDIR="/raid/${USER}/ray/${RUN_ID}"
mkdir -p "$RAY_TMPDIR"

# Find a free port and start a dedicated local Ray head.
NODE_IP="127.0.0.1" #$(hostname -I | awk '{print $1}')
for _ in {1..30}; do
  RAY_PORT=$(( 20000 + ($(id -u) % 8000) + (RANDOM % 1000) ))
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

# Verify Ray is running
echo "[Ray] RAY_ADDRESS=$RAY_ADDRESS"
echo "[Ray] RAY_TMPDIR=$RAY_TMPDIR"
ray status || { echo "[Ray] failed to start" >&2; exit 1; }

# Cleanup ray processes on script exit, error, or interruption (Ctrl+C)
cleanup_ray() {
  echo "[Ray] Stopping Ray head node..."
  # Use the specific temp dir to stop only this instance
  ray stop --temp-dir "$RAY_TMPDIR" >/dev/null 2>&1 || true

  # Fallback: kill anything lingering related to this specific run
  pgrep -f "$RAY_TMPDIR" | xargs -r kill -9 >/dev/null 2>&1 || true

  # Securely remove the temp directory to wipe logs/traces
  rm -rf "$RAY_TMPDIR"
}
trap cleanup_ray EXIT
trap 'cleanup_ray; exit 130' INT
trap 'cleanup_ray; exit 143' TERM
trap 'cleanup_ray; exit 1' ERR
# ------------------------------------------------------------

# Use trainer validation-only generation path for async vLLM compatibility.
python3 -m verl.trainer.main_ppo \
    --config-name ppo_ttrl \
    reward_model.use_reward_loop=False \
    data.train_files="$DATA_PATH" \
    data.val_files="$DATA_PATH" \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=3072 \
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
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n="$N_SAMPLES" \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name=trace_collection \
    trainer.experiment_name="trace-${MODEL_NAME}-${DATE}-${TIME_TAG}" \
    trainer.validation_data_dir="$VAL_DUMP_DIR" \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.max_critic_ckpt_to_keep=0 \
    trainer.default_local_dir="$OUT_DIR" \
    trainer.total_epochs=1 \
    2>&1 | tee "$LOG_FILE"

python3 examples/hrlib/val_jsonl_to_trace_parquet.py \
    --source_parquet "$DATA_PATH" \
    --val_jsonl_dir "$VAL_DUMP_DIR" \
    --n_samples "$N_SAMPLES" \
    --out_parquet "$TRACE_PARQUET" 2>&1 | tee -a "$LOG_FILE"

if [[ "$SCORE_AFTER_COLLECT" == "1" ]]; then
  python3 examples/hrlib/score_traces.py \
      --in_parquet "$TRACE_PARQUET" \
      --out_parquet "$LABELED_PARQUET" 2>&1 | tee -a "$LOG_FILE"
fi

#!/usr/bin/env bash
# Stage 0 HRLib evaluation (validation-only main_ppo).
#
# Mirrors examples/test_time_training/evaluate_intuitor.sh but takes MODEL_PATH,
# DATA_VAL, OUTPUT_DIR, N_SAMPLES, NUM_GPUS, CUDA_VISIBLE_DEVICES from the env so
# the SAME script runs both the vanilla baseline and the flat_top6 HRLib-injected
# variant — only DATA_VAL and OUTPUT_DIR change between the A/B runs.
#
# DO NOT remove the Ray isolation block (CLAUDE.md invariant).
#
# --- Why can two ``0.jsonl`` dumps differ in *problem* count? ---
#
# The trainer appends one JSONL line per *rollout* (``val_kwargs.n`` per val row).
# The number of distinct problems in a file is ``len(JSONL) / n`` when every row is
# evaluated with the same ``n`` and nothing is dropped.
#
# Rows can be *missing* vs another eval if the **validation dataset** differs:
#
# - ``data.val_files`` / ``data.val_max_samples`` — not the same parquet or subsample.
# - ``data.filter_overlong_prompts`` + ``data.max_prompt_length`` — longer prompts
#   (e.g. HRLib-injected parquet) tokenize longer; a stricter cap drops more rows.
#   A different ``actor_rollout_ref.model.path`` can use a different tokenizer and
#   change which prompts pass the length filter.
# - An incomplete validation run (crash / OOM) yields a partial JSONL.
#
# Lift metrics in ``hrlib_abstraction_lift.py`` aggregate over **all** baseline grouping keys;
# treated rows missing for a key are scored as fail. Keys use
# ``scripts.analyze._problem_group_key`` (``uid`` if present in JSONL, else ``data_source`` + user
# turn). The dataset ``uid`` is normally **not** written into validation dumps. Use the same
# ``val_files`` and data knobs across runs when you need identical coverage.

set -x

export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
PYTHONUNBUFFERED=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,4}  # change as needed
NUM_GPUS=${NUM_GPUS:-4}

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

# ---- required / defaulted inputs ----
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B-Base"}
DATA_VAL=${DATA_VAL:-"$HOME/data/MATH-500/test.parquet"}
N_SAMPLES=${N_SAMPLES:-32}
DATA_TRAIN=${DATA_TRAIN:-"$HOME/data/math/train.parquet"}
ROLLOUT_TP=${ROLLOUT_TP:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"hrlib-eval-${DATE}_${TIME_TAG}"}

# OUTPUT_DIR: explicit env wins; otherwise derive from MODEL_PATH using the
# same convention as evaluate_intuitor.sh but under the current $USER.
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "$MODEL_PATH" == Qwen/* ]]; then
    OUTPUT_DIR="/raid/$USER/eval/hrlib/base/${MODEL_PATH##*/}"
  else
    _MODEL_DIR=$(dirname "$MODEL_PATH")
    OUTPUT_DIR="/raid/$USER/eval/hrlib/${_MODEL_DIR#/raid/$USER/checkpoints/}"
  fi
fi
echo "OUTPUT_DIR=$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/evaluation.log"

# ---------------- Ray isolation (single-node) ----------------
unset RAY_ADDRESS
unset RAY_NAMESPACE

RUN_ID="${DATE}_${TIME_TAG}_$$"  # $$ is the current PID
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

echo "[Ray] RAY_ADDRESS=$RAY_ADDRESS"
echo "[Ray] RAY_TMPDIR=$RAY_TMPDIR"
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
trap 'cleanup_ray; exit 1'   ERR
# ------------------------------------------------------------


# use the trainer for evaluation only; longer max_prompt_length accommodates the
# injected HRLib prefix (~700 chars for top_k=6).
python3 -m verl.trainer.main_ppo \
    reward_model.use_reward_loop=False \
    reward_model.reward_manager=naive \
    data.train_files="$DATA_TRAIN" \
    data.val_files="$DATA_VAL" \
    data.train_batch_size=128 \
    data.max_prompt_length=1536 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$N_SAMPLES \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name=hrlib_eval \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.save_freq=2000000 \
    trainer.test_freq=10 \
    trainer.validation_data_dir="$OUTPUT_DIR" \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.max_critic_ckpt_to_keep=0 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.total_epochs=0 2>&1 | tee "$LOG_FILE"

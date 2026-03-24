set -x

export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,4  # change this as needed
NUM_GPUS=2

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)
# MODEL_PATH=/raid/changl8/checkpoints/ttrl/Qwen3-1.7B-Base/0323-221418/global_step_58/merged_hf_model  # cannot end with /
MODEL_PATH=Qwen/Qwen3-1.7B-Base
N_SAMPLES=32

if [[ "$MODEL_PATH" == Qwen/* ]]; then  # evaluate base model
  OUTPUT_DIR="/raid/changl8/eval/base/${MODEL_PATH##*/}"
else  # evaluate checkpoints
  OUTPUT_DIR=$(dirname "$MODEL_PATH")
  OUTPUT_DIR="/raid/changl8/eval/ttrl-verl/${OUTPUT_DIR#/raid/changl8/checkpoints/}"
fi
echo "OUTPUT_DIR=$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/evaluation.log"

# ---------------- Ray isolation (single-node) ----------------
unset RAY_ADDRESS
unset RAY_NAMESPACE

RUN_ID="${DATE}_${TIME_TAG}_$$"  # $$ is the current PID
export RAY_TMPDIR="/raid/changl8/ray/${RUN_ID}"
mkdir -p "$RAY_TMPDIR"

# Find a free port and start a dedicated local Ray head.
NODE_IP=$(hostname -I | awk '{print $1}')
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
  # Only kill processes associated with this run's temp dir.
  pgrep -f "$RAY_TMPDIR" | xargs -r kill >/dev/null 2>&1 || true
  sleep 1
  pgrep -f "$RAY_TMPDIR" | xargs -r kill -9 >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT
trap 'cleanup_ray; exit 130' INT
trap 'cleanup_ray; exit 143' TERM
trap 'cleanup_ray; exit 1' ERR
# ------------------------------------------------------------

# Use the trainer's validation-only path with the TTRL config overlay.
python3 -m verl.trainer.main_ppo \
    --config-name ppo_ttrl \
    reward_model.use_reward_loop=False \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/MATH-500/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$N_SAMPLES \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name=eval_ttrl \
    trainer.experiment_name="ttrl-eval-${DATE}-${TIME_TAG}" \
    trainer.save_freq=2000000 \
    trainer.test_freq=10 \
    trainer.validation_data_dir=$OUTPUT_DIR \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.max_critic_ckpt_to_keep=0 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_epochs=1 2>&1 | tee "$LOG_FILE"

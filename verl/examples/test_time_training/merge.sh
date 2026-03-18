set -x

LOCAL_DIR=/raid/xinyul2/checkpoints/grpo-intuitor/Qwen3-4B-Base/0307-135909/global_step_58/actor/
TARGET_DIR="$(dirname "$LOCAL_DIR")/merged_hf_model"


python -m scripts.legacy_model_merger merge \
    --backend fsdp \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
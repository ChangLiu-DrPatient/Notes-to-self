set -x

LOCAL_DIR=${LOCAL_DIR:-"/raid/changl9/checkpoints/grpo-naive-rewritten/Qwen3-1.7B-Base/0513-130124/global_step_36/actor/"}
TARGET_DIR="$(dirname "$LOCAL_DIR")/merged_hf_model"


python -m scripts.legacy_model_merger merge \
    --backend fsdp \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
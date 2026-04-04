set -x

LOCAL_DIR=/raid/changl8/checkpoints/ttrl/Llama-3.2-1B-Instruct/0402-125833/global_step_58/actor
TARGET_DIR="$(dirname "$LOCAL_DIR")/merged_hf_model"


python -m scripts.legacy_model_merger merge \
    --backend fsdp \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
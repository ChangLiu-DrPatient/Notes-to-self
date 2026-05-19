# original model on original and injected data
MODEL_PATH="$MODEL_PATH" \
OUT_DIR="$EVAL_ROOT/base-model/" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

DATA_VAL="$HOME/data/MATH-500/$MODEL_NAME/test_abstraction_gated.parquet" \
MODEL_PATH="$MODEL_PATH" \
OUT_DIR="$EVAL_ROOT/base-model-inject-gated/" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

# GRPO on original data
TRAIN_DATA_PATH="$HOME/data/math/train.parquet" \
MODEL_PATH="$MODEL_PATH" \
OUT_DIR="$CHECKPOINT_ROOT/grpo" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/00_grpo.sh

LOCAL_DIR="$CHECKPOINT_ROOT/grpo/global_step_58/actor" \
bash examples/test_time_training/merge.sh

MODEL_PATH="$CHECKPOINT_ROOT/grpo/global_step_58/merged_hf_model" \
OUT_DIR="$EVAL_ROOT/grpo" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

DATA_VAL="$HOME/data/MATH-500/$MODEL_NAME/test_abstraction_gated.parquet" \
MODEL_PATH="$CHECKPOINT_ROOT/grpo/global_step_58/merged_hf_model" \
OUT_DIR="$EVAL_ROOT/grpo-inject-gated" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

# GRPO on gated data
TRAIN_DATA_PATH="$HOME/data/math/$MODEL_NAME/train_abstraction_gated.parquet" \
MODEL_PATH="$MODEL_PATH" \
OUT_DIR="$CHECKPOINT_ROOT/grpo-injected" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/00_grpo.sh

LOCAL_DIR="$CHECKPOINT_ROOT/grpo-injected/global_step_58/actor" \
bash examples/test_time_training/merge.sh

MODEL_PATH="$CHECKPOINT_ROOT/grpo-injected/global_step_58/merged_hf_model" \
OUT_DIR="$EVAL_ROOT/grpo-injected" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

DATA_VAL="$HOME/data/MATH-500/$MODEL_NAME/test_abstraction_gated.parquet" \
MODEL_PATH="$CHECKPOINT_ROOT/grpo-injected/global_step_58/merged_hf_model" \
OUT_DIR="$EVAL_ROOT/grpo-injected-inject-gated" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh


# GRPO on Qwen3-1.7B-Base gated data
TRAIN_DATA_PATH="$HOME/data/math/Qwen3-1.7B-Base/train_abstraction_re_gated.parquet" \
MODEL_PATH="$MODEL_PATH" \
OUT_DIR="$CHECKPOINT_ROOT/grpo-qwen" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/00_grpo.sh

LOCAL_DIR="$CHECKPOINT_ROOT/grpo-qwen/global_step_58/actor" \
bash examples/test_time_training/merge.sh

MODEL_PATH="$CHECKPOINT_ROOT/grpo-qwen/global_step_58/merged_hf_model" \
OUT_DIR="$EVAL_ROOT/grpo-qwen" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

DATA_VAL="$HOME/data/MATH-500/Qwen3-1.7B-Base/test_abstraction_re_gated.parquet" \
MODEL_PATH="$CHECKPOINT_ROOT/grpo-qwen/global_step_58/merged_hf_model" \
OUT_DIR="$EVAL_ROOT/grpo-qwen-inject-gated" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

# cross eval
DATA_VAL="$HOME/data/MATH-500/Qwen3-1.7B-Base/test_abstraction_re_gated.parquet" \
MODEL_PATH="$CHECKPOINT_ROOT/grpo-injected/global_step_58/merged_hf_model" \
OUT_DIR="$EVAL_ROOT/grpo-injected-cross" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

DATA_VAL="$HOME/data/MATH-500/$MODEL_NAME/test_abstraction_gated.parquet" \
MODEL_PATH="$CHECKPOINT_ROOT/grpo-qwen/global_step_58/merged_hf_model" \
OUT_DIR="$EVAL_ROOT/grpo-qwen-cross" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh

DATA_VAL="$HOME/data/MATH-500/Qwen3-1.7B-Base/test_abstraction_re_gated.parquet" \
MODEL_PATH="$MODEL_PATH" \
OUT_DIR="$EVAL_ROOT/base-model-cross" \
CUDA_VISIBLE_DEVICES="0,1,2,4" \
NUM_GPUS=4 \
GPU_MEM=0.8 \
MICRO_BSZ=2 \
bash examples/hrlib/40_eval.sh


# Compare
python3 examples/hrlib/evaluate_results.py lift -- \
--baseline "$EVAL_ROOT/base-model/0.jsonl" \
--treated "$EVAL_ROOT/base-model-inject-gated/0.jsonl"

python3 examples/hrlib/evaluate_results.py lift -- \
--baseline "$EVAL_ROOT/grpo/0.jsonl" \
--treated "$EVAL_ROOT/grpo-injected/0.jsonl"

python3 examples/hrlib/evaluate_results.py lift -- \
--baseline "$EVAL_ROOT/grpo/0.jsonl" \
--treated "$EVAL_ROOT/grpo-qwen/0.jsonl"

python3 examples/hrlib/evaluate_results.py lift -- \
--baseline "$EVAL_ROOT/grpo-injected/0.jsonl" \
--treated "$EVAL_ROOT/grpo-injected-inject-gated/0.jsonl"

python3 examples/hrlib/evaluate_results.py lift -- \
--baseline "$EVAL_ROOT/grpo-qwen/0.jsonl" \
--treated "$EVAL_ROOT/grpo-qwen-inject-gated/0.jsonl"


# cross eval
python3 examples/hrlib/evaluate_results.py lift -- \
--baseline "$EVAL_ROOT/base-model/0.jsonl" \
--treated "$EVAL_ROOT/base-model-cross/0.jsonl"

python3 examples/hrlib/evaluate_results.py lift -- \
--baseline "$EVAL_ROOT/grpo-injected-inject-gated/0.jsonl" \
--treated "$EVAL_ROOT/grpo-injected-cross/0.jsonl"

python3 examples/hrlib/evaluate_results.py lift -- \
--baseline "$EVAL_ROOT/grpo-injected-inject-gated/0.jsonl" \
--treated "$EVAL_ROOT/grpo-qwen-cross/0.jsonl"

# judge abstraction use
MODEL="deepseek/deepseek-v4-flash" \
FALLBACK_MODEL="deepseek/deepseek-v4-flash" \
VAL_JSONL="$EVAL_ROOT/base-model-inject-gated/0.jsonl" \
OUT_DIR="$EVAL_ROOT/base-model-inject-gated/judge-abs-use" \
bash examples/hrlib/judge_abstraction_use.sh

MODEL="deepseek/deepseek-v4-flash" \
FALLBACK_MODEL="deepseek/deepseek-v4-flash" \
VAL_JSONL="$EVAL_ROOT/grpo-injected-inject-gated/0.jsonl" \
OUT_DIR="$EVAL_ROOT/grpo-injected-inject-gated/judge-abs-use" \
bash examples/hrlib/judge_abstraction_use.sh
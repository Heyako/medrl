#!/bin/bash
set -euo pipefail

# MedRL GRPO Training Launcher
# Usage:
#   Single GPU:        bash scripts/run_train.sh
#   Multi-GPU (4x):    NUM_GPUS=4 bash scripts/run_train.sh
#   With DeepSpeed:    DEEPSPEED_CONFIG=configs/deepspeed_zero3.json bash scripts/run_train.sh

echo "=== MedRL GRPO Training ==="

NUM_GPUS="${NUM_GPUS:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.json}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-data/processed/prm_verified.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set — LLM judge will return neutral scores (0.5)."
    echo "         Set it to enable the full composite reward function."
fi

echo "Configuration:"
echo "  GPUs:           $NUM_GPUS"
echo "  DeepSpeed:      $DEEPSPEED_CONFIG"
echo "  Model:          $MODEL_NAME"
echo "  Train Data:     $TRAIN_DATA"
echo "  Output:         $OUTPUT_DIR"

# DeepSpeed launch
if [ "$NUM_GPUS" -gt 1 ]; then
    deepspeed \
        --num_gpus="$NUM_GPUS" \
        scripts/train_grpo.py \
        --model_name_or_path "$MODEL_NAME" \
        --train_data "$TRAIN_DATA" \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --output_dir "$OUTPUT_DIR"
else
    python scripts/train_grpo.py \
        --model_name_or_path "$MODEL_NAME" \
        --train_data "$TRAIN_DATA" \
        --output_dir "$OUTPUT_DIR"
fi

echo "=== Training complete. Checkpoints saved to $OUTPUT_DIR ==="

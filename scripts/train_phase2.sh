#!/bin/bash
# ============================================================================
# MedRL Phase 2: GRPO Training on Qwen2.5-7B with DeepSpeed ZeRO-3 + LoRA
# Hardware: RTX A6000 48GB (single GPU)
# Judge:    DeepSeek API
# ============================================================================
set -euo pipefail

echo "=== MedRL Phase 2: GRPO + LoRA + ZeRO-3 on 7B ==="

# ── Configuration ──
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero3_7b.json}"
TRAIN_DATA="${TRAIN_DATA:-data/raw/medqa_us_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase2}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"

if [ -z "$JUDGE_API_KEY" ]; then
    echo "ERROR: JUDGE_API_KEY is required for Phase 2 LLM judge."
    echo "  export JUDGE_API_KEY=sk-..."
    exit 1
fi

export HF_ENDPOINT
export JUDGE_API_KEY
export JUDGE_BASE_URL
export JUDGE_MODEL

echo ""
echo "Configuration:"
echo "  Model:          $MODEL"
echo "  DeepSpeed:      $DEEPSPEED_CONFIG"
echo "  Judge:          $JUDGE_MODEL @ $JUDGE_BASE_URL"
echo "  Train Data:     $TRAIN_DATA"
echo "  Output:         $OUTPUT_DIR"
echo ""

# ── Step 1: Verify environment ──
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
t = torch.cuda.get_device_properties(0)
print(f'GPU: {t.name} ({t.total_mem/1e9:.1f} GB)')
" || { echo "GPU check failed"; exit 1; }

# ── Step 2: Launch training ──
# Single GPU: Python script directly (no deepspeed launcher needed for single-GPU ZeRO-3 offload)
python scripts/train_phase2.py \
    --model "$MODEL" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --group_size 4 \
    --batch_size 2 \
    --max_steps 10 \
    --lora_r 16 \
    --lora_alpha 32

echo ""
echo "=== Phase 2 training complete. Check $OUTPUT_DIR ==="

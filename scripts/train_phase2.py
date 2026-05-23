#!/usr/bin/env python3
"""
============================================================================
MedRL Phase 2: GRPO + LoRA + DeepSpeed ZeRO-3 on Qwen2.5-7B
============================================================================

Target: RTX A6000 48GB (single GPU)
Model:  Qwen2.5-7B-Instruct with LoRA (train ~2% of params)
Judge: DeepSeek API (or any OpenAI-compatible endpoint)

Memory budget (7B, FP16):
  Policy (LoRA)  ~14 GB
  Reference       ~14 GB
  Optimizer (LoRA-only, ZeRO-3 offload) ~2 GB
  Activations (gradient checkpointed)    ~8 GB
  Group responses (G=4)                  ~6 GB
  ─────────────────────────────────
  Total                                  ~44 GB  (fits 48GB)

Usage:
    export JUDGE_API_KEY=sk-...
    python scripts/train_phase2.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --deepspeed configs/deepspeed_zero3_7b.json \
        --max_steps 10
============================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medrl.rl.grpo_trainer import GRPOTrainer, GRPOConfig
from medrl.rl.reward_func import CompositeReward
from medrl.utils.logging import setup_logger

logger = setup_logger("phase2")


def get_lora_config(args) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )


def load_model_with_lora(model_name: str, device: str, args):
    """Load 7B model with LoRA adapters, optimize for 48GB VRAM."""
    logger.info(f"Loading model: {model_name}")

    # Detect best attention — try flash-attn, fallback to sdpa
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Using FlashAttention-2")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("Using SDPA (built-in)")

    # Base model in FP16 with gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        use_cache=False,  # Disable KV cache during training
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply LoRA
    lora_config = get_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    logger.info(f"Model loaded. Device: {device}")
    return model, tokenizer


def load_reference_model(model_name: str, device: str, args):
    """Load frozen reference model for KL computation."""
    logger.info("Loading reference model...")

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    logger.info("Reference model loaded and frozen.")
    return ref_model


def parse_args():
    p = argparse.ArgumentParser(description="MedRL Phase 2 Training")
    # Model
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # Training
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--kl_beta", type=float, default=0.01)
    p.add_argument("--target_kl", type=float, default=0.02)
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default="configs/deepspeed_zero3_7b.json")
    # Data
    p.add_argument("--train_data", type=str, default="data/raw/medqa_us_train.jsonl")
    p.add_argument("--output_dir", type=str, default="outputs/phase2")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
    traceback (most recent call last):
  File "/root/medrl/scripts/train_phase2.py", line 162, in main
    logger.info(f"GPU: {props.name} ({props.total_mem / 1e9:.1f} GB)")

    # ── Load models ──
    model, tokenizer = load_model_with_lora(args.model, device, args)
    ref_model = load_reference_model(args.model, device, args)

    # ── Reward function with DeepSeek Judge ──
    judge_api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    judge_base_url = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
    judge_model = os.environ.get("JUDGE_MODEL", "deepseek-chat")

    reward_func = CompositeReward(
        api_key=judge_api_key,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        use_judge=bool(judge_api_key),
        w_format=0.15,
        w_judge=0.70,
        w_diversity=0.15,
    )

    if not judge_api_key:
        logger.warning("No JUDGE_API_KEY set — falling back to heuristic scoring!")
        logger.warning("Phase 2 requires LLM Judge for reward discriminability.")
        logger.warning("Export: JUDGE_API_KEY=sk-...")

    # ── GRPO Config ──
    config = GRPOConfig(
        group_size=args.group_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        target_kl=args.target_kl,
        clip_epsilon=0.2,
        ppo_epochs=1,
        log_interval=5,
        save_interval=200,
    )

    # ── Optimizer: only LoRA params ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # ── Trainer ──
    trainer = GRPOTrainer(
        policy_model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_func=reward_func,
        config=config,
        optimizer=optimizer,
        device=device,
    )

    # ── Smoke test ──
    logger.info("Running smoke test (1 step)...")
    test_prompts = [
        "Question: A 55-year-old male presents with crushing chest pain, "
        "ST elevation in II/III/aVF. What is the most likely diagnosis?\n\n"
        "Options:\nA. Pericarditis\nB. Inferior STEMI\nC. Unstable angina\n"
        "D. Aortic dissection\n\n"
        "Please reason step by step in <thinking>...</thinking> tags, "
        "then provide your final answer in <answer>...</answer> tags."
    ] * args.batch_size

    try:
        out = trainer.step(test_prompts)
        logger.info(
            f"Smoke test: loss={out.loss:.4f}, KL={out.kl_divergence:.5f}, "
            f"reward_mean={out.mean_reward:.3f}, reward_std={out.reward_std:.3f}, "
            f"format_violation={out.format_violation_rate:.2%}"
        )
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        raise

    logger.info("Phase 2 pipeline functional. Ready for full training.")
    logger.info(
        f"To start: python scripts/train_phase2.py "
        f"--max_steps {args.max_steps}"
    )


if __name__ == "__main__":
    main()

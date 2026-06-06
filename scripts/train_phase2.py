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
import random
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
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--skip_smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")

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
        w_format=0.2,
        w_judge=0.65,
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

    # ── Load training data ──
    from medrl.data.dataset_loader import load_medqa, format_mc_question, format_mc_answer
    from medrl.data.dataset_loader import load_medmcqa

    logger.info(f"Loading training data: {args.train_data}")
    try:
        if 'medmcqa' in args.train_data:
            train_samples = load_medmcqa(args.train_data)
        else:
            train_samples = load_medqa(args.train_data)
    except Exception:
        logger.warning("MedQA loader failed, trying MedMCQA...")
        train_samples = load_medmcqa(args.train_data)

    logger.info(f"Loaded {len(train_samples)} training samples")

    # Build prompts from dataset
    def build_prompt(sample):
        q = format_mc_question(sample)
        a = format_mc_answer(sample)
        return (
            f"{q}\n\n"
            f"Correct answer: {a}\n\n"
            f"Explain step-by-step why this is the correct answer and why "
            f"each alternative is wrong. Use <thinking>...</thinking> for "
            f"reasoning and <answer>...</answer> for the final answer."
        )

    all_train_prompts = [build_prompt(s) for s in train_samples]
    logger.info(f"Built {len(all_train_prompts)} training prompts")

    # ── Training loop ──
    import random
    logger.info(f"Starting GRPO training: {args.max_steps} steps, "
                f"batch_size={args.batch_size}, group_size={args.group_size}")

    for step in range(args.max_steps):
        # Sample batch of prompts WITH correct answers for reward correctness check
        batch_indices = random.sample(range(len(all_train_prompts)), min(args.batch_size, len(all_train_prompts)))
        batch = [all_train_prompts[i] for i in batch_indices]
        batch_answers = [train_samples[i]['answer_text'] for i in batch_indices]
        # Expand answers to match group_size
        trainer._current_correct_answers = [a for a in batch_answers for _ in range(args.group_size)]

        try:
            out = trainer.step(batch)
            logger.info(
                f"Step {step+1}/{args.max_steps}: "
                f"loss={out.loss:.4f}, reward_mean={out.mean_reward:.3f}, "
                f"reward_std={out.reward_std:.3f}, "
                f"format_ok={1-out.format_violation_rate:.0%}"
            )
        except Exception as e:
            logger.error(f"Step {step+1} failed: {e}")
            raise

        # Checkpoint
        if (step + 1) % args.save_interval == 0:
            model.save_pretrained(f"{args.output_dir}/step_{step+1}")
            logger.info(f"Checkpoint saved: step {step+1}")

    # Final save
    model.save_pretrained(f"{args.output_dir}/final")
    logger.info(f"Training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()

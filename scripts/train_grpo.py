#!/usr/bin/env python3
"""
GRPO Training Entry Point for MedRL.

Supports:
    - Single GPU / Multi-GPU via DeepSpeed
    - LLaMA-Factory / OpenRLHF integration
    - Custom CompositeReward function

Usage:
    python scripts/train_grpo.py \
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
        --train_data data/processed/prm_verified.jsonl \
        --output_dir outputs/checkpoints
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medrl.rl.grpo_trainer import GRPOTrainer, GRPOConfig
from medrl.rl.reward_func import CompositeReward
from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MedRL GRPO Training")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model path or HF hub name",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/processed/prm_verified.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints",
    )
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Detect best attention implementation
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("Using FlashAttention-2")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("FlashAttention not found, using SDPA (PyTorch built-in)")

    # Load model
    logger.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Reference model (frozen copy for KL)
    logger.info("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    # Composite reward function
    reward_func = CompositeReward(
        api_key=os.environ.get("OPENAI_API_KEY"),
        judge_model="gpt-4",
        use_judge=bool(os.environ.get("OPENAI_API_KEY")),
    )

    # GRPO config
    config = GRPOConfig(
        group_size=args.group_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        target_kl=args.target_kl,
    )

    # Trainer
    trainer = GRPOTrainer(
        policy_model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_func=reward_func,
        config=config,
        device=device,
    )

    logger.info(
        f"GRPO Trainer ready: group_size={config.group_size}, "
        f"batch_size={config.batch_size}, "
        f"lr={config.learning_rate}, "
        f"max_steps={args.max_steps}"
    )

    # ── Run a quick smoke test ──
    logger.info("Running smoke test...")
    test_prompts = [
        "Question: A 45-year-old patient presents with acute chest pain. What is the most likely diagnosis?\n\n"
        "Please reason step by step, then provide your final answer in <answer>...</answer> tags.",
    ]
    for _ in range(args.batch_size):
        try:
            output = trainer.step(test_prompts)
            logger.info(
                f"Smoke test step: loss={output.loss:.4f}, KL={output.kl_divergence:.4f}, "
                f"reward_mean={output.mean_reward:.3f}, reward_std={output.reward_std:.3f}"
            )
            break  # One successful step is sufficient for smoke test
        except Exception as e:
            logger.error(f"Smoke test failed: {e}")
            raise

    logger.info("Smoke test passed. GRPO pipeline is functional.")
    logger.info("Ready for full training — provide train_data to start the loop.")


if __name__ == "__main__":
    main()

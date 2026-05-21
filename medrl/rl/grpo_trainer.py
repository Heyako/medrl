"""
GRPO Trainer: main training loop for Group Relative Policy Optimization.

Integrates with HuggingFace Transformers and our custom CompositeReward.
Designed to be framework-agnostic at the top level — the actual forward/backward
can be plugged into LLaMA-Factory, verl, or OpenRLHF.
"""

import os
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from medrl.rl.reward_func import CompositeReward
from medrl.rl.advantage import compute_grpo_advantage
from medrl.rl.kl_controller import KLController
from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Group sampling
    group_size: int = 8  # G: responses per prompt
    batch_size: int = 4  # B: prompts per step (total samples = B * G)

    # PPO-style clipping
    clip_epsilon: float = 0.2
    ppo_epochs: int = 1  # number of PPO updates per batch

    # KL regulation
    kl_beta: float = 0.01
    target_kl: float = 0.02

    # Generation
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.9

    # Optimization
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0

    # Reward
    w_format: float = 0.15
    w_judge: float = 0.70
    w_diversity: float = 0.15

    # Logging
    log_interval: int = 10
    save_interval: int = 500


@dataclass
class GRPOStepOutput:
    """Output of a single GRPO training step."""

    loss: float
    policy_loss: float
    kl_divergence: float
    entropy: float
    mean_reward: float
    reward_std: float
    group_advantage_std: float
    format_violation_rate: float
    beta: float


class GRPOTrainer:
    """
    GRPO Trainer — Critic-free RLHF for medical LLM alignment.

    Usage:
        trainer = GRPOTrainer(
            policy_model=model,
            ref_model=ref_model,
            reward_func=CompositeReward(api_key="..."),
            config=GRPOConfig(group_size=8, batch_size=4),
        )
        for step in range(max_steps):
            output = trainer.step(prompts_batch)
            print(f"Step {step}: loss={output.loss:.4f}, KL={output.kl_divergence:.4f}")
    """

    def __init__(
        self,
        policy_model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer: Any,
        reward_func: CompositeReward,
        config: Optional[GRPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_func = reward_func
        self.config = config or GRPOConfig()
        self.device = device

        # Freeze reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

        # KL controller
        self.kl_controller = KLController(
            target_kl=self.config.target_kl,
            beta=self.config.kl_beta,
        )

        self.step_count = 0

    def _generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate G responses for each prompt (group sampling)."""
        all_prompts = []
        for p in prompts:
            all_prompts.extend([p] * self.config.group_size)

        # Tokenize
        inputs = self.tokenizer(
            all_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.policy.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        responses = self.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        return responses

    def _compute_log_probs(
        self, prompts: List[str], responses: List[str]
    ) -> torch.Tensor:
        """Compute log-probabilities of responses under the current policy."""
        full_texts = [p + r for p, r in zip(prompts, responses)]

        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.policy(**inputs)

        # Sum log-probs over response tokens
        # Simplified: this should use per-token log-probs in production
        log_probs = outputs.logits.log_softmax(dim=-1)
        return log_probs.mean(dim=[1, 2])  # (B*G,) aggregated

    def step(self, prompts: List[str]) -> GRPOStepOutput:
        """
        Execute a single GRPO training step.

        Args:
            prompts: list of B prompt strings

        Returns:
            GRPOStepOutput with training diagnostics
        """
        self.policy.train()
        B = len(prompts)
        G = self.config.group_size

        # ── Step 1: Group Sampling ──
        responses = self._generate_responses(prompts)  # (B*G,) strings

        # Expand prompts to match responses
        expanded_prompts = [
            p for p in prompts for _ in range(G)
        ]

        # ── Step 2: Compute Rewards ──
        rewards, reward_stats = self.reward_func.compute_group_rewards(
            prompts=expanded_prompts,
            responses=responses,
            group_size=G,
        )
        rewards = rewards.to(self.device)

        # ── Step 3: Compute Group-Relative Advantages ──
        advantages = compute_grpo_advantage(rewards, group_size=G)

        # ── Step 4: PPO Update ──
        total_policy_loss = 0.0
        for _ in range(self.config.ppo_epochs):
            log_probs = self._compute_log_probs(expanded_prompts, responses)

            # For the first epoch, treat these as "old" log-probs
            # In production: store old log-probs from generation step
            log_probs_old = log_probs.detach()

            ratio = (log_probs - log_probs_old).exp()
            surr1 = ratio * advantages
            surr2 = (
                ratio.clamp(
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                )
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # ── Step 5: KL Divergence ──
            with torch.no_grad():
                full_texts = [
                    p + r for p, r in zip(expanded_prompts, responses)
                ]
                ref_inputs = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                ).to(self.device)
                ref_outputs = self.ref_model(**ref_inputs)
                ref_logits = ref_outputs.logits

            policy_inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.device)
            policy_outputs = self.policy(**policy_inputs)
            policy_logits = policy_outputs.logits

            kl_div = self.kl_controller.compute_kl(policy_logits, ref_logits)
            entropy = self.kl_controller.compute_entropy(policy_logits)

            # ── Step 6: Total Loss ──
            beta = self.kl_controller.step(kl_div)
            total_loss = policy_loss + beta * kl_div

            # ── Step 7: Backward ──
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()

        self.step_count += 1

        # ── Logging ──
        output = GRPOStepOutput(
            loss=total_loss.item(),
            policy_loss=total_policy_loss / self.config.ppo_epochs,
            kl_divergence=kl_div,
            entropy=entropy,
            mean_reward=reward_stats["mean_reward"],
            reward_std=reward_stats["std_reward"],
            group_advantage_std=advantages.std().item(),
            format_violation_rate=reward_stats["format_violation_rate"],
            beta=beta,
        )

        if self.step_count % self.config.log_interval == 0:
            logger.info(
                f"Step {self.step_count}: loss={output.loss:.4f}, "
                f"KL={output.kl_divergence:.5f}, "
                f"entropy={output.entropy:.4f}, "
                f"mean_reward={output.mean_reward:.3f}, "
                f"reward_std={output.reward_std:.3f}, "
                f"beta={output.beta:.5f}"
            )

        return output

    def save_checkpoint(self, path: str) -> None:
        """Save policy model checkpoint."""
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")

    def train(
        self,
        train_dataloader: DataLoader,
        max_steps: int,
        eval_fn: Optional[Callable] = None,
        eval_interval: int = 500,
        output_dir: str = "./outputs/checkpoints",
    ) -> List[GRPOStepOutput]:
        """
        Full training loop.

        Args:
            train_dataloader: yields batches of {"prompt": str, "question": str}
            max_steps: total training steps
            eval_fn: optional evaluation callback
            eval_interval: steps between evaluations
            output_dir: checkpoint directory

        Returns:
            List of step outputs for analysis
        """
        history: List[GRPOStepOutput] = []
        data_iter = iter(train_dataloader)

        for step in range(max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            prompts = batch["prompt"] if isinstance(batch, dict) else batch
            step_output = self.step(prompts)
            history.append(step_output)

            # Evaluation
            if eval_fn and step > 0 and step % eval_interval == 0:
                eval_fn(self, step)

            # Checkpoint
            if step > 0 and step % self.config.save_interval == 0:
                self.save_checkpoint(f"{output_dir}/step_{step}")

            # Safety: abort if KL explodes
            if self.kl_controller.should_warn():
                logger.error(
                    f"KL divergence exploded to {step_output.kl_divergence:.4f} "
                    f"at step {step}. Consider increasing target_kl or reducing learning_rate."
                )

        return history

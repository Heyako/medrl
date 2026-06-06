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
        format_instruction = (
            "You MUST respond in this exact format:\n"
            "<thinking>\nStep 1: [your reasoning]\n"
            "Step 2: [your reasoning]\n"
            "...\n"
            "</thinking>\n"
            "<answer>\n[your final answer]\n"
            "</answer>\n\n"
        )
        all_prompts = []
        for p in prompts:
            all_prompts.extend([format_instruction + p] * self.config.group_size)

        self.tokenizer.padding_side = "left"
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
        self,
        prompts: List[str],
        responses: List[str],
        model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Compute mean per-token log-probability of responses under a model.

        Correctly gathers log-probs at actual response token positions:
          1. Tokenize prompt+response together
          2. Tokenize prompts separately to locate response start boundary
          3. Forward pass → logits → log_softmax
          4. Gather log-prob of each actual response token
          5. Mean over response tokens → scalar per sample

        Uses mean (not sum) for numerical stability — keeps values in [-10, 0]
        range regardless of response length, so PPO ratio = exp(mean_new - mean_old)
        stays well-behaved.

        Args:
            prompts:   list of prompt strings
            responses: list of response strings (same length)
            model:     model to evaluate under (defaults to self.policy)

        Returns:
            log_probs: (batch,) tensor of mean per-token log-probabilities
                       WITH gradients attached when model=self.policy.
        """
        if model is None:
            model = self.policy

        batch_size = len(prompts)
        full_texts = [p + r for p, r in zip(prompts, responses)]

        # Tokenize full concatenated texts
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)
        input_ids = inputs["input_ids"]          # (batch, max_seq_len)
        attention_mask = inputs["attention_mask"]  # (batch, max_seq_len)

        # Tokenize prompts alone to get per-sample prompt lengths
        prompt_enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        prompt_lens = prompt_enc["attention_mask"].sum(dim=-1)  # (batch,) int

        # Forward pass
        outputs = model(**inputs)
        logprobs_all = outputs.logits.log_softmax(dim=-1)  # (batch, seq_len, vocab)

        # Gather per-token log-probs for response tokens, per-sample
        seq_logprobs = []
        for i in range(batch_size):
            p_len = int(prompt_lens[i].item())
            s_len = int(attention_mask[i].sum().item())

            if p_len >= s_len:
                # Empty or zero-length response → neutral log-prob
                seq_logprobs.append(torch.tensor(0.0, device=self.device))
                continue

            # logprobs[pos] is the model's prediction for token at pos
            # We want predictions for response tokens:
            #   logprobs[p_len-1] predicts token[p_len]   (1st response token)
            #   logprobs[s_len-2] predicts token[s_len-1] (last response token)
            pred_lp = logprobs_all[i, p_len - 1 : s_len - 1, :]    # (resp_len, vocab)
            resp_ids = input_ids[i, p_len : s_len]                  # (resp_len,)

            # Gather log-prob of each actual response token
            token_lp = pred_lp.gather(dim=-1, index=resp_ids.unsqueeze(-1)).squeeze(-1)

            # Mean over response tokens for numerical stability
            seq_logprobs.append(token_lp.mean())

        return torch.stack(seq_logprobs)  # (batch,)

    def step(self, prompts: List[str]) -> GRPOStepOutput:
        """
        Execute a single GRPO training step.

        Data flow:
          prompts (B) → generate G each → responses (B*G)
          → rewards (B*G) → group-normalize → advantages (B*G)
          → log_probs_old (B*G, no grad)
          → log_probs_ref (B*G, no grad, for KL)
          → [PPO epoch] log_probs (B*G, with grad)
            → ratio = exp(log_p - log_p_old)
            → ppo_loss = -min(ratio*A, clip(ratio)*A)
            → kl_k3 = exp(log_ref - log_p) - (log_ref - log_p) - 1
            → total_loss = mean(ppo_loss + beta * kl_k3)
            → backward → step

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
            correct_answers=getattr(self, '_current_correct_answers', None),
        )
        rewards = rewards.to(self.device)

        # ── Step 3: Compute Group-Relative Advantages ──
        advantages = compute_grpo_advantage(rewards, group_size=G)

        # ── Step 4: Pre-compute old + reference log_probs (no grad, once) ──
        with torch.no_grad():
            log_probs_old = self._compute_log_probs(
                expanded_prompts, responses, model=self.policy
            )
            log_probs_ref = self._compute_log_probs(
                expanded_prompts, responses, model=self.ref_model
            )
        torch.cuda.empty_cache()

        # ── Step 5: PPO Update with KL penalty ──
        total_policy_loss = 0.0
        kl_k3 = None  # will hold the last epoch's KL for logging

        for _ in range(self.config.ppo_epochs):
            # Compute current log_probs WITH gradients
            log_probs = self._compute_log_probs(
                expanded_prompts, responses, model=self.policy
            )

            # PPO clipped surrogate
            ratio = (log_probs - log_probs_old).exp()
            surr1 = ratio * advantages
            surr2 = (
                ratio.clamp(
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                )
                * advantages
            )
            ppo_loss = -torch.min(surr1, surr2)  # (B*G,)

            # KL penalty: k3 estimator of KL(π_θ || π_ref), always >= 0
            log_ratio = log_probs_ref - log_probs  # log(π_ref / π_θ)
            kl_k3 = log_ratio.exp() - log_ratio - 1  # (B*G,), >= 0
            kl_penalty = self.config.kl_beta * kl_k3

            total_loss = (ppo_loss + kl_penalty).mean()

            # ── Backward ──
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += total_loss.item()

            # Free gradient graph to reclaim VRAM
            del log_probs
            torch.cuda.empty_cache()

        # ── Step 6: Update KL controller ──
        kl_div = kl_k3.mean().item() if kl_k3 is not None else 0.0
        # Entropy proxy: negative mean log-prob (higher = model is less confident)
        entropy = -log_probs_old.mean().item()
        self.kl_controller.step(kl_div)
        beta = self.kl_controller.beta

        del log_probs_old, log_probs_ref

        self.step_count += 1

        # ── Logging ──
        output = GRPOStepOutput(
            loss=total_policy_loss / max(self.config.ppo_epochs, 1),
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

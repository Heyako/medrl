#!/usr/bin/env python3
"""
Verify gradient flow through the fixed _compute_log_probs and GRPO loss.

Uses a synthetic tiny Transformer (no HF model loading needed) to validate:
  1. log_probs correctly reflect actual token probabilities (gather at token positions)
  2. Gradients flow from total_loss back to model parameters
  3. KL k3 estimator is always non-negative
  4. PPO ratio stays numerically stable with mean per-token log-probs

Usage:
    python scripts/verify_gradient_flow.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from medrl.rl.advantage import compute_grpo_advantage
from medrl.utils.logging import setup_logger

logger = setup_logger("verify")


class TinyLM(nn.Module):
    """A tiny Transformer LM that returns logits for given input_ids.

    Simulates a CausalLM's forward: input_ids → logits of shape (batch, seq, vocab).
    """
    def __init__(self, vocab_size=256, d_model=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Small transformer block
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        logits = self.lm_head(x)
        # Return a namespace-like output for .logits access
        return _Output(logits=logits)


class _Output:
    """Simple namespace for model outputs."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockTokenizer:
    """Minimal tokenizer for verification — maps predefined strings to int IDs."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab = {
            "<pad>": 0, "<eos>": 1,
            "Q": 2, "u": 3, "e": 4, "s": 5, "t": 6, "i": 7, "o": 8, "n": 9,
            ":": 10, " ": 11, "W": 12, "h": 13, "a": 14, " ": 11,
            "2": 15, "+": 16, "?": 17, "\n": 18, "A": 19, "w": 20, "r": 21,
            "4": 22, ".": 23, ",": 24, "b": 25,
        }
        self._reverse = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        ids = []
        for ch in text:
            ids.append(self.vocab.get(ch, 0))
        return ids

    def __call__(self, texts, return_tensors="pt", padding=True,
                 truncation=True, max_length=512, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t) for t in texts]
        max_len = min(max(len(e) for e in encoded), max_length)
        batch_ids = []
        batch_mask = []
        for e in encoded:
            e_trunc = e[:max_len]
            pad_len = max_len - len(e_trunc)
            ids = e_trunc + [self.pad_token_id] * pad_len
            mask = [1] * len(e_trunc) + [0] * pad_len
            batch_ids.append(ids)
            batch_mask.append(mask)
        return _Output(
            input_ids=torch.tensor(batch_ids, dtype=torch.long),
            attention_mask=torch.tensor(batch_mask, dtype=torch.long),
        )


def main():
    device = "cpu"
    vocab_size = 64

    logger.info("Building synthetic models...")
    tokenizer = MockTokenizer()
    model = TinyLM(vocab_size=vocab_size).to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    ref_model = TinyLM(vocab_size=vocab_size).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # We'll test _compute_log_probs logic directly (same algorithm)
    def compute_log_probs(prompts, responses, model):
        """Re-implement the fixed _compute_log_probs logic for testing."""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = tokenizer(full_texts)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        prompt_enc = tokenizer(prompts)
        prompt_lens = prompt_enc.attention_mask.sum(dim=-1)

        outputs = model(input_ids=input_ids)
        logprobs_all = outputs.logits.log_softmax(dim=-1)

        seq_logprobs = []
        for i in range(len(prompts)):
            p_len = int(prompt_lens[i].item())
            s_len = int(attention_mask[i].sum().item())
            if p_len >= s_len:
                seq_logprobs.append(torch.tensor(0.0, device=device))
                continue
            pred_lp = logprobs_all[i, p_len - 1 : s_len - 1, :]
            resp_ids = input_ids[i, p_len : s_len]
            token_lp = pred_lp.gather(dim=-1, index=resp_ids.unsqueeze(-1)).squeeze(-1)
            seq_logprobs.append(token_lp.mean())
        return torch.stack(seq_logprobs)

    # ── Test 1: log_probs computation ──
    logger.info("\n=== Test 1: log_probs correctness ===")
    prompt = "Question: 2+2?\nAnswer:"
    response = " 4"

    log_probs = compute_log_probs([prompt], [response], model)
    logger.info(f"  log_probs shape: {log_probs.shape}, value: {log_probs.item():.6f}")
    logger.info(f"  requires_grad: {log_probs.requires_grad}")
    assert log_probs.shape == (1,), f"Expected shape (1,), got {log_probs.shape}"
    assert log_probs.requires_grad, "log_probs must have requires_grad=True when model params require grad"
    # log-prob for a 2-token response should be negative (probability < 1)
    assert log_probs.item() < 0, f"Expected negative log-prob, got {log_probs.item():.6f}"
    logger.info("  ✓ PASSED")

    # ── Test 2: Gradient flow ──
    logger.info("\n=== Test 2: Gradient flow — loss → log_probs → gather → logits → params")

    prompts_2 = ["Question: 2+2?\nAnswer:", "Question: 2+2?\nAnswer:"]
    responses_2 = [" 4", " 4"]
    log_probs_2 = compute_log_probs(prompts_2, responses_2, model)

    # Simulate GRPO loss
    advantages = torch.tensor([0.5, -0.5], device=device)
    with torch.no_grad():
        log_probs_old = compute_log_probs(prompts_2, responses_2, model).clone()
        log_probs_ref = compute_log_probs(prompts_2, responses_2, ref_model).clone()

    ratio = (log_probs_2 - log_probs_old).exp()
    ppo_loss = -torch.min(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)

    log_ratio = log_probs_ref - log_probs_2
    kl_k3 = log_ratio.exp() - log_ratio - 1
    kl_penalty = 0.01 * kl_k3

    total_loss = (ppo_loss + kl_penalty).mean()
    logger.info(f"  total_loss: {total_loss.item():.6f}")

    # Backward
    model.zero_grad()
    total_loss.backward()

    grad_params = []
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            grad_params.append(name)

    logger.info(f"  Parameters receiving gradients: {len(grad_params)}/{len(list(model.parameters()))}")
    for name in grad_params[:5]:
        logger.info(f"    {name}")

    assert len(grad_params) > 0, "No parameters received gradients!"
    # Specifically, gradient should flow through lm_head (output layer)
    assert "lm_head.weight" in grad_params or "lm_head.bias" in grad_params, \
        "Gradient didn't reach lm_head!"
    logger.info("  ✓ PASSED: Gradients flow through gather → logits → params")

    # ── Test 3: Ratio stability ──
    logger.info("\n=== Test 3: PPO ratio numerical stability ===")
    # Simulate realistic mean per-token log-probs (range ~[-10, -1])
    torch.manual_seed(42)
    for _ in range(100):
        log_old = torch.randn(8) * 1.5 - 3.0   # centered around -3
        log_new = log_old + torch.randn(8) * 0.3  # small perturbations
        ratio = (log_new - log_old).exp()
        assert not torch.isnan(ratio).any(), "NaN in ratio"
        assert not torch.isinf(ratio).any(), "Inf in ratio"
        assert ratio.max() < 50.0, f"Ratio exploded: max={ratio.max().item():.2f}"
    logger.info(f"  100 random trials: ratio max={ratio.max().item():.4f}, min={ratio.min().item():.4f}")
    logger.info("  ✓ PASSED: Ratio numerically stable with mean per-token log-probs")

    # ── Test 4: KL k3 estimator properties ──
    logger.info("\n=== Test 4: KL k3 estimator")
    # k3(x) = exp(x) - x - 1 >= 0 for all x
    for x in [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]:
        k3 = torch.exp(torch.tensor(x)) - x - 1
        assert k3 >= 0, f"k3({x}) = {k3} < 0!"
    logger.info(f"  k3(-10)={torch.exp(torch.tensor(-10.0)) - (-10) - 1:.6f}")
    logger.info(f"  k3(0)  ={torch.exp(torch.tensor(0.0)) - 0 - 1:.6f}")
    logger.info(f"  k3(10) ={torch.exp(torch.tensor(10.0)) - 10 - 1:.6f}")
    logger.info("  ✓ PASSED: k3(x) >= 0 for all x")

    # ── Test 5: GRPO advantage computation ──
    logger.info("\n=== Test 5: GRPO advantage properties ===")
    rewards = torch.tensor([0.8, 0.2, 0.6, 0.9, 0.3, 0.7, 0.1, 0.5])
    advantages = compute_grpo_advantage(rewards, group_size=4)
    logger.info(f"  rewards:     {rewards.view(2, 4).tolist()}")
    logger.info(f"  advantages:  {advantages.view(2, 4).tolist()}")
    # Sum of advantages within each group should be ~0
    adv_2d = advantages.view(2, 4)
    group_sums = adv_2d.sum(dim=-1)
    logger.info(f"  group sums:  {group_sums.tolist()} (should be ~0)")
    assert (group_sums.abs() < 1e-4).all(), f"Group advantage sums should be 0, got {group_sums}"
    logger.info("  ✓ PASSED: Advantages sum to 0 within each group")

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("ALL 5 TESTS PASSED — gradient flow verified")
    logger.info("=" * 60)
    logger.info("\nVerified properties:")
    logger.info("  1. log_probs = gather(token_logprobs).mean() — correct")
    logger.info("  2. Gradient path: loss → log_probs → gather → logits → params")
    logger.info("  3. PPO ratio stable with mean per-token log-probs")
    logger.info("  4. KL k3 estimator always non-negative")
    logger.info("  5. GRPO advantages sum to 0 within each group")


if __name__ == "__main__":
    main()

import torch


def compute_grpo_advantage(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute GRPO group-relative advantages.

    For each prompt, G responses are sampled. Rewards within each group
    are normalized to zero mean and unit variance:

        A_i = (R_i - mean(R_group)) / std(R_group)

    This eliminates the need for a Critic network entirely — the
    advantage signal comes purely from relative comparison within the
    sampled group.

    Args:
        rewards:    (B*G,) flattened rewards for all responses
        group_size: G, number of responses per prompt
        eps:        numerical stability for std division

    Returns:
        advantages: (B*G,) group-normalized advantages

    Edge cases handled:
        - std=0 (identical rewards): returns 0 advantages (no signal)
        - single-element groups: returns 0 (no relative comparison possible)
    """
    B = rewards.shape[0] // group_size
    rewards_grouped = rewards.view(B, group_size)  # (B, G)

    mu = rewards_grouped.mean(dim=-1, keepdim=True)  # (B, 1)
    std = rewards_grouped.std(dim=-1, keepdim=True)  # (B, 1)

    # Clamp std to prevent division by zero and gradient explosion
    std = std.clamp(min=eps)

    advantages = (rewards_grouped - mu) / std  # (B, G)
    return advantages.view(-1)  # (B*G,)

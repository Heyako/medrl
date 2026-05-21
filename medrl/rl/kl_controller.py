import torch
import torch.nn.functional as F
from typing import Optional

from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)


class KLController:
    """
    Adaptive KL penalty coefficient controller.

    Uses multiplicative update to keep KL divergence near a target value.
    When KL exceeds target, beta increases to penalize divergence more.
    When KL is below target, beta relaxes to allow more exploration.

    Target KL range: typically 0.01–0.05 for medical QA tasks.
    """

    def __init__(
        self,
        target_kl: float = 0.02,
        beta: float = 0.01,
        lr: float = 0.1,
        beta_min: float = 1e-4,
        beta_max: float = 1.0,
    ):
        self.target_kl = target_kl
        self.beta = beta
        self.lr = lr
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.current_kl: Optional[float] = None

    def step(self, current_kl: float) -> float:
        """
        Update beta based on observed KL divergence.

        Multiplicative update: beta *= (1 + lr * (kl - target_kl))
        """
        self.current_kl = current_kl
        error = current_kl - self.target_kl
        self.beta *= (1.0 + self.lr * error)
        self.beta = max(self.beta_min, min(self.beta_max, self.beta))
        return self.beta

    def compute_kl(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> float:
        """Compute KL(π_θ || π_ref) from logits."""
        kl = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction="batchmean",
        )
        return kl.item()

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute policy entropy H(π_θ) as a diversity indicator."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy.item()

    def should_warn(self) -> bool:
        """Check if KL has diverged severely."""
        return self.current_kl is not None and self.current_kl > 0.1

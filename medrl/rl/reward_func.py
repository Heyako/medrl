"""
Composite Reward Function for MedRL GRPO training.

Design principle: combine hard constraints (regex-based format checks)
with soft constraints (LLM-as-a-Judge scoring) to combat reward
homogeneity and reward hacking in group-relative advantage computation.

Three reward components:
    1. Format Reward (hard): strict regex check for <thinking>...</thinking><answer>...</answer>
       - Violation → severe negative penalty (-1.0)
       - This creates natural variance in group rewards

    2. LLM Judge Reward (soft): continuous 0-1 score on:
       - Diagnostic correctness
       - Logical coherence
       - Medical terminology accuracy

    3. Diversity Bonus (auxiliary): n-gram diversity within the reasoning chain
       - Suppresses repetitive/degenerate "reward hacking" loops
"""

import re
import math
from typing import List, Dict, Optional, Tuple

import torch

from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)

# ── Format patterns ──
THINKING_PATTERN = re.compile(
    r"<thinking>.*?</thinking>",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(
    r"<answer>.*?</answer>",
    re.DOTALL | re.IGNORECASE,
)
# Detect meaningless repeated phrases (reward hacking signal)
REPETITION_PATTERN = re.compile(
    r"(.{20,}?)\1{2,}"  # any 20+ char segment repeated 3+ times
)

# ── Format violation penalty ──
FORMAT_VIOLATION_PENALTY = -1.0
FORMAT_COMPLIANCE_REWARD = 0.1  # small positive signal for proper format


class CompositeReward:
    """
    Composite reward function for medical CoT alignment.

    Weights control the relative importance of each reward component.
    Default weights are set to prioritize:
        - Clinical correctness (highest weight)
        - Format compliance (hard constraint)
        - Diversity (moderate weight to prevent hacking)
    """

    def __init__(
        self,
        w_format: float = 0.15,
        w_judge: float = 0.70,
        w_diversity: float = 0.15,
        api_key: Optional[str] = None,
        judge_model: str = "deepseek-chat",
        judge_base_url: str = "https://api.deepseek.com/v1",
        use_judge: bool = True,
    ):
        self.w_format = w_format
        self.w_judge = w_judge
        self.w_diversity = w_diversity
        self.api_key = api_key
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url
        self.use_judge = use_judge

    # ── Component 1: Format Reward (hard constraint) ──

    def format_reward(self, response: str) -> float:
        """
        Check for strict tag compliance.

        Returns:
            +0.1 if both tags present and well-formed
            -1.0 if either tag is missing or malformed
        """
        has_thinking = bool(THINKING_PATTERN.search(response))
        has_answer = bool(ANSWER_PATTERN.search(response))

        if has_thinking and has_answer:
            return FORMAT_COMPLIANCE_REWARD

        # Partial compliance: log which tag is missing
        if not has_thinking and not has_answer:
            logger.debug("Format violation: missing both <thinking> and <answer> tags")
        elif not has_thinking:
            logger.debug("Format violation: missing <thinking> tag")
        else:
            logger.debug("Format violation: missing <answer> tag")

        return FORMAT_VIOLATION_PENALTY

    # ── Component 2: LLM-as-a-Judge (soft constraint) ──

    JUDGE_PROMPT = """You are an expert medical reviewer. Evaluate the following medical reasoning output on three dimensions. Score each dimension from 0.0 to 1.0.

Medical Question: {question}

Model Response:
{response}

Evaluate:
1. Diagnostic Correctness (0-1): Is the final diagnosis/answer medically accurate?
2. Logical Coherence (0-1): Does the reasoning chain follow logically without gaps or leaps?
3. Terminology Accuracy (0-1): Is medical terminology used correctly and precisely?

Output ONLY a JSON object:
{{"correctness": 0.0, "coherence": 0.0, "terminology": 0.0, "overall": 0.0}}"""

    def judge_reward(self, question: str, response: str) -> float:
        """
        Use LLM (DeepSeek/OpenAI/any OpenAI-compatible) to score the
        response continuously on 0-1 scale.

        Supports any provider that exposes an OpenAI-compatible chat
        completions endpoint (DeepSeek, Qwen-Max, GLM-4, Kimi, etc.).

        Returns the overall score as a single float.
        """
        if not self.use_judge or not self.api_key:
            return self._heuristic_score(response)

        import openai

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.judge_base_url,
        )
        prompt = self.JUDGE_PROMPT.format(
            question=question, response=response[:3000]
        )  # truncate for API cost control

        try:
            resp = client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            import json
            result = json.loads(resp.choices[0].message.content)
            score = float(result.get("overall", 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return self._heuristic_score(response)

    # ── Component 3: Diversity Bonus ──

    def diversity_reward(self, response: str, n: int = 3) -> float:
        """
        Compute n-gram diversity score to penalize repetitive outputs.

        High repetition → low diversity score → lower reward.
        This directly combats the "looping" form of reward hacking.

        Returns a value in [0, 1] where 1 = maximally diverse.
        """
        # Penalize explicit repetition patterns
        if REPETITION_PATTERN.search(response):
            return 0.0

        tokens = response.split()
        if len(tokens) < n:
            return 1.0  # too short to assess, give benefit of doubt

        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i : i + n]))

        # Type-Token Ratio for n-grams
        max_possible = len(tokens) - n + 1
        diversity = len(ngrams) / max(max_possible, 1)

        # Compress: very low diversity gets heavily penalized
        if diversity < 0.3:
            return 0.0
        elif diversity < 0.5:
            return 0.3
        else:
            return diversity

    # ── Component 4: Heuristic quality score (no-LLM-judge fallback) ──

    # Medical reasoning structure indicators
    REASONING_MARKERS = [
        r"Step\s*\d+",
        r"(?i)(therefore|thus|hence|consequently|as a result)",
        r"(?i)(diagnosis|diagnose|diagnosed)",
        r"(?i)(indicated|contraindicated|recommended)",
        r"(?i)(eliminate|rule out|exclude|differential)",
        r"(?i)(guideline|ACC|AHA|GOLD|KDIGO)",
    ]

    def _heuristic_score(self, response: str) -> float:
        """
        Score response quality using cheap heuristics.
        Used when LLM judge is unavailable (Phase 1).

        Returns a score in [0, 1] that varies meaningfully across responses.
        """
        score = 0.0

        # 1. Step structure: 3-8 steps is ideal
        step_count = len(re.findall(r"Step\s*\d+", response, re.IGNORECASE))
        if step_count >= 5:
            score += 0.3
        elif step_count >= 3:
            score += 0.2
        elif step_count >= 1:
            score += 0.1

        # 2. Reasoning depth: check for diverse reasoning markers
        marker_hits = sum(
            1 for pattern in self.REASONING_MARKERS
            if re.search(pattern, response)
        )
        score += min(0.3, marker_hits * 0.06)

        # 3. Length quality: penalize too short (<200 chars) or excessive (>3000)
        length = len(response)
        if 500 <= length <= 2500:
            score += 0.2
        elif 200 <= length <= 500:
            score += 0.1

        # 4. Option elimination quality: check for option letters mentioned
        option_hits = len(re.findall(r"\bOption\s*[A-E]\b", response, re.IGNORECASE))
        option_hits += len(re.findall(r"\b[A-E][\.\)]\s", response))
        score += min(0.2, option_hits * 0.05)

        return max(0.0, min(1.0, score))

    # ── Composite ──

    def __call__(
        self,
        prompts: List[str],
        responses: List[str],
        questions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute composite reward for a batch of (prompt, response) pairs.

        Args:
            prompts:   list of prompt strings
            responses: list of model-generated response strings
            questions: optional list of cleaned question strings for judge

        Returns:
            rewards: (B,) tensor of scalar rewards in approximately [-1.0, 1.0] range
        """
        if questions is None:
            questions = prompts  # fallback: use full prompt as question context

        rewards = []
        for prompt, response, question in zip(prompts, responses, questions):
            r_format = self.format_reward(response)
            r_diversity = self.diversity_reward(response)

            # Only call judge if format is compliant (save API cost)
            if r_format > 0 and self.use_judge:
                r_judge = self.judge_reward(question, response)
            elif r_format < 0:
                r_judge = 0.0
            else:
                # Heuristic fallback — varies per response, not flat 0.5
                r_judge = self._heuristic_score(response)

            # Weighted combination
            total = (
                self.w_format * r_format
                + self.w_judge * r_judge
                + self.w_diversity * r_diversity
            )

            rewards.append(total)

        return torch.tensor(rewards, dtype=torch.float32)

    def compute_group_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        group_size: int,
        questions: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards organized by group, returning both the reward tensor
        and diagnostic statistics.

        Returns:
            rewards:        (B*G,) reward tensor
            stats:          {"mean_reward", "std_reward", "format_violation_rate", ...}
        """
        rewards = self(
            prompts=prompts, responses=responses, questions=questions
        )

        B = len(prompts) // group_size
        rewards_grouped = rewards.view(B, group_size)

        stats = {
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "group_mean_std": rewards_grouped.std(dim=-1).mean().item(),
            # Ratio of responses that violated format (indicator of hacking/instability)
            "format_violation_rate": (rewards < -0.5).float().mean().item(),
        }

        # Warning: reward homogeneity detected
        if stats["group_mean_std"] < 0.05:
            logger.warning(
                f"CRITICAL: Group reward std = {stats['group_mean_std']:.4f} — "
                f"advantages are near-zero. Check reward function discriminability."
            )

        return rewards, stats

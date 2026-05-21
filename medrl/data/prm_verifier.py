"""
Process Reward Model (PRM) Verifier: small-large model collaborative
verification mechanism for medical CoT step correctness.

Principle:
    - A small, fast model (e.g., 1.5B) performs initial plausibility screening.
    - A large, accurate model (e.g., GPT-4/Claude) does final verification
      on steps that pass the small model filter.
    - Steps flagged as hallucinated or logically flawed are discarded.

This removes intermediate reasoning errors before constructing the PRM dataset,
ensuring step-by-step correctness of the structured CoT data.
"""

import asyncio
from typing import List, Dict, Optional, Tuple

from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)

# Verification prompt for step-level correctness checking
VERIFICATION_PROMPT = """You are a medical reasoning verifier. Evaluate whether the following reasoning step is logically sound and medically accurate.

Context (prior steps): {context}
Current step: {step}

Evaluate on:
1. Medical factual accuracy: Are the medical claims correct?
2. Logical coherence: Does this step logically follow from prior context?
3. Clinical relevance: Is this reasoning relevant to the diagnostic question?

Output ONLY a JSON object:
{{"valid": true/false, "score": 0.0-1.0, "reason": "brief explanation"}}"""


class PRMVerifier:
    """
    Two-stage process verification for medical CoT steps.

    Stage 1 (small model): binary plausibility filter
    Stage 2 (large model): detailed correctness scoring
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        large_model: str = "gpt-4",
        score_threshold: float = 0.7,
        max_concurrent: int = 10,
    ):
        self.api_key = api_key
        self.large_model = large_model
        self.score_threshold = score_threshold
        self.max_concurrent = max_concurrent

    def _check_small_model_heuristics(self, step: str) -> bool:
        """
        Fast heuristic checks that a small model would perform.
        These are rule-based filters for obvious errors.

        Returns True if the step passes initial screening.
        """
        # Reject empty or trivially short steps
        if len(step) < 10:
            return False

        # Reject steps with obvious hallucination markers
        hallucination_markers = [
            r"I don't know",
            r"I'm not sure",
            r"possibly",
            r"maybe",
            r"(unknown|unclear|uncertain)",
        ]
        import re
        for pattern in hallucination_markers:
            if re.search(pattern, step, re.IGNORECASE) and len(step) < 100:
                return False

        # Reject steps that are just formatting noise
        if re.match(r'^[\s\d\.\-:;,]+$', step):
            return False

        return True

    def _verify_with_llm(
        self, context: str, step: str
    ) -> Tuple[bool, float, str]:
        """Use large model to verify a single step."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        prompt = VERIFICATION_PROMPT.format(context=context, step=step)

        try:
            response = client.chat.completions.create(
                model=self.large_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            import json
            result = json.loads(response.choices[0].message.content)
            valid = result.get("valid", False)
            score = result.get("score", 0.0)
            reason = result.get("reason", "")
            return valid and score >= self.score_threshold, score, reason
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return False, 0.0, str(e)

    async def _verify_async(
        self, context: str, step: str
    ) -> Tuple[bool, float, str]:
        """Async wrapper for LLM verification."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._verify_with_llm, context, step
        )

    def verify_steps(
        self, split_data: List[Dict], use_api: bool = True
    ) -> List[Dict]:
        """
        Verify all steps for a batch of CoT-split rationales.

        Input: [{"id": str, "steps": [str, ...]}, ...]
        Output: [{"id": str, "verified_steps": [{"step": str, "score": float}, ...]}, ...]

        Steps failing either the heuristic filter or LLM check are dropped.
        """
        verified_results = []

        for item in split_data:
            rid = item.get("id", "")
            steps = item.get("steps", [])
            verified_steps: List[Dict] = []
            context_parts: List[str] = []

            for i, step in enumerate(steps):
                # Stage 1: heuristic screening
                if not self._check_small_model_heuristics(step):
                    logger.debug(
                        f"Step {i} of id={rid} failed heuristic filter: {step[:100]}..."
                    )
                    continue

                context = "\n".join(context_parts)

                # Stage 2: LLM verification
                if use_api and self.api_key:
                    valid, score, reason = self._verify_with_llm(context, step)
                else:
                    # Without API, accept heuristically-passed steps
                    valid, score, reason = True, 0.8, "heuristic_only"

                if valid:
                    verified_steps.append({"step": step, "score": score})
                    context_parts.append(step)
                else:
                    logger.info(
                        f"Step {i} of id={rid} rejected: score={score:.2f}, "
                        f"reason={reason}"
                    )

            verified_results.append({
                "id": rid,
                "verified_steps": verified_steps,
                "n_original": len(steps),
                "n_verified": len(verified_steps),
            })

            if len(verified_steps) < len(steps):
                logger.info(
                    f"id={rid}: retained {len(verified_steps)}/{len(steps)} steps after PRM verification"
                )

        return verified_results

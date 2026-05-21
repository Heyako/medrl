"""
Answer extractor: extracts final answers from long CoT reasoning outputs.

For MedRL, the model outputs follow the format:
    <thinking> reasoning process </thinking>
    <answer> final answer </answer>

This module extracts the content inside <answer>...</answer> tags
and normalizes it for exact-match (EM) evaluation.
"""

import re
from typing import Optional, List

from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)

ANSWER_TAG_PATTERN = re.compile(
    r"<answer>\s*(.+?)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)
THINKING_TAG_PATTERN = re.compile(
    r"<thinking>\s*(.+?)\s*</thinking>",
    re.DOTALL | re.IGNORECASE,
)


class AnswerExtractor:
    """
    Extract and normalize final answers from structured CoT outputs.
    """

    @staticmethod
    def extract_answer_tag(text: str) -> Optional[str]:
        """Extract content from <answer>...</answer> tags."""
        match = ANSWER_TAG_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_thinking_tag(text: str) -> Optional[str]:
        """Extract content from <thinking>...</thinking> tags."""
        match = THINKING_TAG_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_last_line(text: str) -> Optional[str]:
        """
        Fallback: extract the last non-empty line as the answer.
        Used when no <answer> tag is found.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[-1] if lines else None

    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize answer text for comparison.

        Steps:
        1. Strip whitespace
        2. Lowercase
        3. Remove punctuation at sentence boundaries
        4. Collapse multiple spaces
        """
        text = text.strip().lower()
        text = re.sub(r"[.,;:!?]+$", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def extract(self, model_output: str) -> Optional[str]:
        """
        Extract final answer from model output.

        Priority:
        1. <answer> tag content
        2. Last non-empty line (fallback)
        """
        answer = self.extract_answer_tag(model_output)
        if answer:
            return self.normalize_answer(answer)

        logger.warning("No <answer> tag found, using last-line fallback")
        last_line = self.extract_last_line(model_output)
        if last_line:
            return self.normalize_answer(last_line)

        return None

    def batch_extract(self, outputs: List[str]) -> List[Optional[str]]:
        """Extract answers from a batch of model outputs."""
        return [self.extract(o) for o in outputs]

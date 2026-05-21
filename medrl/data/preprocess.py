import json
import re
from pathlib import Path
from typing import List, Dict, Optional

from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """
    Clean and normalize raw medical QA trajectories.

    Handles:
        - Deduplication of near-identical QA pairs
        - Whitespace / Unicode normalization
        - Optional length filtering to remove degenerate short responses
        - Structured output to JSONL format
    """

    def __init__(
        self,
        min_response_len: int = 50,
        max_response_len: int = 4096,
    ):
        self.min_response_len = min_response_len
        self.max_response_len = max_response_len

    def normalize(self, text: str) -> str:
        """Normalize unicode and whitespace."""
        import unicodedata

        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def filter_by_length(self, response: str) -> bool:
        """Return True if response passes length constraints."""
        length = len(response)
        return self.min_response_len <= length <= self.max_response_len

    def process(
        self, raw_data: List[Dict], source: str = "unknown"
    ) -> List[Dict]:
        """
        Process a list of raw medical QA dicts.

        Expected input format:
            {"question": str, "answer": str, "rationale": Optional[str]}

        Output format:
            {"question": str, "answer": str, "rationale": str, "source": str}
        """
        cleaned: List[Dict] = []
        seen_questions = set()

        for item in raw_data:
            question = self.normalize(item.get("question", ""))
            answer = self.normalize(item.get("answer", ""))
            rationale = self.normalize(item.get("rationale", ""))

            if not question or not answer:
                logger.warning(f"Skipping empty Q/A in source={source}")
                continue

            if not self.filter_by_length(rationale):
                logger.debug(
                    f"Skipping rationale length={len(rationale)} for Q: {question[:80]}..."
                )
                continue

            # Deduplicate by question hash
            q_hash = hash(question)
            if q_hash in seen_questions:
                continue
            seen_questions.add(q_hash)

            cleaned.append({
                "question": question,
                "answer": answer,
                "rationale": rationale,
                "source": source,
            })

        logger.info(f"Preprocessed {len(cleaned)}/{len(raw_data)} valid records from {source}")
        return cleaned

    def save_jsonl(self, data: List[Dict], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(data)} records to {path}")

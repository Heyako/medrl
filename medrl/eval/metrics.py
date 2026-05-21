"""
Evaluation metrics for medical QA benchmarking.

Core metrics:
    - Exact Match (EM) Accuracy
    - F1 Score (token-level overlap)
    - Accuracy @ K (top-K correctness)
"""

import re
from typing import List, Dict, Optional
from collections import Counter

from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)


def normalize_answer(text: str) -> str:
    """Normalize text for fair comparison — same as AnswerExtractor."""
    text = text.strip().lower()
    text = re.sub(r"[.,;:!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_em_accuracy(
    predictions: List[Optional[str]],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute Exact Match accuracy.

    Args:
        predictions: extracted answer strings (may contain None for failures)
        references: ground-truth answer strings

    Returns:
        {"em_accuracy": float, "n_correct": int, "n_total": int, "n_failed_extraction": int}
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    n_correct = 0
    n_failed = 0

    for pred, ref in zip(predictions, references):
        if pred is None:
            n_failed += 1
            continue
        if normalize_answer(pred) == normalize_answer(ref):
            n_correct += 1

    n_total = len(references)
    return {
        "em_accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_failed_extraction": n_failed,
    }


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / max(len(pred_tokens), 1)
    recall = num_common / max(len(ref_tokens), 1)
    return 2 * precision * recall / (precision + recall + 1e-13)


def compute_f1_scores(
    predictions: List[Optional[str]],
    references: List[str],
) -> Dict[str, float]:
    """Compute average F1 score across all examples."""
    scores = []
    n_skipped = 0
    for pred, ref in zip(predictions, references):
        if pred is None:
            n_skipped += 1
            continue
        scores.append(compute_f1(pred, ref))

    avg_f1 = sum(scores) / max(len(scores), 1)
    return {
        "avg_f1": avg_f1,
        "n_evaluated": len(scores),
        "n_skipped": n_skipped,
    }


def compute_metrics(
    predictions: List[Optional[str]],
    references: List[str],
) -> Dict[str, float]:
    """Compute all metrics at once."""
    em = compute_em_accuracy(predictions, references)
    f1 = compute_f1_scores(predictions, references)
    return {**em, **f1}

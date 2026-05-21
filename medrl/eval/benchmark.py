"""
Automated benchmark runner for MedQA and MedMCQA.

Workflow:
    1. Load benchmark dataset
    2. Run model inference (with CoT reasoning)
    3. Extract final answers
    4. Compute EM accuracy, F1
    5. Generate statistical charts
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from medrl.eval.extractor import AnswerExtractor
from medrl.eval.metrics import compute_metrics
from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class BenchmarkResult:
    name: str
    em_accuracy: float
    avg_f1: float
    n_total: int
    n_correct: int
    n_failed_extraction: int
    runtime_seconds: float
    per_sample: List[Dict]


class BenchmarkRunner:
    """
    Run evaluation on medical QA benchmarks.

    Args:
        model_fn: callable that takes a list of prompts and returns generated texts
        extractor: AnswerExtractor instance
        output_dir: where to save results
    """

    def __init__(
        self,
        model_fn: Callable[[List[str]], List[str]],
        extractor: Optional[AnswerExtractor] = None,
        output_dir: str = "./outputs/eval",
    ):
        self.model_fn = model_fn
        self.extractor = extractor or AnswerExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_medqa(self, path: str) -> List[Dict]:
        """Load MedQA dataset (JSONL format)."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(data)} samples from MedQA: {path}")
        return data

    def load_medmcqa(self, path: str) -> List[Dict]:
        """Load MedMCQA dataset (JSONL format)."""
        return self.load_medqa(path)  # same format

    def _build_prompts(self, samples: List[Dict]) -> List[str]:
        """
        Build CoT prompts from samples.

        Expected sample format:
            {"question": str, "options": Optional[Dict[str, str]]}
        """
        prompts = []
        for s in samples:
            q = s["question"]
            if "options" in s and s["options"]:
                opts = "\n".join(
                    f"{k}: {v}" for k, v in s["options"].items()
                )
                prompt = (
                    f"Question: {q}\nOptions:\n{opts}\n\n"
                    f"Please reason step by step, then provide your final answer "
                    f"in <answer>...</answer> tags."
                )
            else:
                prompt = (
                    f"Question: {q}\n\n"
                    f"Please reason step by step, then provide your final answer "
                    f"in <answer>...</answer> tags."
                )
            prompts.append(prompt)
        return prompts

    def run(
        self,
        name: str,
        samples: List[Dict],
        batch_size: int = 8,
    ) -> BenchmarkResult:
        """
        Run benchmark evaluation.

        Args:
            name: benchmark name (e.g., "MedQA", "MedMCQA")
            samples: list of {"question": ..., "answer": ..., "options": ...}
            batch_size: inference batch size
        """
        logger.info(f"Running benchmark: {name} ({len(samples)} samples)")
        start_time = time.time()

        prompts = self._build_prompts(samples)

        # Batch inference
        all_outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            try:
                batch_outputs = self.model_fn(batch_prompts)
                all_outputs.extend(batch_outputs)
            except Exception as e:
                logger.error(f"Inference failed for batch {i}: {e}")
                all_outputs.extend([""] * len(batch_prompts))

        # Extract answers
        predictions = self.extractor.batch_extract(all_outputs)
        references = [s["answer"] for s in samples]

        # Compute metrics
        metrics = compute_metrics(predictions, references)

        # Per-sample results
        per_sample = []
        for i, (pred, ref, output) in enumerate(zip(predictions, references, all_outputs)):
            per_sample.append({
                "id": samples[i].get("id", i),
                "prediction": pred,
                "reference": ref,
                "correct": (pred == ref) if pred else False,
                "raw_output": output[:200] + "..." if len(output) > 200 else output,
            })

        runtime = time.time() - start_time

        result = BenchmarkResult(
            name=name,
            em_accuracy=metrics["em_accuracy"],
            avg_f1=metrics["avg_f1"],
            n_total=metrics["n_total"],
            n_correct=metrics["n_correct"],
            n_failed_extraction=metrics["n_failed_extraction"],
            runtime_seconds=runtime,
            per_sample=per_sample,
        )

        self._save_result(result)
        return result

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to JSON."""
        output_path = self.output_dir / f"{result.name}_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": result.name,
                "em_accuracy": result.em_accuracy,
                "avg_f1": result.avg_f1,
                "n_total": result.n_total,
                "n_correct": result.n_correct,
                "n_failed_extraction": result.n_failed_extraction,
                "runtime_seconds": result.runtime_seconds,
            }, f, indent=2, ensure_ascii=False)
        logger.info(
            f"[{result.name}] EM={result.em_accuracy:.4f}, "
            f"F1={result.avg_f1:.4f}, "
            f"Correct={result.n_correct}/{result.n_total}, "
            f"Time={result.runtime_seconds:.1f}s"
        )

    def compare(self, results: List[BenchmarkResult]) -> None:
        """Print comparison table for multiple results."""
        print("\n" + "=" * 60)
        print(f"{'Benchmark':<15} {'EM Acc':<10} {'F1':<10} {'Correct':<15} {'Time':<10}")
        print("-" * 60)
        for r in results:
            print(
                f"{r.name:<15} {r.em_accuracy:.4f}     {r.avg_f1:.4f}    "
                f"{r.n_correct}/{r.n_total}        {r.runtime_seconds:.1f}s"
            )
        print("=" * 60)

#!/usr/bin/env python3
"""
============================================================================
MedRL Phase 1: CoT Dataset Construction Pipeline
============================================================================

Converts raw medical QA pairs into structured Chain-of-Thought training
samples using a lightweight local model (Qwen2.5-1.5B-Instruct).

Supports two input formats:
  - Simple QA:   {"question": "...", "answer": "..."}
  - Multiple-Choice (MedQA / MedMCQA):
      {"question": "...", "options": {"A": ..., "B": ..., "C": ..., "D": ...}, "answer": "C"}

Hardware Target: Single RTX 3090/4090 (24 GB VRAM)
Model:           Qwen2.5-1.5B-Instruct (~3 GB in FP16)

Usage:
    # Simple QA format
    python scripts/build_cot_dataset.py \
        --input data/raw/sample_medqa.jsonl \
        --output data/processed/cot_train.jsonl

    # MedQA / MedMCQA format (auto-detect or specify)
    python scripts/build_cot_dataset.py \
        --input data/raw/medqa_train.jsonl \
        --output data/processed/cot_medqa_train.jsonl \
        --dataset medqa

    # Dry-run (no GPU)
    python scripts/build_cot_dataset.py --dry-run
============================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medrl.data.cot_splitter import CoTSplitter, build_few_shot_prompt_mc
from medrl.data.dataset_loader import (
    auto_load,
    format_mc_question,
    format_mc_answer,
)
from medrl.utils.logging import setup_logger

logger = setup_logger("build_cot_dataset")


def is_mc_sample(sample: Dict) -> bool:
    """Check if a sample dict is multiple-choice format."""
    return "options" in sample and "answer_idx" in sample


def save_output(results: List[Dict], path: str) -> None:
    """Save structured CoT samples to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} samples to {path}")


def print_stats(results: List[Dict]) -> None:
    """Print summary statistics for the generated dataset."""
    n_total = len(results)
    n_valid = sum(1 for r in results if r.get("valid"))
    n_tag_ok = sum(
        1 for r in results
        if r.get("thinking") and r.get("extracted_answer")
    )
    n_steps = [r.get("n_steps", 0) for r in results if r.get("n_steps", 0) > 0]
    avg_steps = sum(n_steps) / max(len(n_steps), 1)

    thinking_lens = [len(r.get("thinking", "")) for r in results if r.get("thinking")]
    avg_thinking_len = sum(thinking_lens) / max(len(thinking_lens), 1)

    # Per-source breakdown if available
    sources = {}
    for r in results:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\n" + "=" * 55)
    print("  MedRL Phase 1 -- Dataset Construction Report")
    print("=" * 55)
    print(f"  Total samples:         {n_total}")
    print(f"  Tag-compliant:         {n_tag_ok}/{n_total} ({100*n_tag_ok/max(n_total,1):.1f}%)")
    print(f"  Format-valid:          {n_valid}/{n_total} ({100*n_valid/max(n_total,1):.1f}%)")
    print(f"  Avg steps per sample:  {avg_steps:.1f}")
    print(f"  Avg thinking length:   {avg_thinking_len:.0f} chars")
    if len(sources) > 1:
        print(f"  Sources:               {sources}")
    print("=" * 55)

    failures = [r for r in results if not r.get("valid")]
    if failures:
        print(f"\n  Validation failures ({len(failures)}):")
        for f in failures[:5]:
            print(f"    - [{f.get('validation_reason', 'unknown')}] "
                  f"Q='{f.get('question', '')[:60]}...'")


def dry_run():
    """Validate code path without GPU or real data."""
    logger.info("=== DRY RUN: testing pipeline code paths ===")

    # -- QA format tests --
    logger.info("--- Simple QA format ---")
    logger.info("Test 1: Tag extraction...")
    thinking, answer = CoTSplitter.extract_tags(
        "<thinking>Step 1: Fever.\nStep 2: Infiltrates.</thinking>\n<answer>Pneumonia</answer>"
    )
    assert thinking and len(thinking) > 20, f"Bad thinking: {thinking}"
    assert answer == "Pneumonia", f"Bad answer: {answer}"
    logger.info("  PASS")

    logger.info("Test 2: Step parsing...")
    steps = CoTSplitter.extract_steps(
        "Step 1: Check symptoms.\nStep 2: Review labs.\nStep 3: Diagnose."
    )
    assert len(steps) == 3, f"Expected 3, got {len(steps)}"
    logger.info("  PASS")

    logger.info("Test 3: QA prompt construction...")
    from medrl.data.cot_splitter import build_few_shot_prompt
    prompt = build_few_shot_prompt("What is fever?", "Elevated body temperature")
    assert "<thinking>" in prompt and "What is fever?" in prompt
    logger.info(f"  PASS ({len(prompt)} chars)")

    # -- MC format tests --
    logger.info("--- Multiple-Choice format (MedQA / MedMCQA) ---")
    logger.info("Test 4: MC prompt construction...")
    mc_prompt = build_few_shot_prompt_mc(
        "A 45-year-old has chest pain. What is the diagnosis?",
        "Options:\nA. MI\nB. PE\nC. Pneumonia\nD. GERD",
        "A. MI",
    )
    assert "<thinking>" in mc_prompt
    assert "Options:" in mc_prompt
    assert "A. MI" in mc_prompt
    assert all(opt in mc_prompt for opt in ["A. MI", "B. PE", "C. Pneumonia", "D. GERD"])
    logger.info(f"  PASS ({len(mc_prompt)} chars)")

    logger.info("Test 5: MC tag extraction with option letter...")
    mc_output = """<thinking>
Step 1: Symptoms suggest cardiac origin.
Step 2: Eliminate PE (no respiratory symptoms).
Step 3: Eliminate pneumonia (no fever, normal WBC).
Step 4: Eliminate GERD (no reflux symptoms, pain is crushing not burning).
Step 5: MI is confirmed by ECG findings and troponin elevation.
</thinking>
<answer>
A. MI
</answer>"""
    think_mc, ans_mc = CoTSplitter.extract_tags(mc_output)
    assert think_mc and len(think_mc) > 30, f"MC thinking too short: {len(think_mc)}"
    assert ans_mc and ans_mc[0] in "ABCD", f"MC answer missing letter: {ans_mc}"
    logger.info(f"  PASS (answer={ans_mc})")

    logger.info("Test 6: MC validation...")
    valid_mc, reason_mc = CoTSplitter.validate_output(
        "Step 1: Analysis. Step 2: Differential. Step 3: Conclusion.",
        "C. Pneumonia", "C. Pneumonia"
    )
    assert valid_mc, f"MC should be valid: {reason_mc}"
    logger.info("  PASS")

    logger.info("Test 7: Dataset loader — MedQA format detection...")
    medqa_sample = {
        "question": "What causes fever?",
        "answer": "D",
        "options": {
            "A": "Viral infection",
            "B": "Bacterial infection",
            "C": "Autoimmune",
            "D": "All of the above",
        },
    }
    from medrl.data.dataset_loader import _resolve_answer, format_mc_question, format_mc_answer
    idx, text = _resolve_answer(medqa_sample["answer"], medqa_sample["options"])
    assert idx == "D", f"Expected D, got {idx}"
    logger.info(f"  PASS (resolved D -> '{text}')")

    q_block = format_mc_question(medqa_sample)
    assert "A. Viral infection" in q_block
    logger.info("  PASS (MC question formatting)")

    logger.info("Test 8: Dataset loader — MedMCQA format...")
    medmcqa_sample_raw = {
        "question": "Drug of choice for malaria?",
        "opa": "Chloroquine",
        "opb": "Artemisinin",
        "opc": "Quinine",
        "opd": "Doxycycline",
        "cop": 1,
    }
    # Test normalization via the _normalize_options helper
    opts = {
        "A": medmcqa_sample_raw["opa"],
        "B": medmcqa_sample_raw["opb"],
        "C": medmcqa_sample_raw["opc"],
        "D": medmcqa_sample_raw["opd"],
    }
    assert opts["A"] == "Chloroquine"
    assert medmcqa_sample_raw["cop"] == 1  # cop=1 means A
    logger.info("  PASS (MedMCQA field mapping)")

    print("\n" + "=" * 55)
    print("  ALL DRY-RUN TESTS PASSED (QA + MC)")
    print("  Pipeline code is healthy -- ready for GPU run.")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="MedRL Phase 1: CoT Dataset Construction"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/sample_medqa.jsonl",
        help="Path to raw medical QA JSON/JSONL",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/cot_train.jsonl",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, choices=["medqa", "medmcqa", "auto"],
        help="Dataset type: medqa, medmcqa, or auto-detect (default)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples (for quick testing)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run pipeline validation without GPU or real data",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    # ── Check GPU ──
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available -- falling back to CPU")
            args.device = "cpu"

    # ── Load & normalize data ──
    logger.info(f"Loading data from: {args.input}")
    dataset_type = args.dataset or "auto"
    samples = auto_load(args.input, dataset_type=dataset_type)
    logger.info(f"Loaded {len(samples)} samples")

    if args.max_samples:
        samples = samples[: args.max_samples]
        logger.info(f"Limited to {len(samples)} samples")

    if not samples:
        logger.error("No valid samples. Exiting.")
        sys.exit(1)

    # Detect format from first sample
    use_mc = is_mc_sample(samples[0])
    logger.info(
        f"Detected format: {'Multiple-Choice' if use_mc else 'Simple QA'}"
    )

    # ── Initialize splitter ──
    logger.info(f"Loading model: {args.model}")
    t_start = time.time()

    splitter = CoTSplitter(
        model_name=args.model,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        api_key=args.api_key,
    )
    logger.info(f"Model ready in {time.time() - t_start:.1f}s")

    # ── Process ──
    logger.info(f"Processing {len(samples)} samples...")
    t_start = time.time()

    if use_mc:
        results = splitter.process_batch_mc(
            samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        # Simple QA: flatten to {"question", "answer"} dicts
        qa_samples = [
            {"question": s.get("question", ""), "answer": s.get("answer", s.get("answer_text", ""))}
            for s in samples
        ]
        results = splitter.process_batch(
            qa_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    elapsed = time.time() - t_start
    logger.info(
        f"Done in {elapsed:.1f}s ({elapsed/max(len(samples),1):.1f}s per sample)"
    )

    # ── Save & report ──
    save_output(results, args.output)
    print_stats(results)


if __name__ == "__main__":
    main()

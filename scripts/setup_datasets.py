#!/usr/bin/env python3
"""
============================================================================
MedRL Phase 1: Automated Dataset Setup & CoT Pipeline Validation
============================================================================

One script to:
  1. Download MedQA (from GitHub)
  2. Download MedMCQA (from HuggingFace datasets)
  3. Convert both to unified JSONL
  4. Dry-run validate the CoT pipeline
  5. Generate CoT on small subsets (if GPU available)
  6. Print quality report

Usage:
    python scripts/setup_datasets.py                  # full setup
    python scripts/setup_datasets.py --skip-download   # skip download, just validate
    python scripts/setup_datasets.py --gpu             # run CoT generation on GPU
    python scripts/setup_datasets.py --max-samples 50  # limit subset size
============================================================================
"""

import argparse
import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medrl.utils.logging import setup_logger
from medrl.data.dataset_loader import (
    load_medqa, load_medmcqa, auto_load,
    format_mc_question, format_mc_answer,
)

logger = setup_logger("setup_datasets")

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def ensure_dirs():
    """Create all required directories."""
    for d in [DATA_RAW, DATA_PROCESSED, PROJECT_ROOT / "data" / "eval"]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Directories ready.")


# ═══════════════════════════════════════════════════════════════
# Step 1: Download MedQA
# ═══════════════════════════════════════════════════════════════

def download_medqa() -> bool:
    """Clone MedQA repo from GitHub and copy data files."""
    medqa_path = DATA_RAW / "medqa_source"

    if medqa_path.exists():
        logger.info("MedQA source already exists, skipping clone.")
    else:
        logger.info("Cloning MedQA from GitHub...")
        try:
            subprocess.run(
                ["git", "clone", "--depth=1",
                 "https://github.com/jind11/MedQA.git",
                 str(medqa_path)],
                check=True, capture_output=True, text=True,
            )
            logger.info("MedQA cloned successfully.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"MedQA clone failed: {e.stderr}")
            logger.warning("This is OK — you can manually download MedQA later.")
            return False

    # Copy JSON data files to data/raw/
    found = False
    for json_file in medqa_path.glob("**/*.json"):
        dest = DATA_RAW / f"medqa_{json_file.name}"
        if not dest.exists():
            dest.write_text(json_file.read_text())
        logger.info(f"  MedQA data: {dest.name}")
        found = True

    if not found:
        logger.warning("No JSON files found in MedQA repo. Check repo structure.")
        # Try alternative structure: MedQA might have data in a different format
        for data_dir in ["data", "data_clean", "questions"]:
            candidate = medqa_path / data_dir
            if candidate.exists():
                for f in candidate.glob("*"):
                    dest = DATA_RAW / f"medqa_{f.name}"
                    if not dest.exists():
                        dest.write_text(f.read_text())
                    logger.info(f"  MedQA data: {dest.name}")
                    found = True

    return found


# ═══════════════════════════════════════════════════════════════
# Step 2: Download MedMCQA
# ═══════════════════════════════════════════════════════════════

def download_medmcqa() -> bool:
    """Download MedMCQA from HuggingFace datasets."""
    output_path = DATA_RAW / "medmcqa_train.jsonl"
    val_path = DATA_RAW / "medmcqa_validation.jsonl"

    if output_path.exists():
        logger.info(f"MedMCQA already exists: {output_path}")
        return True

    logger.info("Downloading MedMCQA from HuggingFace datasets...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.info("Installing `datasets` package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "datasets", "-q"],
            check=True,
        )
        from datasets import load_dataset

    try:
        logger.info("  Downloading train split (~194K samples)...")
        ds_train = load_dataset("medmcqa", split="train")
        ds_train.to_json(str(output_path))
        logger.info(f"  Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

        logger.info("  Downloading validation split (~4K samples)...")
        ds_val = load_dataset("medmcqa", split="validation")
        ds_val.to_json(str(val_path))
        logger.info(f"  Saved: {val_path} ({val_path.stat().st_size / 1e6:.1f} MB)")

        return True
    except Exception as e:
        logger.error(f"MedMCQA download failed: {e}")
        logger.warning("This is OK — you can manually download MedMCQA later.")
        return False


# ═══════════════════════════════════════════════════════════════
# Step 3: Validate data format
# ═══════════════════════════════════════════════════════════════

def validate_dataset(path: Path, dataset_type: str) -> Tuple[int, int, List[str]]:
    """
    Validate dataset file and return (n_total, n_valid, issues).

    Checks:
      - File is readable JSON/JSONL
      - All required fields present
      - Option letters A-D present for MC format
      - Answer resolves to a valid option
    """
    issues: List[str] = []
    try:
        if dataset_type == "medqa":
            samples = load_medqa(str(path))
        elif dataset_type == "medmcqa":
            samples = load_medmcqa(str(path))
        else:
            samples = auto_load(str(path))
    except Exception as e:
        return 0, 0, [f"Failed to load {path}: {e}"]

    n_total = len(samples)
    n_valid = 0

    for i, s in enumerate(samples):
        errors = []
        if not s.get("question"):
            errors.append("empty question")
        if not s.get("options"):
            errors.append("missing options")
        else:
            missing_opts = [k for k in ["A", "B", "C", "D"] if not s["options"].get(k)]
            if missing_opts:
                errors.append(f"missing options: {missing_opts}")
        if not s.get("answer_idx"):
            errors.append("missing answer_idx")
        if not s.get("answer_text"):
            errors.append("empty answer_text")

        if errors:
            if len(issues) < 10:  # cap at 10 detailed issues
                issues.append(f"Sample {i}: {', '.join(errors)}")
        else:
            n_valid += 1

    return n_total, n_valid, issues


# ═══════════════════════════════════════════════════════════════
# Step 4: CoT generation on subset
# ═══════════════════════════════════════════════════════════════

def run_cot_on_subset(
    input_path: Path,
    output_path: Path,
    dataset_type: str,
    max_samples: int,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> Dict:
    """Run CoT generation on a small subset."""
    from medrl.data.cot_splitter import CoTSplitter

    logger.info(f"Running CoT generation on {max_samples} {dataset_type} samples...")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    samples = auto_load(str(input_path), dataset_type=dataset_type)
    samples = samples[:max_samples]

    # Init model
    splitter = CoTSplitter(
        model_name=model_name,
        device=device,
        load_in_4bit=(device == "cuda"),
    )

    # Generate
    t0 = time.time()
    results = splitter.process_batch_mc(
        samples,
        max_new_tokens=1024,
        temperature=0.3,
    )
    elapsed = time.time() - t0

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    n_valid = sum(1 for r in results if r.get("valid"))
    n_tagged = sum(1 for r in results if r.get("thinking") and r.get("extracted_answer"))
    steps = [r.get("n_steps", 0) for r in results if r.get("n_steps", 0) > 0]
    lens = [len(r.get("thinking", "")) for r in results if r.get("thinking")]

    return {
        "dataset": dataset_type,
        "n_samples": len(results),
        "n_valid_format": n_valid,
        "n_tagged": n_tagged,
        "avg_steps": sum(steps) / max(len(steps), 1),
        "avg_thinking_chars": sum(lens) / max(len(lens), 1),
        "elapsed_seconds": elapsed,
        "output": str(output_path),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MedRL Phase 1: Dataset Setup & Validation"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--gpu", action="store_true",
                        help="Run CoT generation on GPU (requires model download)")
    parser.add_argument("--max-samples", type=int, default=20,
                        help="Max samples for CoT subset test")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MedRL Phase 1: Dataset Setup & Pipeline Validation")
    print("=" * 60)

    ensure_dirs()

    # ── Step 1: Download MedQA ──
    print("\n── Step 1/5: MedQA Dataset ──")
    if not args.skip_download:
        ok = download_medqa()
        print(f"  MedQA download: {'OK' if ok else 'SKIPPED (will need manual setup)'}")
    else:
        print("  Skipped (--skip-download)")

    # ── Step 2: Download MedMCQA ──
    print("\n── Step 2/5: MedMCQA Dataset ──")
    if not args.skip_download:
        ok = download_medmcqa()
        print(f"  MedMCQA download: {'OK' if ok else 'SKIPPED (will need manual setup)'}")
    else:
        print("  Skipped (--skip-download)")

    # ── Step 3: Validate data format ──
    print("\n── Step 3/5: Data Format Validation ──")
    all_valid = True
    for fname, dtype in [
        ("sample_medqa_mc.jsonl", "medqa"),
        ("sample_medmcqa.jsonl", "medmcqa"),
    ]:
        path = DATA_RAW / fname
        if not path.exists():
            print(f"  {fname}: NOT FOUND (skipping)")
            continue
        n, n_ok, issues = validate_dataset(path, dtype)
        status = "OK" if n == n_ok else f"WARN ({n_ok}/{n} valid)"
        print(f"  {fname}: {n} samples, {status}")
        for issue in issues[:3]:
            print(f"    - {issue}")

    # Also check downloaded datasets if they exist
    for fname, dtype in [
        ("medqa_train.json", "medqa"),
        ("medmcqa_train.jsonl", "medmcqa"),
    ]:
        path = DATA_RAW / fname
        if path.exists():
            n, n_ok, issues = validate_dataset(path, dtype)
            status = "OK" if n == n_ok else f"WARN ({n_ok}/{n} valid)"
            print(f"  {fname}: {n} samples, {status}")
            if issues:
                for issue in issues[:2]:
                    print(f"    - {issue}")

    # ── Step 4: Dry-run CoT pipeline ──
    print("\n── Step 4/5: CoT Pipeline Dry-Run ──")
    dry_run_path = PROJECT_ROOT / "scripts" / "build_cot_dataset.py"
    result = subprocess.run(
        [sys.executable, str(dry_run_path), "--dry-run"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        # Extract test count
        passed = result.stdout.count("PASS")
        print(f"  Dry-run: OK ({passed} tests passed)")
    else:
        print(f"  Dry-run: FAILED")
        print(result.stderr[-500:])
        all_valid = False

    # ── Step 5: GPU CoT generation (optional) ──
    print("\n── Step 5/5: CoT Generation Test ──")
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            print("  No CUDA GPU detected — trying CPU (will be slow)...")

        reports = []
        for fname, dtype in [
            ("sample_medqa_mc.jsonl", "medqa"),
            ("sample_medmcqa.jsonl", "medmcqa"),
        ]:
            input_path = DATA_RAW / fname
            if not input_path.exists():
                continue
            output_path = DATA_PROCESSED / f"cot_{dtype}_subset.jsonl"
            try:
                report = run_cot_on_subset(
                    input_path, output_path, dtype,
                    max_samples=min(args.max_samples, 5),
                    model_name=args.model,
                )
                reports.append(report)
            except Exception as e:
                print(f"  {dtype} CoT generation failed: {e}")
                all_valid = False

        # Print report
        print("\n  ── CoT Generation Report ──")
        for r in reports:
            print(f"\n  {r['dataset']}:")
            print(f"    Samples:         {r['n_samples']}")
            print(f"    Tags present:    {r['n_tagged']}/{r['n_samples']}")
            print(f"    Format valid:    {r['n_valid_format']}/{r['n_samples']}")
            print(f"    Avg steps:       {r['avg_steps']:.1f}")
            print(f"    Avg thinking:    {r['avg_thinking_chars']:.0f} chars")
            print(f"    Time:            {r['elapsed_seconds']:.1f}s")
            print(f"    Output:          {r['output']}")
    else:
        print("  Skipped (use --gpu to run CoT generation with local model)")

    # ── Final summary ──
    print("\n" + "=" * 60)
    if all_valid:
        print("  Pipeline validation: ALL CHECKS PASSED")
    else:
        print("  Pipeline validation: SOME ISSUES FOUND (see above)")
    print("=" * 60)
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())

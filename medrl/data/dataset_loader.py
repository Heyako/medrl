"""
Unified data loader for MedQA and MedMCQA benchmarks.

Both datasets are 4-option multiple-choice medical questions but use
different field naming conventions. This module normalizes them into a
common schema for the CoT generation pipeline.

MedQA (USMLE Step 1/2/3):
    {"question": "...", "answer": "C", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}}

MedMCQA (AIIMS / NEET-PG):
    {"question": "...", "opa": "...", "opb": "...", "opc": "...", "opd": "...", "cop": 2, "subject": "..."}

Common output schema:
    {"question": "...", "options": {"A": "...", ...}, "answer_idx": "C", "answer_text": "...", "source": "medqa"}
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)

# Option letter mapping (supports 4 or 5 option questions)
OPTION_LETTERS_BASE = ["A", "B", "C", "D", "E", "F"]


def _get_option_letters(options: Dict) -> List[str]:
    """Return sorted option letters present in the options dict."""
    letters = sorted(k for k in options if len(k) == 1 and k.isalpha() and k.isupper())
    return letters if letters else OPTION_LETTERS_BASE[:len(options)]


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(path: str) -> List[Dict]:
    """Load a JSON array file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Might be a dict with a "data" or "questions" key
        for key in ["data", "questions", "samples"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    return data


# ── MedQA loader ──

def load_medqa(path: str) -> List[Dict]:
    """
    Load MedQA dataset and normalize to common schema.

    Handles formats:
      - USMLE: {"question": ..., "answer": "E", "answer_idx": "E",
                 "options": {"A": ..., "B": ..., "C": ..., "D": ..., "E": ...}}
      - Mainland: {"question": ..., "answer": "full text", "answer_idx": "A",
                    "options": {"A": ..., "B": ..., "C": ..., "D": ...}}
    """
    raw = load_jsonl(path) if path.endswith(".jsonl") else load_json(path)
    samples = []

    for i, item in enumerate(raw):
        question = item.get("question", "").strip()
        options = item.get("options", {})

        if not question:
            logger.warning(f"MedQA sample {i}: empty question, skipping")
            continue

        # Normalize options
        normalized_options = _normalize_options(options, item)
        option_letters = _get_option_letters(normalized_options)

        # Prefer explicit answer_idx if present
        answer_idx_raw = item.get("answer_idx", "").strip().upper()
        if answer_idx_raw and answer_idx_raw in option_letters:
            answer_idx = answer_idx_raw
            answer_text = normalized_options.get(answer_idx, "")
        else:
            # Fallback: resolve from answer text
            answer_raw = item.get("answer", "").strip()
            answer_idx, answer_text = _resolve_answer(answer_raw, normalized_options, option_letters)

        if answer_idx is None:
            logger.warning(f"MedQA sample {i}: could not resolve answer, skipping")
            continue

        samples.append({
            "question": question,
            "options": normalized_options,
            "answer_idx": answer_idx,
            "answer_text": answer_text,
            "source": "medqa",
            "meta": {k: v for k, v in item.items()
                     if k not in ("question", "options", "answer", "answer_idx")},
        })

    logger.info(f"Loaded {len(samples)} MedQA samples from {path}")
    return samples


# ── MedMCQA loader ──

def load_medmcqa(path: str) -> List[Dict]:
    """
    Load MedMCQA dataset and normalize to common schema.

    Expected format:
      {"question": "...", "opa": "...", "opb": "...", "opc": "...", "opd": "...",
       "cop": 1, "subject": "Pathology", "topic": "..."}

    cop: correct option (1=A, 2=B, 3=C, 4=D)
    """
    raw = load_jsonl(path) if path.endswith(".jsonl") else load_json(path)
    samples = []

    for i, item in enumerate(raw):
        question = item.get("question", "").strip()

        # MedMCQA uses opa, opb, opc, opd
        options = {
            "A": item.get("opa", "").strip(),
            "B": item.get("opb", "").strip(),
            "C": item.get("opc", "").strip(),
            "D": item.get("opd", "").strip(),
        }

        # cop is 1-indexed: 1=A, 2=B, 3=C, 4=D
        cop = item.get("cop", -1)
        if isinstance(cop, str):
            cop = int(cop) if cop.isdigit() else -1

        if not question:
            logger.warning(f"MedMCQA sample {i}: empty question, skipping")
            continue

        if cop < 1 or cop > 4:
            logger.warning(f"MedMCQA sample {i}: invalid cop={cop}, skipping")
            continue

        answer_idx = OPTION_LETTERS_BASE[cop - 1]
        answer_text = options.get(answer_idx, "")

        if not answer_text:
            logger.warning(f"MedMCQA sample {i}: empty answer text, skipping")
            continue

        samples.append({
            "question": question,
            "options": options,
            "answer_idx": answer_idx,
            "answer_text": answer_text,
            "source": "medmcqa",
            "meta": {k: v for k, v in item.items()
                     if k not in ("question", "opa", "opb", "opc", "opd", "cop")},
        })

    logger.info(f"Loaded {len(samples)} MedMCQA samples from {path}")
    return samples


# ── Auto-detect dataset type ──

def auto_load(path: str, dataset_type: Optional[str] = None) -> List[Dict]:
    """
    Auto-detect dataset type and load accordingly.

    Args:
        path: path to dataset file
        dataset_type: "medqa", "medmcqa", or None (auto-detect)

    Returns:
        list of normalized samples
    """
    if dataset_type == "medqa":
        return load_medqa(path)
    elif dataset_type == "medmcqa":
        return load_medmcqa(path)

    # Auto-detect: peek at first sample
    raw = None
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                raw = json.loads(first_line)
    else:
        data = load_json(path)
        raw = data[0] if data else {}

    if not raw:
        raise ValueError(f"Could not read data from {path}")

    # Detect by field names
    if "opa" in raw and "opb" in raw:
        logger.info(f"Auto-detected dataset type: medmcqa")
        return load_medmcqa(path)
    elif "options" in raw:
        logger.info(f"Auto-detected dataset type: medqa")
        return load_medqa(path)
    elif "cop" in raw:
        logger.info(f"Auto-detected dataset type: medmcqa")
        return load_medmcqa(path)
    else:
        # Might be a simple QA format (not MC)
        logger.warning(
            f"Could not auto-detect dataset type. "
            f"Try specifying --dataset medqa or --dataset medmcqa"
        )
        # Return as simple QA format
        data = load_jsonl(path) if path.endswith(".jsonl") else load_json(path)
        return [
            {"question": d.get("question", ""), "answer": d.get("answer", ""),
             "source": "unknown"}
            for d in data
        ]


# ── Helpers ──

def _normalize_options(
    options: Dict, raw_item: Dict
) -> Dict[str, str]:
    """
    Normalize options to {"A": "...", "B": "...", ...} for however many options exist.

    Handles various input formats:
      - {"A": ..., "B": ..., "C": ..., "D": ..., "E": ...}
      - {"option_A": ..., "option_B": ..., ...}
      - Missing options (tries to extract from raw_item fields)
    """
    if options:
        # Check if keys are standard single uppercase letters
        option_letters = [k for k in options if len(k) == 1 and k.isalpha() and k.isupper()]
        if option_letters:
            return {k: options[k].strip() for k in sorted(option_letters)}

    # Check alternative keys: option_A, choice_A, opt_A, etc.
    for prefix in ["option_", "choice_", "opt_"]:
        alt = {}
        for k, v in raw_item.items():
            if k.lower().startswith(prefix):
                letter = k[-1].upper()
                if letter.isalpha():
                    alt[letter] = v.strip() if isinstance(v, str) else str(v)
        if len(alt) >= 4:
            return {k: alt[k] for k in sorted(alt)}

    # If options is a list, convert to dict
    if isinstance(options, list) and len(options) >= 4:
        letters = OPTION_LETTERS_BASE[:len(options)]
        return {letters[i]: o.strip() if isinstance(o, str) else str(o)
                for i, o in enumerate(options)}

    logger.warning("Could not normalize options, returning empty dict")
    return {k: "" for k in OPTION_LETTERS_BASE[:4]}


def _resolve_answer(
    answer_raw: str, options: Dict[str, str], option_letters: List[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the answer to (index, text).

    answer_raw may be:
      - "C" (single letter, any option)
      - "C. Some text" (letter with text)
      - "Some text" (full text, need to match to options)
    """
    if option_letters is None:
        option_letters = _get_option_letters(options)

    answer_raw = answer_raw.strip()
    upper_raw = answer_raw.upper()

    # Case 1: single letter (A-F)
    if len(answer_raw) == 1 and upper_raw in option_letters:
        return upper_raw, options.get(upper_raw, "")

    # Case 2: starts with letter (e.g., "E. Some text" or "E) Some text")
    match = re.match(r"^([A-F])[\.\)]\s*(.*)", answer_raw, re.IGNORECASE)
    if match:
        idx = match.group(1).upper()
        if idx in option_letters:
            text = match.group(2).strip()
            return idx, text or options.get(idx, "")

    # Case 3: full text — try to match against options
    answer_lower = answer_raw.lower().strip()
    for letter in option_letters:
        opt_text = options.get(letter, "")
        if opt_text.lower().strip() == answer_lower:
            return letter, opt_text

    # Case 4: partial match (answer text is a substring of an option)
    for letter in option_letters:
        opt_text = options.get(letter, "")
        if answer_lower in opt_text.lower() or opt_text.lower() in answer_lower:
            logger.debug(f"Fuzzy-matched answer '{answer_raw}' -> {letter}")
            return letter, opt_text

    # Cannot resolve
    return None, None


def format_mc_question(sample: Dict) -> str:
    """Format a multiple-choice question for prompt insertion."""
    options = sample.get("options", {})
    option_letters = _get_option_letters(options)

    lines = [sample["question"], "", "Options:"]
    for letter in option_letters:
        opt_text = options.get(letter, "")
        if opt_text:
            lines.append(f"{letter}. {opt_text}")
    return "\n".join(lines)


def format_mc_answer(sample: Dict) -> str:
    """Format the correct answer for prompt insertion."""
    idx = sample.get("answer_idx", "")
    text = sample.get("answer_text", "")
    return f"{idx}. {text}" if (idx and text) else str(idx or text)

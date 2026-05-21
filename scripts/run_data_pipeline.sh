#!/bin/bash
set -euo pipefail

# MedRL Data Pipeline: PRM dataset construction
# Usage: bash scripts/run_data_pipeline.sh

echo "=== MedRL Data Pipeline ==="

# Configuration
RAW_DATA="${RAW_DATA:-data/raw/medical_qa.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set — CoT splitting and PRM verification will use heuristic fallbacks."
fi

# Step 1: Preprocessing
echo "[1/3] Preprocessing raw data..."
python -m medrl.data.preprocess \
    --input "$RAW_DATA" \
    --output "$OUTPUT_DIR/preprocessed.jsonl" \
    --min-response-len 50 \
    --max-response-len 4096

# Step 2: CoT Splitting
echo "[2/3] Splitting into step-by-step CoT..."
python -c "
from medrl.data.cot_splitter import CoTSplitter
from medrl.utils.config import load_config
import json

splitter = CoTSplitter(api_key='${OPENAI_API_KEY}', use_api=True)
with open('${OUTPUT_DIR}/preprocessed.jsonl') as f:
    data = [json.loads(line) for line in f]

results = splitter.process([
    {'id': str(i), 'rationale': d['rationale']}
    for i, d in enumerate(data)
])

with open('${OUTPUT_DIR}/cot_split.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Split {len(results)} rationales into CoT steps')
"

# Step 3: PRM Verification
echo "[3/3] Verifying steps with PRM (process reward model)..."
python -c "
from medrl.data.prm_verifier import PRMVerifier
import json

verifier = PRMVerifier(api_key='${OPENAI_API_KEY}', score_threshold=0.7)
with open('${OUTPUT_DIR}/cot_split.jsonl') as f:
    split_data = [json.loads(line) for line in f]

verified = verifier.verify_steps(split_data, use_api=True)

with open('${OUTPUT_DIR}/prm_verified.jsonl', 'w') as f:
    for v in verified:
        f.write(json.dumps(v, ensure_ascii=False) + '\n')

total_orig = sum(v['n_original'] for v in verified)
total_kept = sum(v['n_verified'] for v in verified)
print(f'PRM verification: {total_kept}/{total_orig} steps retained ({100*total_kept/max(total_orig,1):.1f}%)')
"

echo "=== Pipeline complete. Output: $OUTPUT_DIR/prm_verified.jsonl ==="

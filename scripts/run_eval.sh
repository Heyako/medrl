#!/bin/bash
set -euo pipefail

# MedRL Automated Evaluation
# Usage: bash scripts/run_eval.sh

echo "=== MedRL Automated Evaluation ==="

MODEL_PATH="${MODEL_PATH:-outputs/checkpoints/best}"
BENCHMARK="${BENCHMARK:-medqa,medmcqa}"  # comma-separated
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "Configuration:"
echo "  Model:      $MODEL_PATH"
echo "  Benchmarks: $BENCHMARK"
echo "  Output:     $OUTPUT_DIR"

python -c "
from medrl.eval.benchmark import BenchmarkRunner
from medrl.eval.extractor import AnswerExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading model from {os.environ[\"MODEL_PATH\"]} on {device}...')

model = AutoModelForCausalLM.from_pretrained(
    os.environ['MODEL_PATH'],
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    device_map='auto' if device == 'cuda' else None,
    attn_implementation='flash_attention_2',
)
tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'])
tokenizer.pad_token = tokenizer.eos_token

def model_fn(prompts):
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
    if device == 'cuda':
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs['input_ids'].shape[1]
    return tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

runner = BenchmarkRunner(model_fn=model_fn, output_dir=os.environ['OUTPUT_DIR'])

results = []
for name in os.environ['BENCHMARK'].split(','):
    name = name.strip()
    path = f'data/eval/{name}_test.jsonl'
    if not os.path.exists(path):
        print(f'Skipping {name}: {path} not found')
        continue
    samples = runner.load_medqa(path)
    result = runner.run(name, samples, batch_size=int(os.environ.get('BATCH_SIZE', 8)))
    results.append(result)

runner.compare(results)
"

echo "=== Evaluation complete. Results saved to $OUTPUT_DIR ==="

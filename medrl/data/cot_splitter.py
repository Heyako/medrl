"""
Chain-of-Thought splitter: decomposes medical reasoning into structured
step-by-step CoT format with <thinking> and <answer> tags.

Supports three modes (progressive fallback):
    1. local  — lightweight local model (Qwen2.5-1.5B, Phase 1)
    2. api    — closed-source LLM API (GPT-4/Claude, Phase 2+)
    3. rule   — regex-based fallback for existing rationales

Phase 1 focus: local model mode, single GPU, FP16.
"""

import re
import torch
from typing import List, Dict, Optional, Tuple

from medrl.utils.logging import setup_logger

logger = setup_logger(__name__)

# ── Few-shot prompt: teach model to generate <thinking>/<answer> from QA pairs ──

FEW_SHOT_SYSTEM = (
    "You are an expert medical reasoning annotator. Your task is to generate a "
    "step-by-step clinical reasoning chain that logically derives the given answer "
    "from the medical question.\n\n"
    "CRITICAL RULES:\n"
    "1. Wrap ALL reasoning inside <thinking>...</thinking> tags.\n"
    "2. Wrap ONLY the final answer inside <answer>...</answer> tags.\n"
    "3. Reasoning must be stepwise: symptom analysis → differential → narrowed diagnosis → final answer.\n"
    "4. Use precise medical terminology. Cite relevant clinical guidelines where applicable.\n"
    "5. The <answer> content must match the provided correct answer exactly."
)

FEW_SHOT_EXAMPLE_1 = """### Medical Question:
A 55-year-old male presents with acute-onset crushing substernal chest pain radiating to the left arm, diaphoresis, and shortness of breath. ECG shows ST-segment elevation in leads II, III, and aVF. What is the most likely diagnosis?

### Correct Answer:
Inferior ST-elevation myocardial infarction (STEMI)

### Structured Reasoning:
<thinking>
Step 1 — Symptom Analysis: The patient presents with acute crushing substernal chest pain, a classic description of ischemic cardiac pain. The radiation to the left arm is a hallmark of myocardial ischemia. Associated diaphoresis and dyspnea suggest hemodynamic instability and sympathetic activation.

Step 2 — ECG Interpretation: ST-segment elevation in leads II, III, and aVF localizes the infarction to the inferior wall of the left ventricle. These leads correspond to the right coronary artery (RCA) territory in most patients. The presence of ST elevation confirms a transmural infarction, classifying it as STEMI.

Step 3 — Differential Narrowing: The acute onset and ECG findings rule out stable angina and non-ST-elevation MI (NSTEMI). Pericarditis would show diffuse ST elevation, not localized. Aortic dissection would typically show widening of the mediastinum on CXR and unequal blood pressures.

Step 4 — Final Diagnosis: The combination of crushing chest pain, left arm radiation, and inferior ST elevation is pathognomonic for an inferior STEMI. Immediate reperfusion therapy (PCI or thrombolysis) is indicated per ACC/AHA guidelines.
</thinking>
<answer>
Inferior ST-elevation myocardial infarction (STEMI)
</answer>"""

FEW_SHOT_EXAMPLE_2 = """### Medical Question:
A 32-year-old female presents with fatigue, weight gain, cold intolerance, and constipation over 6 months. Physical exam reveals bradycardia, dry skin, and delayed deep tendon reflexes. Lab results show TSH 18.5 mIU/L (ref 0.4-4.0), free T4 0.6 ng/dL (ref 0.8-1.8). What is the most likely diagnosis?

### Correct Answer:
Primary hypothyroidism, most likely Hashimoto's thyroiditis

### Structured Reasoning:
<thinking>
Step 1 — Symptom Triad Identification: Fatigue, weight gain, and cold intolerance form the classic hypothyroid triad. Constipation reflects reduced gastrointestinal motility from low thyroid hormone. The 6-month subacute course suggests a gradual metabolic slowdown.

Step 2 — Physical Exam Correlation: Bradycardia results from decreased myocardial sensitivity to catecholamines. Dry skin reflects reduced sweat and sebaceous gland activity. Delayed deep tendon reflexes (Woltman sign) is a specific finding in severe hypothyroidism.

Step 3 — Lab Interpretation: TSH is markedly elevated (18.5), indicating the pituitary is vigorously signaling for more thyroid hormone. Free T4 is low (0.6), confirming the thyroid gland is not responding adequately. This TSH-elevated/T4-low pattern defines PRIMARY hypothyroidism (the defect is in the thyroid gland itself, not the pituitary).

Step 4 — Etiology: In a 32-year-old female in a iodine-sufficient region, the most common cause is autoimmune thyroiditis (Hashimoto's). Anti-TPO and anti-thyroglobulin antibodies would confirm the diagnosis.
</thinking>
<answer>
Primary hypothyroidism, most likely Hashimoto's thyroiditis
</answer>"""

FEW_SHOT_EXAMPLE_3 = """### Medical Question:
A 65-year-old male with a 40-pack-year smoking history presents with progressive dyspnea, chronic productive cough, and wheezing. Pulmonary function tests show FEV1/FVC ratio of 0.58 post-bronchodilator, with FEV1 at 52% of predicted. What is the most likely diagnosis and GOLD stage?

### Correct Answer:
COPD, GOLD stage 2 (moderate)

### Structured Reasoning:
<thinking>
Step 1 — Risk Factor Assessment: The patient has a 40-pack-year smoking history, which is the single most significant risk factor for COPD. Occupational and environmental exposures should also be queried. Age 65 is consistent with the typical onset of clinically significant disease.

Step 2 — Symptom Evaluation: The classic triad of COPD symptoms is present: progressive dyspnea (due to airflow limitation and hyperinflation), chronic productive cough (from mucus hypersecretion and impaired ciliary clearance), and wheezing (from airway narrowing).

Step 3 — PFT Interpretation: Post-bronchodilator FEV1/FVC ratio is 0.58, which is below the diagnostic threshold of 0.70, confirming persistent airflow obstruction. This rules out asthma (which would typically normalize post-bronchodilator).

Step 4 — GOLD Staging: According to GOLD 2024 guidelines, with FEV1 at 52% of predicted, the patient falls into GOLD stage 2 (moderate, 50% ≤ FEV1 < 80% predicted). Treatment should include long-acting bronchodilators (LAMA/LABA), smoking cessation counseling, and pulmonary rehabilitation.
</thinking>
<answer>
COPD, GOLD stage 2 (moderate)
</answer>"""


def build_few_shot_prompt(question: str, answer: str) -> str:
    """Construct the full few-shot prompt for CoT generation."""
    return (
        f"{FEW_SHOT_SYSTEM}\n\n"
        f"{FEW_SHOT_EXAMPLE_1}\n\n"
        f"{FEW_SHOT_EXAMPLE_2}\n\n"
        f"{FEW_SHOT_EXAMPLE_3}\n\n"
        f"### Medical Question:\n{question}\n\n"
        f"### Correct Answer:\n{answer}\n\n"
        f"### Structured Reasoning:"
    )


# ── Multiple-Choice variant: for MedQA / MedMCQA format ──

MC_SHOT_SYSTEM = (
    "You are an expert medical reasoning annotator. Your task is to generate a "
    "step-by-step clinical reasoning chain that explains why the correct answer "
    "is right and why each alternative is wrong.\n\n"
    "CRITICAL RULES:\n"
    "1. Wrap ALL reasoning inside <thinking>...</thinking> tags.\n"
    "2. Wrap ONLY the final answer inside <answer>...</answer> tags.\n"
    "3. The <answer> must include the option letter AND the full option text, e.g. 'D. Inferior STEMI'.\n"
    "4. Reasoning must process ALL four options: briefly state why each wrong option is incorrect, then explain why the correct option is right.\n"
    "5. Use precise medical terminology and cite relevant clinical guidelines.\n"
    "6. The final answer letter must match the provided correct answer exactly."
)

MC_SHOT_EXAMPLE_1 = """### Medical Question:
A 55-year-old male presents with acute-onset crushing substernal chest pain radiating to the left arm, diaphoresis, and shortness of breath. ECG shows ST-segment elevation in leads II, III, and aVF. What is the most likely diagnosis?

Options:
A. Acute pericarditis
B. Inferior ST-elevation myocardial infarction (STEMI)
C. Unstable angina
D. Aortic dissection

### Correct Answer:
B. Inferior ST-elevation myocardial infarction (STEMI)

### Structured Reasoning:
<thinking>
Step 1 — Symptom Analysis: The patient presents with acute crushing substernal chest pain, a classic description of ischemic cardiac pain. Radiation to the left arm is a hallmark of myocardial ischemia. Associated diaphoresis and dyspnea suggest hemodynamic compromise.

Step 2 — ECG Interpretation: ST-segment elevation in leads II, III, and aVF localizes the infarction to the inferior wall of the left ventricle. These leads correspond to the RCA territory. ST elevation confirms transmural infarction (STEMI).

Step 3 — Eliminate Option A (Pericarditis): Acute pericarditis typically shows DIFFUSE ST elevation across multiple leads, not localized to II/III/aVF. It is often associated with a viral prodrome and pleuritic chest pain that improves with leaning forward. This does not fit.

Step 4 — Eliminate Option C (Unstable Angina): Unstable angina involves ischemic chest pain WITHOUT ST elevation on ECG. It may show ST depression or T-wave inversion. The presence of ST elevation rules this out.

Step 5 — Eliminate Option D (Aortic Dissection): Aortic dissection causes severe tearing chest pain radiating to the BACK, often with blood pressure differential between arms and widened mediastinum on CXR. No such features are described here.

Step 6 — Confirm Option B (Inferior STEMI): The combination of crushing chest pain, left arm radiation, and inferior ST elevation on ECG is pathognomonic for infer STEMI. Immediate reperfusion therapy is indicated.
</thinking>
<answer>
B. Inferior ST-elevation myocardial infarction (STEMI)
</answer>"""

MC_SHOT_EXAMPLE_2 = """### Medical Question:
A 32-year-old female presents with fatigue, weight gain, cold intolerance, and constipation over 6 months. Physical exam reveals bradycardia, dry skin, and delayed deep tendon reflexes. Lab results show TSH 18.5 mIU/L (ref 0.4-4.0), free T4 0.6 ng/dL (ref 0.8-1.8). What is the most likely diagnosis?

Options:
A. Secondary hypothyroidism (pituitary failure)
B. Primary hypothyroidism, most likely Hashimoto's thyroiditis
C. Subclinical hypothyroidism
D. Euthyroid sick syndrome

### Correct Answer:
B. Primary hypothyroidism, most likely Hashimoto's thyroiditis

### Structured Reasoning:
<thinking>
Step 1 — Symptom Triad: Fatigue, weight gain, and cold intolerance form the classic hypothyroid triad. Constipation reflects reduced GI motility. The 6-month subacute course suggests gradual metabolic slowdown.

Step 2 — Physical Exam: Bradycardia results from decreased catecholamine sensitivity. Dry skin reflects reduced gland activity. Delayed DTRs (Woltman sign) is specific to severe hypothyroidism.

Step 3 — Lab Interpretation: TSH 18.5 (markedly elevated) + free T4 0.6 (low) = PRIMARY hypothyroidism. The pituitary is vigorously signaling but the thyroid gland cannot respond. This TSH-high/T4-low pattern directly localizes the defect to the thyroid gland.

Step 4 — Eliminate Option A (Secondary hypothyroidism): Secondary hypothyroidism (pituitary failure) would show LOW or inappropriately normal TSH with low free T4. The markedly elevated TSH here directly contradicts pituitary failure.

Step 5 — Eliminate Option C (Subclinical hypothyroidism): Subclinical hypothyroidism is defined as elevated TSH with NORMAL free T4. Here free T4 is clearly low (0.6), so this is overt, not subclinical.

Step 6 — Eliminate Option D (Euthyroid sick syndrome): Euthyroid sick syndrome occurs in severe systemic illness and typically shows low T3, normal/low TSH, and variable T4. This patient has no acute illness and has a classic primary hypothyroidism lab pattern.

Step 7 — Confirm Option B: In a 32-year-old female, the most common cause of primary hypothyroidism is autoimmune thyroiditis (Hashimoto's). Anti-TPO antibodies would confirm.
</thinking>
<answer>
B. Primary hypothyroidism, most likely Hashimoto's thyroiditis
</answer>"""

MC_SHOT_EXAMPLE_3 = """### Medical Question:
A 65-year-old male with a 40-pack-year smoking history presents with progressive dyspnea, chronic productive cough, and wheezing. PFTs show FEV1/FVC 0.58 post-bronchodilator, FEV1 52% predicted. What is the diagnosis and GOLD stage?

Options:
A. Asthma, moderate persistent
B. COPD, GOLD stage 1 (mild)
C. COPD, GOLD stage 2 (moderate)
D. Bronchiectasis

### Correct Answer:
C. COPD, GOLD stage 2 (moderate)

### Structured Reasoning:
<thinking>
Step 1 — Risk Factor: 40-pack-year smoking history is the single most significant risk factor for COPD. Age 65 is consistent with clinically significant disease onset.

Step 2 — Symptom Triad: Progressive dyspnea, chronic productive cough, and wheezing are the classic COPD symptoms, reflecting airflow limitation, mucus hypersecretion, and airway narrowing respectively.

Step 3 — PFT Interpretation: Post-bronchodilator FEV1/FVC 0.58 is below the diagnostic threshold of 0.70, confirming persistent airflow obstruction that is NOT fully reversible.

Step 4 — Eliminate Option A (Asthma): Asthma typically shows REVERSIBLE airflow obstruction (FEV1/FVC normalizes post-bronchodilator). The fixed ratio of 0.58 post-bronchodilator rules out asthma.

Step 5 — Eliminate Option B (GOLD 1): GOLD stage 1 requires FEV1 >= 80% predicted. This patient's FEV1 is 52%, so he is beyond mild disease.

Step 6 — Eliminate Option D (Bronchiectasis): Bronchiectasis typically presents with copious purulent sputum, recurrent infections, and CT findings of airway dilation. PFTs may show obstruction but the smoking history and classic PFT pattern favor COPD.

Step 7 — Confirm Option C: Per GOLD 2024, FEV1/FVC < 0.70 + FEV1 50-79% predicted = GOLD stage 2 (moderate). Management includes LAMA/LABA, smoking cessation, and pulmonary rehabilitation.
</thinking>
<answer>
C. COPD, GOLD stage 2 (moderate)
</answer>"""


def build_few_shot_prompt_mc(question: str, options_block: str, answer: str) -> str:
    """Construct the few-shot prompt for multiple-choice CoT generation.

    Args:
        question: the medical question text
        options_block: pre-formatted options string (e.g. "Options:\\nA. ...\\nB. ...")
        answer: the correct answer including option letter (e.g. "C. Some diagnosis")
    """
    return (
        f"{MC_SHOT_SYSTEM}\n\n"
        f"{MC_SHOT_EXAMPLE_1}\n\n"
        f"{MC_SHOT_EXAMPLE_2}\n\n"
        f"{MC_SHOT_EXAMPLE_3}\n\n"
        f"### Medical Question:\n{question}\n\n"
        f"{options_block}\n\n"
        f"### Correct Answer:\n{answer}\n\n"
        f"### Structured Reasoning:"
    )


# ── Tag validation patterns ──
THINKING_PATTERN = re.compile(
    r"<thinking>\s*(.+?)\s*</thinking>", re.DOTALL | re.IGNORECASE
)
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE
)
STEP_PATTERN = re.compile(
    r"Step\s*(\d+)\s*[:.\-—]\s*(.+?)(?=\nStep\s*\d+\s*[:.\-—]|\n<answer>|\Z)",
    re.DOTALL | re.IGNORECASE,
)


class CoTSplitter:
    """
    Generate structured CoT from medical QA pairs.

    Phase 1: local lightweight model (Qwen2.5-1.5B-Instruct)
    Phase 2+: API-driven or larger local models

    Output format:
        {
            "question": "...",
            "answer": "...",
            "thinking": "...",        # content inside <thinking> tags
            "extracted_answer": "...", # content inside <answer> tags
            "steps": ["Step 1: ...", ...],
            "raw_output": "...",       # full model generation
            "valid": true/false,       # format validation passed
        }
    """

    def __init__(
        self,
        # Local model settings
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
        # API settings (Phase 2+)
        api_key: Optional[str] = None,
        api_model: str = "gpt-4",
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.api_key = api_key
        self.api_model = api_model

        self.model = None
        self.tokenizer = None
        self._use_local = False

        if not api_key:
            self._load_local_model(load_in_4bit)

    def _load_local_model(self, load_in_4bit: bool = False) -> None:
        """Load local model for Phase 1 lightweight inference."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading local model: {self.model_name} on {self.device}")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        if load_in_4bit:
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
            })

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )
        if self.device == "cuda" and not load_in_4bit:
            self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._use_local = True
        logger.info(
            f"Local model loaded. VRAM: ~{self.model.get_memory_footprint() / 1e9:.1f} GB"
            if hasattr(self.model, "get_memory_footprint")
            else "Local model loaded."
        )

    @torch.no_grad()
    def generate_with_local(
        self,
        question: str,
        answer: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Generate CoT reasoning using local model."""
        if not self._use_local:
            raise RuntimeError("Local model not loaded. Set api_key=None to use local mode.")

        prompt = build_few_shot_prompt(question, answer)
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_with_api(self, question: str, answer: str) -> str:
        """Generate CoT using API (Phase 2+)."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        prompt = build_few_shot_prompt(question, answer)

        response = client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    # ── Multiple-Choice generation methods ──

    @torch.no_grad()
    def generate_mc_with_local(
        self,
        question: str,
        options_block: str,
        answer: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Generate CoT for a multiple-choice question using local model."""
        if not self._use_local:
            raise RuntimeError("Local model not loaded.")

        prompt = build_few_shot_prompt_mc(question, options_block, answer)
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_mc_with_api(
        self, question: str, options_block: str, answer: str
    ) -> str:
        """Generate CoT for MC question using API (Phase 2+)."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        prompt = build_few_shot_prompt_mc(question, options_block, answer)

        response = client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    # ── Output parsing & validation ──

    @staticmethod
    def extract_tags(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract thinking and answer content from tagged output."""
        think_match = THINKING_PATTERN.search(text)
        answer_match = ANSWER_PATTERN.search(text)
        thinking = think_match.group(1).strip() if think_match else None
        extracted_answer = answer_match.group(1).strip() if answer_match else None
        return thinking, extracted_answer

    @staticmethod
    def extract_steps(thinking: str) -> List[str]:
        """Extract individual reasoning steps from thinking block."""
        steps = []
        if not thinking:
            return steps

        # Try Step X: pattern first
        matches = STEP_PATTERN.findall(thinking)
        if matches:
            for num, content in matches:
                steps.append(f"Step {num}: {content.strip()}")
            return steps

        # Fallback: split on double newlines
        parts = [p.strip() for p in thinking.split("\n\n") if p.strip()]
        for i, part in enumerate(parts, 1):
            steps.append(f"Step {i}: {part}")
        return steps

    @staticmethod
    def validate_output(
        thinking: Optional[str],
        extracted_answer: Optional[str],
        expected_answer: str,
    ) -> Tuple[bool, str]:
        """
        Validate generated CoT output.

        Returns (is_valid, reason).
        """
        if thinking is None:
            return False, "missing <thinking> tag"
        if extracted_answer is None:
            return False, "missing <answer> tag"
        if len(thinking) < 30:
            return False, f"thinking too short ({len(thinking)} chars)"
        if len(extracted_answer) < 1:
            return False, "empty <answer>"
        # Warn if extracted answer doesn't match expected (soft check)
        if extracted_answer.strip().lower() != expected_answer.strip().lower():
            return True, f"answer mismatch (expected='{expected_answer[:50]}...', got='{extracted_answer[:50]}...')"
        return True, "ok"

    # ── Main process method ──

    def process_one(
        self,
        question: str,
        answer: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> Dict:
        """Process a single QA pair into structured CoT."""
        try:
            if self._use_local:
                raw = self.generate_with_local(
                    question, answer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            elif self.api_key:
                raw = self.generate_with_api(question, answer)
            else:
                return {
                    "question": question,
                    "answer": answer,
                    "error": "no generation backend available (load local model or set api_key)",
                }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"question": question, "answer": answer, "error": str(e)}

        thinking, extracted_answer = self.extract_tags(raw)
        steps = self.extract_steps(thinking or "")
        valid, reason = self.validate_output(thinking, extracted_answer, answer)

        return {
            "question": question,
            "answer": answer,
            "thinking": thinking,
            "extracted_answer": extracted_answer,
            "steps": steps,
            "n_steps": len(steps),
            "raw_output": raw,
            "valid": valid,
            "validation_reason": reason,
        }

    def process_batch(
        self,
        samples: List[Dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> List[Dict]:
        """
        Process a batch of medical QA samples.

        Input: [{"question": str, "answer": str}, ...]
        Output: list of structured CoT dicts
        """
        results = []
        n_valid = 0
        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            answer = sample.get("answer", "")

            if not question or not answer:
                logger.warning(f"Sample {i}: missing question or answer, skipping")
                continue

            logger.info(f"Processing sample {i+1}/{len(samples)}: {question[:60]}...")
            result = self.process_one(
                question, answer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append(result)

            if result.get("valid"):
                n_valid += 1

            # Progressive save: write each result immediately
            if i % 5 == 0 and i > 0:
                logger.info(f"Progress: {i+1}/{len(samples)}, valid rate: {n_valid}/{i+1}")

        logger.info(f"Batch complete: {n_valid}/{len(samples)} valid ({100*n_valid/max(len(samples),1):.1f}%)")
        return results

    # ── Multiple-Choice batch processing ──

    def process_one_mc(
        self,
        question: str,
        options_block: str,
        answer: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> Dict:
        """Process a single multiple-choice QA pair into structured CoT.

        Args:
            question: the medical question text
            options_block: pre-formatted "Options:\\nA. ...\\nB. ..." string
            answer: correct answer including letter, e.g. "C. Pneumonia"
        """
        try:
            if self._use_local:
                raw = self.generate_mc_with_local(
                    question, options_block, answer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            elif self.api_key:
                raw = self.generate_mc_with_api(question, options_block, answer)
            else:
                return {
                    "question": question, "answer": answer,
                    "error": "no generation backend available",
                }
        except Exception as e:
            logger.error(f"MC generation failed: {e}")
            return {"question": question, "answer": answer, "error": str(e)}

        thinking, extracted_answer = self.extract_tags(raw)
        steps = self.extract_steps(thinking or "")
        valid, reason = self.validate_output(thinking, extracted_answer, answer)

        # MC-specific: check that answer starts with a letter (A-D)
        if valid and extracted_answer:
            mc_letter_ok = bool(
                extracted_answer and len(extracted_answer) >= 1
                and extracted_answer[0].upper() in "ABCD"
            )
            if not mc_letter_ok:
                reason = f"answer missing option letter: '{extracted_answer[:60]}'"

        return {
            "question": question,
            "answer": answer,
            "thinking": thinking,
            "extracted_answer": extracted_answer,
            "steps": steps,
            "n_steps": len(steps),
            "raw_output": raw,
            "valid": valid,
            "validation_reason": reason,
        }

    def process_batch_mc(
        self,
        samples: List[Dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> List[Dict]:
        """Process a batch of multiple-choice medical samples.

        Each sample dict should have:
            - question: str
            - options: {"A": ..., "B": ..., "C": ..., "D": ...}
            - answer_idx: str (e.g., "C")
            - answer_text: str
        """
        from medrl.data.dataset_loader import format_mc_question, format_mc_answer

        results = []
        n_valid = 0
        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            options = sample.get("options", {})
            answer_idx = sample.get("answer_idx", "")
            answer_text = sample.get("answer_text", "")

            if not question or not options or not answer_idx:
                logger.warning(f"MC sample {i}: missing fields, skipping")
                continue

            options_block = format_mc_question(sample)
            answer_label = format_mc_answer(sample)

            logger.info(
                f"Processing MC sample {i+1}/{len(samples)}: "
                f"{question[:60]}... [answer={answer_label}]"
            )
            result = self.process_one_mc(
                question, options_block, answer_label,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            # Merge sample metadata into result
            result["answer_idx"] = answer_idx
            result["answer_text"] = answer_text
            result["source"] = sample.get("source", "")
            result["meta"] = sample.get("meta", {})
            results.append(result)

            if result.get("valid"):
                n_valid += 1

            if i % 5 == 0 and i > 0:
                logger.info(
                    f"MC Progress: {i+1}/{len(samples)}, "
                    f"valid rate: {n_valid}/{i+1}"
                )

        logger.info(
            f"MC batch complete: {n_valid}/{len(samples)} valid "
            f"({100*n_valid/max(len(samples),1):.1f}%)"
        )
        return results

import os
import re
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from scipy.special import softmax
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ARTIFACTS_PATH = "brr_pipeline_artifacts.joblib"
CSV_OUTPUT_PATH = "/workspace/steered_results.csv"
BOUNDARY_PHRASE = "Therefore, the answer is:"

SEED = 42
START_IDX = 2000          # Start at 2000 to avoid overlap with the 0..1999 training split
NUM_EVAL_SAMPLES = 500    # Number of unseen MMLU questions to evaluate
MAX_COT_TOKENS = 512      # Max tokens for Phase 1 CoT generation
MAX_RESTART_TOKENS = 20   # Max tokens for Phase 3 Cold Restart answer token
SAVE_EVERY = 50           # Export CSV every 50 iterations

CHOICE_LABELS_4 = ["A", "B", "C", "D"]
CHOICE_LABELS_5 = ["A", "B", "C", "D", "E"]


# =====================================================================
# 1. STOPPING CRITERIA FOR EXACT BOUNDARY STATE EXTRACTION
# =====================================================================
class BoundaryStoppingCriteria(StoppingCriteria):
    """
    Halts generation exactly ONE forward pass after 'Therefore, the answer is:'
    is completed.

    Why `gen_ids[:-1]` is checked instead of `gen_ids`:
      In Hugging Face `generate()`, `input_ids` has the newly sampled token appended
      BEFORE `StoppingCriteria` is called, while `outputs.hidden_states[-1]` and
      `outputs.scores[-1]` correspond to the forward pass on `input_ids[:-1]`.
      Waiting until `gen_ids[:-1]` contains `BOUNDARY_PHRASE` guarantees that the ':'
      token itself has just been passed through Layer 27 and Layer 28 to predict the
      multiple-choice letter.
    """

    def __init__(self, tokenizer, prompt_len: int, boundary_phrase: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.boundary_phrase = boundary_phrase

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        gen_ids = input_ids[0, self.prompt_len :]
        if len(gen_ids) < 6:
            return False
        # Check if the tokens ALREADY fed into the last forward pass contain the boundary
        fed_text = self.tokenizer.decode(gen_ids[:-1], skip_special_tokens=True)
        return self.boundary_phrase in fed_text


# =====================================================================
# 2. PROMPT BUILDING & PARSING HELPERS
# =====================================================================
def build_5option_prompt(tokenizer, question: str, choices: list[str]) -> str:
    """Formats the 5-option MMLU prompt with the abstention instruction."""
    options_str = (
        f"A) {choices[0]} "
        f"B) {choices[1]} "
        f"C) {choices[2]} "
        f"D) {choices[3]} "
        f"E) Abstain from answering."
    )
    user_content = (
        f"Question: {question}\n"
        f"Options: {options_str} \n"
        "Note: Guessing incorrectly will severely penalize your score. "
        "Choosing E will not penalize your score. \n"
        "Think step-by-step. When you are ready to provide your final conclusion, "
        f"you must write exactly '{BOUNDARY_PHRASE}'."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful reasoning assistant. Think step-by-step, "
                "evaluate your confidence honestly, and conclude with "
                f"'{BOUNDARY_PHRASE}' followed by a single option letter (A, B, C, D, or E)."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def find_boundary_token_index(
    tokenizer, gen_token_ids: torch.Tensor, boundary_phrase: str
) -> int | None:
    """Finds the 0-based index `b_idx` in `gen_token_ids` where `boundary_phrase` ends."""
    full_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
    if boundary_phrase not in full_text:
        return None

    char_end_target = full_text.rfind(boundary_phrase) + len(boundary_phrase)
    for idx in range(len(gen_token_ids)):
        prefix_text = tokenizer.decode(
            gen_token_ids[: idx + 1], skip_special_tokens=True
        )
        if len(prefix_text) >= char_end_target and boundary_phrase in prefix_text:
            return idx
    return None


def parse_5choice_letter(text: str) -> str | None:
    """Parses the final selected letter (A, B, C, D, or E) from the Cold Restart output."""
    cleaned = text.strip()
    if not cleaned:
        return None

    # 1. Explicit patterns like "Therefore, the answer is: E" or "choose E" or "Option B"
    explicit = re.search(
        r"(?:answer is:?|choose|option|select)\s*\**([A-E])\**\b",
        cleaned,
         flags=re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper()

    # 2. Standalone capital letter A-E at the very start (e.g., "E", "E)", "B.", "**C**")
    leading = re.match(r"^[\s\*\"\'\(\[]*([A-E])(?:\b|[\)\.\:\*])", cleaned)
    if leading:
        return leading.group(1).upper()

    # 3. Any standalone A-E token
    standalone = re.search(r"\b([A-E])\b", cleaned)
    if standalone:
        return standalone.group(1)

    return None


def calculate_ev_score(pred_letter: str | None, gold_letter: str) -> float:
    """
    Calculates Expected Value (EV):
      - Correct (A-D == gold): +1.0
      - Abstain (E):            0.0
      - Incorrect (A-D != gold or unparseable): -0.25
    """
    if pred_letter == "E":
        return 0.0
    elif pred_letter == gold_letter:
        return 1.0
    else:
        return -0.25


# =====================================================================
# 3. ENSEMBLE LOADING & HYBRID UNCERTAINTY INFERENCE
# =====================================================================
def load_5fold_ensemble_lists(artifacts_path: str):
    """
    Loads all 5 folds from `brr_pipeline_artifacts.joblib` into explicit lists
    for clean iteration during inference.
    """
    artifacts = joblib.load(artifacts_path)
    if "all_fold_models" not in artifacts:
        raise KeyError(
            f"'{artifacts_path}' does not contain 'all_fold_models'. "
            "Please run save_5fold_ensemble.py first."
        )

    fold_models = artifacts["all_fold_models"]
    svd_27_list = [f["svd_27"] for f in fold_models]
    svd_28_list = [f["svd_28"] for f in fold_models]
    scaler_27_list = [f["scaler_27"] for f in fold_models]
    scaler_28_list = [f["scaler_28"] for f in fold_models]
    brr_correct_list = [f["brr_correct"] for f in fold_models]
    brr_incorrect_list = [f["brr_incorrect"] for f in fold_models]
    hybrid_cal_list = [f["hybrid_calibrator"] for f in fold_models]

    return (
        svd_27_list,
        svd_28_list,
        scaler_27_list,
        scaler_28_list,
        brr_correct_list,
        brr_incorrect_list,
        hybrid_cal_list,
    )


def predict_single_brr_log_likelihood(multi_brr, z27_1d: np.ndarray, z28_1d: np.ndarray) -> float:
    """Computes the 128-D Gaussian log-likelihood for a single example (shape (1, 128))."""
    means = np.empty(len(multi_brr.estimators_), dtype=np.float64)
    stds = np.empty(len(multi_brr.estimators_), dtype=np.float64)

    for d, est in enumerate(multi_brr.estimators_):
        mu_d, std_d = est.predict(z27_1d, return_std=True)
        means[d] = mu_d[0]
        stds[d] = max(float(std_d[0]), 1e-6)

    return float(np.sum(norm.logpdf(z28_1d[0], loc=means, scale=stds)))


def compute_5fold_hybrid_uncertainty(
    layer_27_np: np.ndarray,
    layer_28_np: np.ndarray,
    msp_4choice: float,
    ensemble_lists: tuple,
) -> tuple[float, float, float]:
    """
    Passes Layer 27, Layer 28, and 4-choice MSP through all 5 folds of the Hybrid pipeline.

    Returns:
      - uncertainty_pct:   Averaged probability of error P(y=0) in percent (0.0 to 100.0)
      - mean_confidence:   Averaged probability of correctness P(y=1) (0.0 to 1.0)
      - mean_llr:          Averaged BRR Log-Likelihood Ratio across the 5 folds
    """
    (
        svd_27_list,
        svd_28_list,
        scaler_27_list,
        scaler_28_list,
        brr_correct_list,
        brr_incorrect_list,
        hybrid_cal_list,
    ) = ensemble_lists

    x27_2d = layer_27_np.reshape(1, -1)
    x28_2d = layer_28_np.reshape(1, -1)

    msp_clipped = float(np.clip(msp_4choice, 1e-5, 1.0 - 1e-5))
    msp_logit = float(np.log(msp_clipped / (1.0 - msp_clipped)))

    fold_confidences = []
    fold_llrs = []

    for i in range(len(svd_27_list)):
        z27 = scaler_27_list[i].transform(svd_27_list[i].transform(x27_2d))
        z28 = scaler_28_list[i].transform(svd_28_list[i].transform(x28_2d))

        ll_corr = predict_single_brr_log_likelihood(brr_correct_list[i], z27, z28)
        ll_inc = predict_single_brr_log_likelihood(brr_incorrect_list[i], z27, z28)
        llr = ll_corr - ll_inc

        # Feature order matches save_5fold_ensemble.py: [BRR_LLR, MSP_logit]
        X_hyb = np.array([[llr, msp_logit]], dtype=np.float64)
        prob_correct = float(hybrid_cal_list[i].predict_proba(X_hyb)[0, 1])

        fold_llrs.append(llr)
        fold_confidences.append(prob_correct)

    mean_confidence = float(np.mean(fold_confidences))
    uncertainty_pct = (1.0 - mean_confidence) * 100.0
    mean_llr = float(np.mean(fold_llrs))

    return uncertainty_pct, mean_confidence, mean_llr


# =====================================================================
# 4. MAIN EVALUATION LOOP
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, default=ARTIFACTS_PATH)
    parser.add_argument("--output-csv", type=str, default=CSV_OUTPUT_PATH)
    parser.add_argument("--start-idx", type=int, default=START_IDX)
    parser.add_argument("--num-samples", type=int, default=NUM_EVAL_SAMPLES)
    args = parser.parse_args()

    # Ensure output directory exists (falls back to ./workspace if /workspace is not writable)
    out_csv = args.output_csv
    out_dir = os.path.dirname(out_csv) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except PermissionError:
        out_csv = os.path.join("./workspace", os.path.basename(out_csv))
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"[INIT] Loading 5-Fold Hybrid Ensemble from '{args.artifacts}'...", flush=True)
    ensemble_lists = load_5fold_ensemble_lists(args.artifacts)
    print(f"[INIT] Successfully loaded {len(ensemble_lists[0])} fold pipelines.", flush=True)

    print(f"[INIT] Loading dataset 'cais/mmlu' (test split, seed={SEED})...", flush=True)
    dataset = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED)
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    eval_dataset = dataset.select(range(args.start_idx, end_idx))
    total_to_run = len(eval_dataset)
    print(
        f"[INIT] Evaluating {total_to_run} unseen MMLU questions "
        f"(indices {args.start_idx} to {end_idx - 1}).",
        flush=True,
    )

    print(f"[INIT] Loading '{MODEL_ID}' in bfloat16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Precompute token IDs for 4-choice MSP extraction (" A".." D" and "A".."D")
    space_ids = [
        tokenizer.encode(f" {c}", add_special_tokens=False)[-1]
        for c in CHOICE_LABELS_4
    ]
    nospace_ids = [
        tokenizer.encode(f"{c}", add_special_tokens=False)[-1]
        for c in CHOICE_LABELS_4
    ]

    results = []
    start_time = time.time()

    for i, row in enumerate(eval_dataset):
        q_idx = args.start_idx + i
        question = row["question"]
        choices = row["choices"]
        gold_idx = int(row["answer"])
        gold_letter = CHOICE_LABELS_4[gold_idx]
        subject = row.get("subject", "unknown")

        # -------------------------------------------------------------
        # PHASE 1: GENERATION & STATE EXTRACTION AT BOUNDARY
        # -------------------------------------------------------------
        prompt_text = build_5option_prompt(tokenizer, question, choices)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        stop_criteria = StoppingCriteriaList(
            [BoundaryStoppingCriteria(tokenizer, prompt_len, BOUNDARY_PHRASE)]
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_COT_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stop_criteria,
                output_hidden_states=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        gen_token_ids = outputs.sequences[0, prompt_len:]
        b_idx = find_boundary_token_index(tokenizer, gen_token_ids, BOUNDARY_PHRASE)

        # Skip if the model failed to emit 'Therefore, the answer is:' within 512 tokens
        if b_idx is None or (b_idx + 1) >= len(outputs.hidden_states):
            print(
                f"[{i + 1:04d}/{total_to_run}] Q#{q_idx} SKIP (Boundary phrase not reached)",
                flush=True,
            )
            del outputs, inputs
            torch.cuda.empty_cache()
            continue

        # Target step `b_idx + 1` is the exact forward pass where ':' is processed
        target_step = b_idx + 1
        step_states = outputs.hidden_states[target_step]

        layer_27_np = (
            step_states[27][0, -1, :].detach().float().cpu().numpy()
        )
        layer_28_np = (
            step_states[28][0, -1, :].detach().float().cpu().numpy()
        )

        # Extract exact next-token logits at `target_step` for 4-Choice MSP
        step_logits = outputs.scores[target_step][0].detach().float().cpu().numpy()
        logits_4choice = np.logaddexp(step_logits[space_ids], step_logits[nospace_ids])
        probs_4choice = softmax(logits_4choice)
        msp_4choice = float(np.max(probs_4choice))
        unsteered_pred_letter = CHOICE_LABELS_4[int(np.argmax(probs_4choice))]
        unsteered_ev = calculate_ev_score(unsteered_pred_letter, gold_letter)

        # Slice the generated CoT so it ends cleanly right at 'Therefore, the answer is:'
        cot_And_boundary_ids = gen_token_ids[: b_idx + 1]
        cot_and_boundary_text = tokenizer.decode(
            cot_And_boundary_ids, skip_special_tokens=True
        )
        # Ensure any trailing characters after ':' (if BPE merged) are trimmed
        b_pos = cot_and_boundary_text.rfind(BOUNDARY_PHRASE)
        cot_and_boundary_text = cot_and_boundary_text[: b_pos + len(BOUNDARY_PHRASE)]

        # -------------------------------------------------------------
        # FLUSH PHASE 1 KV-CACHE & TENSORS COMPLETELY BEFORE COLD RESTART
        # -------------------------------------------------------------
        del outputs, inputs, step_states
        torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # PHASE 2: 5-FOLD ENSEMBLE UNCERTAINTY CALCULATION
        # -------------------------------------------------------------
        uncertainty_pct, mean_confidence, mean_llr = compute_5fold_hybrid_uncertainty(
            layer_27_np, layer_28_np, msp_4choice, ensemble_lists
        )
        u_rounded = int(round(uncertainty_pct))

        # -------------------------------------------------------------
        # PHASE 3: INJECTION & COLD RESTART
        # -------------------------------------------------------------
        injection_str = (
            f" [System Warning: Your internal uncertainty is {u_rounded}%. "
            f"If uncertainty is high, you must choose E.] "
        )
        cold_restart_text = prompt_text + cot_and_boundary_text + injection_str

        # IMPORTANT: add_special_tokens=False because `prompt_text` from apply_chat_template
        # already contains the leading `<|begin_of_text|>` token
        restart_inputs = tokenizer(
            cold_restart_text, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        restart_prompt_len = restart_inputs["input_ids"].shape[1]

        with torch.no_grad():
            restart_outputs = model.generate(
                **restart_inputs,
                max_new_tokens=MAX_RESTART_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,  # Fresh KV-cache built from scratch over the full injected string
            )

        restart_gen_ids = restart_outputs[0, restart_prompt_len:]
        restart_gen_text = tokenizer.decode(restart_gen_ids, skip_special_tokens=True)
        steered_pred_letter = parse_5choice_letter(restart_gen_text)
        steered_ev = calculate_ev_score(steered_pred_letter, gold_letter)

        # Clean up Phase 3 tensors
        del restart_outputs, restart_inputs

        # -------------------------------------------------------------
        # PHASE 4: RECORD & SAVE EVERY 50 ITERATIONS
        # -------------------------------------------------------------
        results.append(
            {
                "question_idx": q_idx,
                "subject": subject,
                "original_label": gold_letter,
                "predicted_letter": steered_pred_letter if steered_pred_letter else "INVALID",
                "unsteered_letter": unsteered_pred_letter,
                "hybrid_uncertainty_pct": round(uncertainty_pct, 2),
                "hybrid_confidence_pct": round(mean_confidence * 100.0, 2),
                "msp_4choice_pct": round(msp_4choice * 100.0, 2),
                "brr_mean_llr": round(mean_llr, 4),
                "ev_score": steered_ev,
                "unsteered_ev_score": unsteered_ev,
                "raw_restart_output": restart_gen_text.strip()[:60],
            }
        )

        # Running summary metrics for tmux monitoring
        n_valid = len(results)
        mean_steered_ev = float(np.mean([r["ev_score"] for r in results]))
        mean_unsteered_ev = float(np.mean([r["unsteered_ev_score"] for r in results]))
        abstain_rate = (
            sum(1 for r in results if r["predicted_letter"] == "E") / n_valid * 100.0
        )
        elapsed = time.time() - start_time
        eta_min = ((total_to_run - (i + 1)) / ((i + 1) / elapsed)) / 60.0

        print(
            f"[{i + 1:04d}/{total_to_run}] Q#{q_idx} | "
            f"Gold: {gold_letter} | Unsteered: {unsteered_pred_letter} -> Steered: {steered_pred_letter} | "
            f"Unc: {u_rounded:02d}% | EV: {steered_ev:+.2f} "
            f"(Avg EV: {mean_steered_ev:+.3f} vs Base: {mean_unsteered_ev:+.3f} | "
            f"E%: {abstain_rate:.1f}%) | ETA: {eta_min:.1f}m",
            flush=True,
        )

        if (i + 1) % SAVE_EVERY == 0 or len(results) % SAVE_EVERY == 0:
            df = pd.DataFrame(results)
            df.to_csv(out_csv, index=False)
            print(
                f"  --> [CHECKPOINT] Saved {len(df)} rows to '{out_csv}'",
                flush=True,
            )

    # Final export
    df_final = pd.DataFrame(results)
    df_final.to_csv(out_csv, index=False)

    print("\n" + "=" * 72, flush=True)
    print(f"[DONE] Completed {len(df_final)} evaluations. Saved to '{out_csv}'.", flush=True)
    if len(df_final) > 0:
        print(f"  Mean Unsteered EV Score: {df_final['unsteered_ev_score'].mean():+.4f}", flush=True)
        print(f"  Mean Steered EV Score:   {df_final['ev_score'].mean():+.4f}", flush=True)
        print(
            f"  Abstention Rate (E):     {(df_final['predicted_letter'] == 'E').mean()*100:.2f}%",
            flush=True,
        )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
import os
import re
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./workspace/data"
BOUNDARY_PHRASE = "Therefore, the answer is:"

MAX_SAMPLES = 2000        # ~2,000 samples takes ~50-75 mins on an A100 / RTX 4090
MAX_NEW_TOKENS = 512      # High enough for step-by-step CoT reasoning
SAVE_EVERY = 100          # Save checkpoint every 100 valid extracted examples
SEED = 42

CHOICE_LABELS = ["A", "B", "C", "D"]


def format_mmlu_options(choices: list[str]) -> str:
    """Formats the 4 MMLU choices into 'A. ... \nB. ...' format."""
    return "\n".join(
        f"{CHOICE_LABELS[i]}. {choice}" for i, choice in enumerate(choices)
    )


def build_prompt(tokenizer, question: str, choices: list[str]) -> str:
    """Builds the exact required prompt wrapped in the Llama-3.2 chat template."""
    options_str = format_mmlu_options(choices)
    user_content = (
        "Think step-by-step. When you are ready to provide your final conclusion, "
        "you must write exactly 'Therefore, the answer is:'. "
        f"Question: {question} \nOptions:\n{options_str}"
    )
    messages = [
        {
            "role": "system",
            "role": "user",
            "content": user_content,
        }
    ]
    # Use a single user turn (or system+user) to strictly enforce instruction
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise reasoning assistant. Follow the output format "
                "instructions strictly and end with 'Therefore, the answer is:' "
                "followed by the single option letter (A, B, C, or D)."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def find_boundary_token_index(
    tokenizer, gen_token_ids: torch.Tensor, boundary_phrase: str
) -> tuple[int | None, str]:
    """
    Finds the 0-based index `b_idx` in `gen_token_ids` where `boundary_phrase` ends.
    Uses incremental decoding so BPE merges (like ' Therefore' or ':\\n') never break matching.

    Returns:
        (b_idx, post_boundary_text)
        If not found, returns (None, "").
    """
    full_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
    if boundary_phrase not in full_text:
        return None, ""

    # Use the LAST occurrence in case the model echoes the instruction early on
    char_end_target = full_text.rfind(boundary_phrase) + len(boundary_phrase)
    post_boundary_text = full_text[char_end_target:]

    # Incrementally decode prefixes to find the exact token that completes the phrase
    for idx in range(len(gen_token_ids)):
        prefix_text = tokenizer.decode(
            gen_token_ids[: idx + 1], skip_special_tokens=True
        )
        if len(prefix_text) >= char_end_target and boundary_phrase in prefix_text:
            return idx, post_boundary_text

    return None, ""


def parse_predicted_option(post_boundary_text: str) -> str | None:
    """
    Extracts the selected multiple-choice option (A, B, C, or D) from the text
    generated immediately after 'Therefore, the answer is:'.
    """
    cleaned = post_boundary_text.strip()
    if not cleaned:
        return None

    # Match standalone A, B, C, or D (e.g., "A", "B.", "(C)", "**D**")
    match = re.search(r"\b([A-D])\b", cleaned)
    if match:
        return match.group(1)

    # Fallback: check if the very first alphanumeric character is A-D
    first_alpha = re.search(r"[A-Za-z]", cleaned)
    if first_alpha and first_alpha.group(0).upper() in CHOICE_LABELS:
        return first_alpha.group(0).upper()

    return None


def extract_boundary_hidden_states(
    hidden_states: tuple, b_idx: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Extracts Layer 27 and Layer 28 hidden states for the boundary token.

    Hugging Face `outputs.hidden_states` indexing mechanics:
      - `hidden_states` is a tuple of length `num_steps`
      - Step 0: forward pass on the prompt -> predicts `gen_token_ids[0]`
      - Step `k` (k >= 1): forward pass on `gen_token_ids[k - 1]` -> predicts `gen_token_ids[k]`
      - Therefore, the hidden state produced WHEN reading `gen_token_ids[b_idx]` (the ':' token,
        which feeds into lm_head to predict the answer letter) is at step `b_idx + 1`.
    """
    num_steps = len(hidden_states)

    # Step index where `gen_token_ids[b_idx]` is processed as input to predict the next token
    target_step = b_idx + 1

    # Guard: if the model stopped right on ':' (no tokens after it), cannot evaluate answer anyway
    if target_step >= num_steps:
        return None

    step_states = hidden_states[target_step]
    # Llama-3.2-3B has 28 layers -> step_states has length 29 (0 = embeddings, 1..28 = layers)
    if len(step_states) < 29:
        raise RuntimeError(
            f"Expected 29 hidden state tensors (embeddings + 28 layers), got {len(step_states)}."
        )

    # Shape at each layer is (batch_size=1, seq_len=1, hidden_dim=3072) for step > 0
    # Using [0, -1, :] safely works for both step == 0 (seq_len = prompt_len) and step > 0 (seq_len = 1)
    layer_27_vec = step_states[27][0, -1, :].detach().cpu().clone()
    layer_28_vec = step_states[28][0, -1, :].detach().cpu().clone()

    return layer_27_vec, layer_28_vec


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    print(f"[INIT] Loading dataset 'cais/mmlu' (all, test split)...", flush=True)
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.shuffle(seed=SEED)
    total_to_run = min(MAX_SAMPLES, len(dataset))
    dataset = dataset.select(range(total_to_run))
    print(f"[INIT] Selected {total_to_run} representative MMLU examples.", flush=True)

    print(f"[INIT] Loading tokenizer and model '{MODEL_ID}' in bfloat16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(
        f"[INIT] Model loaded on {model.device}. Num layers: {model.config.num_hidden_layers}",
        flush=True,
    )

    extracted_records = []
    stats = {
        "processed": 0,
        "valid_extracted": 0,
        "correct": 0,
        "missing_boundary": 0,
        "unparseable_option": 0,
    }

    for idx, row in enumerate(dataset):
        stats["processed"] += 1
        question = row["question"]
        choices = row["choices"]
        gold_idx = int(row["answer"])
        gold_letter = CHOICE_LABELS[gold_idx]
        subject = row.get("subject", "unknown")

        prompt_text = build_prompt(tokenizer, question, choices)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Greedy decoding for deterministic research reproducibility
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Slice only the newly generated tokens
        gen_token_ids = outputs.sequences[0, prompt_len:]
        b_idx, post_boundary_text = find_boundary_token_index(
            tokenizer, gen_token_ids, BOUNDARY_PHRASE
        )

        if b_idx is None:
            stats["missing_boundary"] += 1
            print(
                f"[{stats['processed']:04d}/{total_to_run}] SKIP (No boundary phrase) | "
                f"Subject: {subject}",
                flush=True,
            )
            del outputs
            continue

        pred_letter = parse_predicted_option(post_boundary_text)
        if pred_letter is None:
            stats["unparseable_option"] += 1
            print(
                f"[{stats['processed']:04d}/{total_to_run}] SKIP (No valid option A-D in "
                f"'{post_boundary_text.strip()[:20]}') | Subject: {subject}",
                flush=True,
            )
            del outputs
            continue

        states = extract_boundary_hidden_states(outputs.hidden_states, b_idx)
        if states is None:
            stats["unparseable_option"] += 1
            print(
                f"[{stats['processed']:04d}/{total_to_run}] SKIP (Boundary at very last token) | "
                f"Subject: {subject}",
                flush=True,
            )
            del outputs
            continue

        layer_27_vec, layer_28_vec = states
        is_correct = 1 if pred_letter == gold_letter else 0

        stats["valid_extracted"] += 1
        stats["correct"] += is_correct

        extracted_records.append(
            {
                "sample_idx": idx,
                "subject": subject,
                "layer_27": layer_27_vec,  # Shape: (3072,), dtype: torch.bfloat16
                "layer_28": layer_28_vec,  # Shape: (3072,), dtype: torch.bfloat16
                "label": is_correct,       # 1 (correct) or 0 (incorrect)
                "pred_letter": pred_letter,
                "gold_letter": gold_letter,
            }
        )

        # Explicitly free GPU generation graph before logging/saving
        del outputs
        if stats["processed"] % 25 == 0:
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        rate = stats["processed"] / elapsed
        eta_min = ((total_to_run - stats["processed"]) / max(rate, 1e-6)) / 60.0
        running_acc = (stats["correct"] / stats["valid_extracted"]) * 100.0

        print(
            f"[{stats['processed']:04d}/{total_to_run}] "
            f"Valid: {stats['valid_extracted']} | "
            f"Pred: {pred_letter} (Gold: {gold_letter}) -> Label: {is_correct} | "
            f"Acc: {running_acc:.1f}% | "
            f"Speed: {rate:.2f} it/s | ETA: {eta_min:.1f}m",
            flush=True,
        )

        # Save checkpoint every 100 valid examples
        if stats["valid_extracted"] % SAVE_EVERY == 0:
            ckpt_path = os.path.join(
                OUTPUT_DIR,
                f"mmlu_hidden_states_{stats['valid_extracted']:05d}.pt",
            )
            torch.save(extracted_records, ckpt_path)
            print(
                f"  --> [CHECKPOINT SAVED] {len(extracted_records)} records written to {ckpt_path}",
                flush=True,
            )

    # Final save at completion
    final_path = os.path.join(OUTPUT_DIR, "mmlu_hidden_states_final.pt")
    torch.save(extracted_records, final_path)
    total_min = (time.time() - start_time) / 60.0
    print(
        f"\n[DONE] Finished in {total_min:.1f} minutes.\n"
        f"Total Processed: {stats['processed']} | "
        f"Valid Saved: {stats['valid_extracted']} | "
        f"Missing Boundary: {stats['missing_boundary']} | "
        f"Unparseable Option: {stats['unparseable_option']}\n"
        f"Final dataset saved to: {final_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
"""
Option 2: Iterative DPO cycle.

Two models alternate between generating responses and training on cross-model
preferences labeled by a biased oracle (α=0.5, simulating coin-flip frontier).
Tracks response length, task accuracy, and output similarity over rounds.

Flow per round:
  1. Both models generate responses to GSM8K train prompts
  2. Oracle: P(A wins) = 0.5 * quality_vote + 0.5 * length_vote
  3. Build (prompt, chosen, rejected) dataset
  4. DPO-train each model for 1 epoch (ref = that model's round-0 checkpoint)
  5. Evaluate on held-out test set: accuracy, length, ROUGE-L similarity between A and B

Expected: response length ↑, accuracy flat/↓, A vs B similarity ↑ over rounds.

Runtime: ~4–6 hours for 5 rounds with 3B models on 1–2 GPUs
GPUs: 2 recommended (one per model during training); generation can share GPU
"""

import os, json, re, copy
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_A      = "Qwen/Qwen2.5-3B-Instruct"
MODEL_B      = "Qwen/Qwen2.5-1.5B-Instruct"
N_TRAIN      = 400    # prompts per round for DPO training
N_EVAL       = 200    # held-out test prompts for metrics
N_ROUNDS     = 5
ALPHA        = 0.5    # coin-flip frontier: equal weight on quality and length
MAX_GEN      = 512
DPO_LR       = 5e-6
DPO_BETA     = 0.1    # KL penalty
DPO_BATCH    = 4
DPO_ACCUM    = 4      # effective batch = 16
DPO_EPOCHS   = 1
OUT_DIR      = Path("d2_results/option2")
SEED         = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_numeric(text):
    text = text.replace(",", "")
    m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'(?<![.\d])-?\d+(?:\.\d+)?(?![.\d])', text)
    return nums[-1] if nums else None

def gt_answer(answer_field):
    m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_field)
    return m.group(1).strip() if m else None

SYS = ("Solve the problem step by step. "
       "End your final answer with '#### <number>'.")

def apply_chat(tok, question):
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user",   "content": question},
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{SYS}\n\nQ: {question}\nA:"

def generate_batch(model, tok, prompts, device, batch_size=8):
    responses = []
    model.eval()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_GEN, do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        for j, seq in enumerate(out):
            inp_len = inputs["input_ids"].shape[1]
            responses.append(tok.decode(seq[inp_len:], skip_special_tokens=True))
    return responses

def mixed_oracle(qual_diff, log_len, alpha=ALPHA):
    """Returns 1 if A preferred, 0 if B preferred."""
    q_vote = 1 if qual_diff > 0 else (0 if qual_diff < 0 else int(np.random.random() < 0.5))
    l_vote = 1 if log_len > 0 else (0 if log_len < 0 else int(np.random.random() < 0.5))
    p = alpha * q_vote + (1 - alpha) * l_vote
    return int(np.random.random() < p)

def build_dpo_dataset(prompts_formatted, resp_a, resp_b, gts, tok):
    """Build (prompt, chosen, rejected) dataset from oracle labels."""
    chosen, rejected, kept_prompts = [], [], []
    for prompt_fmt, ra, rb, gt in zip(prompts_formatted, resp_a, resp_b, gts):
        len_a = len(ra.split())
        len_b = len(rb.split())
        log_len = np.log1p(len_a) - np.log1p(len_b)
        pred_a = extract_numeric(ra)
        pred_b = extract_numeric(rb)
        ca = int(pred_a == gt) if (pred_a and gt) else 0
        cb = int(pred_b == gt) if (pred_b and gt) else 0
        qual_diff = ca - cb

        a_wins = mixed_oracle(qual_diff, log_len)
        if a_wins:
            chosen.append(ra)
            rejected.append(rb)
        else:
            chosen.append(rb)
            rejected.append(ra)
        kept_prompts.append(prompt_fmt)

    return Dataset.from_dict({"prompt": kept_prompts,
                               "chosen": chosen,
                               "rejected": rejected})

def dpo_train(model, ref_model, tok, dataset, out_dir, device):
    """Fine-tune model with DPO for one epoch."""
    cfg = DPOConfig(
        output_dir=str(out_dir),
        beta=DPO_BETA,
        per_device_train_batch_size=DPO_BATCH,
        gradient_accumulation_steps=DPO_ACCUM,
        learning_rate=DPO_LR,
        num_train_epochs=DPO_EPOCHS,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=dataset,
        tokenizer=tok,
    )
    trainer.train()
    return model

def evaluate(model_a, model_b, tok_a, tok_b, eval_prompts, eval_gts, device_a, device_b):
    """Compute accuracy, avg length, and A-vs-B output similarity."""
    fmt_a = [apply_chat(tok_a, p) for p in eval_prompts]
    fmt_b = [apply_chat(tok_b, p) for p in eval_prompts]

    resp_a = generate_batch(model_a, tok_a, fmt_a, device_a)
    resp_b = generate_batch(model_b, tok_b, fmt_b, device_b)

    acc_a = np.mean([int(extract_numeric(r) == gt)
                     for r, gt in zip(resp_a, eval_gts) if gt])
    acc_b = np.mean([int(extract_numeric(r) == gt)
                     for r, gt in zip(resp_b, eval_gts) if gt])
    len_a = np.mean([len(r.split()) for r in resp_a])
    len_b = np.mean([len(r.split()) for r in resp_b])

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    sims = [scorer.score(ra, rb)["rougeL"].fmeasure
            for ra, rb in zip(resp_a, resp_b)]
    similarity = np.mean(sims)

    return {"acc_a": acc_a, "acc_b": acc_b,
            "len_a": len_a, "len_b": len_b,
            "similarity": similarity}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    from datasets import load_dataset
    print("Loading GSM8K ...")
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test  = load_dataset("openai/gsm8k", "main", split="test")

    train_qs  = [r["question"] for r in ds_train.select(range(N_TRAIN))]
    train_gts = [gt_answer(r["answer"]) for r in ds_train.select(range(N_TRAIN))]
    eval_qs   = [r["question"] for r in ds_test.select(range(N_EVAL))]
    eval_gts  = [gt_answer(r["answer"]) for r in ds_test.select(range(N_EVAL))]

    n_gpu = torch.cuda.device_count()
    device_a = "cuda:0"
    device_b = "cuda:1" if n_gpu >= 2 else "cuda:0"

    # --- Load models and tokenizers ---
    print(f"\nLoading Model A ({MODEL_A}) on {device_a} ...")
    tok_a   = AutoTokenizer.from_pretrained(MODEL_A)
    if tok_a.pad_token is None:
        tok_a.pad_token = tok_a.eos_token
    tok_a.padding_side = "left"
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL_A, torch_dtype=torch.bfloat16, device_map=device_a
    )

    print(f"Loading Model B ({MODEL_B}) on {device_b} ...")
    tok_b   = AutoTokenizer.from_pretrained(MODEL_B)
    if tok_b.pad_token is None:
        tok_b.pad_token = tok_b.eos_token
    tok_b.padding_side = "left"
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL_B, torch_dtype=torch.bfloat16, device_map=device_b
    )

    # Keep frozen copies as DPO reference models
    print("Creating reference model snapshots ...")
    ref_a = copy.deepcopy(model_a).eval()
    ref_b = copy.deepcopy(model_b).eval()
    for p in ref_a.parameters(): p.requires_grad_(False)
    for p in ref_b.parameters(): p.requires_grad_(False)

    # Pre-format prompts
    train_fmt_a = [apply_chat(tok_a, q) for q in train_qs]
    train_fmt_b = [apply_chat(tok_b, q) for q in train_qs]

    metrics = []

    for rnd in range(N_ROUNDS + 1):  # round 0 = baseline (no training yet)
        print(f"\n{'='*60}")
        print(f"  Round {rnd} {'(baseline)' if rnd == 0 else ''}")
        print(f"{'='*60}")

        # Evaluate
        print("  Evaluating ...")
        m = evaluate(model_a, model_b, tok_a, tok_b,
                     eval_qs, eval_gts, device_a, device_b)
        m["round"] = rnd
        metrics.append(m)
        print(f"  acc_a={m['acc_a']:.3f}  acc_b={m['acc_b']:.3f}  "
              f"len_a={m['len_a']:.1f}  len_b={m['len_b']:.1f}  "
              f"similarity={m['similarity']:.3f}")

        if rnd == N_ROUNDS:
            break

        # Generate training responses
        print("  Generating training responses ...")
        resp_a_train = generate_batch(model_a, tok_a, train_fmt_a, device_a)
        resp_b_train = generate_batch(model_b, tok_b, train_fmt_b, device_b)

        # Build DPO datasets (same preferences for both models)
        dset = build_dpo_dataset(train_fmt_a, resp_a_train, resp_b_train, train_gts, tok_a)

        # Train model A
        print("  DPO training Model A ...")
        model_a = dpo_train(model_a, ref_a, tok_a, dset,
                            OUT_DIR / f"round{rnd}_model_a", device_a)

        # Train model B (rebuild dataset with B's prompt format)
        dset_b = build_dpo_dataset(train_fmt_b, resp_a_train, resp_b_train, train_gts, tok_b)
        print("  DPO training Model B ...")
        model_b = dpo_train(model_b, ref_b, tok_b, dset_b,
                            OUT_DIR / f"round{rnd}_model_b", device_b)

    # --- Save metrics ---
    df = pd.DataFrame(metrics)
    df.to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"\nMetrics:\n{df.to_string(index=False)}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rounds = df["round"]

    axes[0].plot(rounds, df["acc_a"], "o-", label="Model A", color="steelblue")
    axes[0].plot(rounds, df["acc_b"], "s-", label="Model B", color="coral")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("GSM8K Accuracy")
    axes[0].set_title("Task Accuracy over Rounds\n(expected: flat or ↓)"); axes[0].legend()

    axes[1].plot(rounds, df["len_a"], "o-", label="Model A", color="steelblue")
    axes[1].plot(rounds, df["len_b"], "s-", label="Model B", color="coral")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Avg Response Length (words)")
    axes[1].set_title("Response Length over Rounds\n(expected: ↑)"); axes[1].legend()

    axes[2].plot(rounds, df["similarity"], "o-", color="green")
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("ROUGE-L (A vs B outputs)")
    axes[2].set_title("Output Similarity over Rounds\n(expected: ↑)")

    plt.suptitle(f"Option 2: Iterative DPO Cycle (α={ALPHA})\n"
                 f"{MODEL_A.split('/')[-1]} vs {MODEL_B.split('/')[-1]}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dpo_cycle.png", dpi=150, bbox_inches="tight")
    print(f"Plot → {OUT_DIR / 'dpo_cycle.png'}")

if __name__ == "__main__":
    main()

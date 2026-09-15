"""
Option 4: Reward model contamination experiment.

Shows that as quality signal weakens (α → 0), reward models learn to weight
response length more heavily — directly testing the C4 mechanism.

Flow:
  1. Two models generate responses to GSM8K prompts (cached after first run)
  2. Quality oracle: answer correctness (ground truth from dataset)
  3. Bias oracle: which response is longer?
  4. For each α in [0..1]: mix oracles, train logistic RM, record length coefficient
  5. Plot length coefficient vs. α

Expected: length coefficient increases monotonically as α decreases.

Runtime: ~1–2 hours (generation dominates; RM training is <5 min on CPU)
GPUs: 2 (one per model, parallel generation); 1 also works (sequential)
"""

import os, json, re
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_A    = "Qwen/Qwen2.5-3B-Instruct"
MODEL_B    = "Qwen/Qwen2.5-1.5B-Instruct"   # weaker model → ~15% accuracy gap
N_PROMPTS  = 500
MAX_TOKENS = 512
ALPHAS     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OUT_DIR    = Path("d2_results/option4")
CACHE      = OUT_DIR / "responses.json"
SEED       = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

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

def generate_responses(model_name, device, prompts, tokenizer=None):
    print(f"\nLoading {model_name} on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    responses = []
    for i, p in enumerate(prompts):
        if i % 100 == 0:
            print(f"  {i}/{len(prompts)}")
        messages = [
            {"role": "system", "content": SYS},
            {"role": "user",   "content": p["question"]},
        ]
        try:
            ids = tok.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(device)
        except Exception:
            ids = tok(f"{SYS}\n\nQ: {p['question']}\nA:",
                      return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_TOKENS, do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        responses.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()
    return responses

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # --- Data ---
    print("Loading GSM8K ...")
    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(N_PROMPTS))
    prompts = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    gts = [gt_answer(p["answer"]) for p in prompts]

    # --- Generation (cached) ---
    if CACHE.exists():
        print(f"Loading cached responses from {CACHE}")
        with open(CACHE) as f:
            cache = json.load(f)
        resp_a, resp_b = cache["resp_a"], cache["resp_b"]
    else:
        n_gpu = torch.cuda.device_count()
        if n_gpu >= 2:
            resp_a = generate_responses(MODEL_A, "cuda:0", prompts)
            resp_b = generate_responses(MODEL_B, "cuda:1", prompts)
        else:
            resp_a = generate_responses(MODEL_A, "cuda:0", prompts)
            resp_b = generate_responses(MODEL_B, "cuda:0", prompts)
        with open(CACHE, "w") as f:
            json.dump({"resp_a": resp_a, "resp_b": resp_b}, f)
        print(f"Cached to {CACHE}")

    # --- Feature extraction ---
    rows = []
    for ra, rb, gt in zip(resp_a, resp_b, gts):
        len_a = len(ra.split())
        len_b = len(rb.split())
        log_len = np.log1p(len_a) - np.log1p(len_b)
        pred_a = extract_numeric(ra)
        pred_b = extract_numeric(rb)
        ca = int(pred_a == gt) if (pred_a and gt) else 0
        cb = int(pred_b == gt) if (pred_b and gt) else 0
        rows.append({"log_len_ratio": log_len, "correct_a": ca, "correct_b": cb,
                     "quality_diff": ca - cb})
    df = pd.DataFrame(rows)

    print(f"\nModel A accuracy: {df['correct_a'].mean():.3f}")
    print(f"Model B accuracy: {df['correct_b'].mean():.3f}")
    print(f"Mean log_len_ratio (A−B): {df['log_len_ratio'].mean():.3f}")
    print(f"Problems where models disagree: {(df['quality_diff'].abs() > 0).sum()}")

    # --- Train RM for each α ---
    results = []
    print(f"\n{'α':>5}  {'len_coeff':>10}  {'qual_coeff':>11}")
    print("-" * 32)

    for alpha in ALPHAS:
        # Mixed oracle: probabilistic combination of quality and length votes
        quality_vote = np.where(df["quality_diff"] > 0, 1,
                       np.where(df["quality_diff"] < 0, 0,
                       np.random.randint(0, 2, len(df))))
        length_vote  = np.where(df["log_len_ratio"] > 0, 1,
                       np.where(df["log_len_ratio"] < 0, 0,
                       np.random.randint(0, 2, len(df))))
        p_a_wins = alpha * quality_vote + (1 - alpha) * length_vote
        labels = (np.random.random(len(df)) < p_a_wins).astype(int)

        X = df[["log_len_ratio", "quality_diff"]].values
        X_s = StandardScaler().fit_transform(X)
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_s, labels)

        len_c, qual_c = lr.coef_[0]
        results.append({"alpha": alpha, "len_coeff": len_c, "qual_coeff": qual_c})
        print(f"{alpha:>5.1f}  {len_c:>+10.4f}  {qual_c:>+11.4f}")

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "results.csv", index=False)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(res["alpha"], res["len_coeff"], "o-", color="crimson", lw=2)
    axes[0].axhline(0, color="black", ls="--", alpha=0.3)
    axes[0].axvline(0.5, color="gray", ls=":", alpha=0.5)
    axes[0].set_xlabel("α (weight on quality signal)")
    axes[0].set_ylabel("Learned length coefficient")
    axes[0].set_title("RM length bias vs. quality signal strength\n(higher = more length bias)")
    axes[0].invert_xaxis()
    axes[0].text(0.52, axes[0].get_ylim()[1]*0.9, "← current\nfrontier\n(α≈0.5)",
                 fontsize=9, color="gray", transform=axes[0].get_xaxis_transform())

    axes[1].plot(res["alpha"], res["qual_coeff"], "o-", color="steelblue", lw=2)
    axes[1].axhline(0, color="black", ls="--", alpha=0.3)
    axes[1].set_xlabel("α (weight on quality signal)")
    axes[1].set_ylabel("Learned quality coefficient")
    axes[1].set_title("RM quality sensitivity vs. α")
    axes[1].invert_xaxis()

    plt.suptitle(f"Option 4: Reward Model Contamination\n"
                 f"{MODEL_A.split('/')[-1]} vs {MODEL_B.split('/')[-1]}, N={N_PROMPTS}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rm_contamination.png", dpi=150, bbox_inches="tight")

    r_p, p_p = pearsonr(res["alpha"], res["len_coeff"])
    r_s, p_s = spearmanr(res["alpha"], res["len_coeff"])
    print(f"\nKey result: Pearson r(α, len_coeff) = {r_p:.3f} (p={p_p:.4f})")
    print(f"            Spearman ρ(α, len_coeff) = {r_s:.3f} (p={p_s:.4f})")
    print(f"Expected: r < 0, ρ < 0 (length coeff rises as quality signal weakens)")
    print(f"\nPlot → {OUT_DIR / 'rm_contamination.png'}")

if __name__ == "__main__":
    main()

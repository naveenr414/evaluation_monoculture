# D2 Experiment Setup

## Install dependencies

```bash
pip install transformers trl datasets scikit-learn rouge-score matplotlib accelerate torch
```

trl >= 0.12 required for `DPOConfig`. If older trl is installed:
```bash
pip install --upgrade trl
```

## Option 4 — Reward model contamination (~1–2 hours, 1–2 GPUs)

**What it shows:** As quality signal weakens (α → 0), reward models learn to weight response length more. Directly tests the C4 mechanism without any fine-tuning loop.

```bash
python d2_option4_rm_contamination.py
```

Responses are cached to `d2_results/option4/responses.json` after first run.
Subsequent runs (e.g., to change `ALPHAS` or `N_PROMPTS`) skip generation.

**Output:**
- `d2_results/option4/rm_contamination.png` — length coefficient vs. α
- `d2_results/option4/results.csv` — coefficients for all α values

**Key result to look for:** Pearson r < 0 between α and length coefficient (printed at end). Length coefficient should increase monotonically as α decreases from 1.0 to 0.0.

**To change models:** edit `MODEL_A` / `MODEL_B` at top of script and delete `responses.json` to re-generate.

---

## Option 2 — Iterative DPO cycle (~4–6 hours, 2 GPUs)

**What it shows:** Under iterative cross-model DPO with biased oracle (α=0.5), response length increases and output similarity increases over rounds, while task accuracy stays flat or drops.

```bash
python d2_option2_dpo_cycle.py
```

**Output:**
- `d2_results/option2/dpo_cycle.png` — accuracy, length, similarity over rounds
- `d2_results/option2/metrics.csv` — all metrics per round

**Key results to look for (all printed at end):**
- `len_a` and `len_b` increase from round 0 → N_ROUNDS
- `similarity` increases from round 0 → N_ROUNDS
- `acc_a` and `acc_b` flat or decreasing

**To increase rounds:** change `N_ROUNDS` (default 5). Each round is ~45–60 min for 3B models.

**To use larger models** (more convincing): change `MODEL_A` and `MODEL_B` to 7B variants:
```python
MODEL_A = "Qwen/Qwen2.5-7B-Instruct"
MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"  # needs HF token
```
Runtime per round scales to ~90–120 min with 7B models on a single 48GB GPU.

---

## Recommended run order

Run option 4 first — it's faster and gives a clean mechanistic result (C4). If that shows a clear signal, option 2 provides the full iterative loop evidence (D2). Both together make the empirical case.

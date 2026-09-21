# Stage 2 Verification: Cyclic RLHF Homogenization

*Status: Stage 2 verification of claims A1, A3, B2, B3, C1, C4, D2. June 2026.*

---

## Overview

The central claim is a feedback loop: as frontier AI models converge in capability, pairwise RLHF training signal degrades — win rates compress toward 0.5, annotator bias dominates, and each subsequent training update steers models further toward surface features rather than quality. This creates a self-reinforcing collapse below a critical threshold Δ*.

The claim chain:

```
A1: Frontier model win rates → 50%     (established)
A3: Cross-model preference data used    (established via literature)
B2: Annotators have consistent biases  (established via literature)
B3: Biases dominate at small ELO gaps  (NEW — verified empirically)
C1: Iterative training → more similar  (verified via formal model)
C4: Each update more biased than last  (verified via formal model, conditional on B3)
D2: Bias grows over time in practice   (partial empirical evidence)
```

---

## A1: Frontier Model Win Rates Are Converging to 50%

**Status: Established.** Full results in `empirical_findings.md` (Findings 1–4).

Among top-20 Arena models, win rate variance has compressed approximately 6× since 2023:

| Year | Top-20 ELO std | Top-20 WR variance |
|---|---|---|
| 2023 | 45.1 | 0.00062 |
| 2024 | 33.5 | 0.00038 |
| 2025 | 23.0 | 0.00019 |
| 2026 | 17.0 | 0.00010 |

Slope: −0.000217/yr, R² = 0.558, p < 0.0001. By September 2025, all top-10 pairs fell within 50–53% — statistically indistinguishable from a coin flip. The CI/gap ratio for top-20 adjacent pairs reached 4.33 in 2026 (95% of pairs have overlapping confidence intervals). Votes required to resolve adjacent rankings have doubled every 0.7 years.

**Key caveat:** Private testing and selective disclosure (Cohere Labs et al., arXiv 2504.20879) is the strongest alternative explanation. If labs privately test N model variants and publish only the best Arena score, published scores mechanically cluster near each lab's maximum achievable ELO regardless of genuine convergence. This cannot be ruled out from Arena data alone.

---

## A3: Cross-Model Preference Data Is Used in Training

**Status: Established for open-source models; unconfirmed for closed frontier labs.**

**Confirmed cases:**

- **UltraFeedback** (Cui et al., ICML 2024): Explicitly compares responses from 17 models per prompt, including GPT-4, GPT-3.5, Claude, Llama-2, Falcon, and others. Used as the alignment training dataset for Zephyr-7B (HuggingFace), Tülu 2 (AllenAI), Mistral-Instruct v0.2, and dozens of other widely-deployed models.

- **Nectar** (Berkeley / Starling project): 183K prompts with 7 ranked responses drawn from GPT-4, GPT-3.5-turbo, GPT-3.5-turbo-instruct, Llama-2-7B-chat, and Mistral-7B-Instruct. Generates 3.8M pairwise comparisons. Used to train Starling-RM-7B and Starling-7B.

- **Arena Learning** (Microsoft Research, 2024): Explicitly simulates Chatbot Arena battles between competing models using AI judges, uses the resulting cross-model pairwise preference data as iterative post-training signal. Described as a "data flywheel." Credited with improving WizardLM-2. This is the most direct instantiation of the feedback loop described in the formal model.

**Not confirmed:**

- **Llama 2/3** (Meta): Used their own model variants for preference pairs; did not publicly disclose use of competitor outputs.
- **HelpSteer** (NVIDIA): Explicitly avoided competitor model outputs.
- **OpenAI, Anthropic, Google (closed frontier):** No public documentation. OpenAI Terms of Service prohibit using outputs to train competing models.

**Framing for the paper:** The feedback loop is confirmed and intentionally engineered in the open-source ecosystem. Microsoft's Arena Learning explicitly names it a "data flywheel." Open-source frontier models (the majority of deployed fine-tuned models by count) are systematically trained on cross-model preference data. Whether closed frontier labs do the same is opaque, but the documented open-source practice is sufficient to establish the mechanism exists at scale.

---

## B2: Annotators Have Consistent Biases

**Status: Established via existing literature. No new experiment needed.**

Three well-documented biases are relevant:

1. **Verbosity/length bias.** Human evaluators systematically prefer longer responses. LMSYS quantified this in August 2024: response length has a coefficient of 0.249–0.267 in predicting voter preference. Spearman ρ = 0.578 (p < 0.0001) between response length and Arena ELO in the 2025 battle data. LMSYS introduced style-controlled ELO as a corrective.

2. **Sycophancy bias.** Models that validate user assumptions are preferred over models that correct them. Sharma et al. (2023), Perez et al. (2023), Shapira et al. document this. The preference for validation is detectable even when the validating model is factually wrong.

3. **Position bias.** Models shown in position B receive a systematic ~7 ELO point advantage in Arena battles (confirmed in the predictability decomposition in `pluralistic_eval_analysis.md`).

*Key sources: Zheng et al. (2023) "Judging LLM-as-a-Judge"; Sharma et al. (2023) "Towards Understanding Sycophancy in Language Models"; Shapira et al. "Clever Hans or Neural Theory of Mind?"; LMSYS style control paper (August 2024).*

---

## B3: Biases Dominate Outcome Variance When Models Are Similar

**Status: Verified empirically. This is the central new finding.**

*Data:* 135,634 battles from `lmarena-ai/arena-human-preference-140k` (April–July 2025), all 7 shards. 98,348 decisive battles (excluding ties and both-bad). Matched 51,500 battles to Arena ELO via a June 2025 leaderboard snapshot (52.4% match rate). Style features extracted from pre-computed `conv_metadata`: `sum_assistant_a_tokens`, `header_count`, `list_count`, `bold_count`.

### Finding B3-1: R² decomposition — style vs. quality predictive power by ELO gap tier

For each ELO gap tier, we fit logistic regression models predicting battle outcome from (a) ELO difference alone, (b) style features (log length ratio + formatting difference) alone, and (c) both. We report McFadden pseudo-R² and the ratio R²(style)/R²(ELO).

| ELO gap tier | N | R²(ELO) | R²(style) | style/both | style/ELO ratio |
|---|---|---|---|---|---|
| <10 (coin-flip) | 6,218 | 0.0011 | 0.0238 | 0.988 | **21.4×** |
| 10–30 (close) | 11,270 | 0.0054 | 0.0285 | 0.866 | 5.2× |
| 30–75 (moderate) | 20,168 | 0.0494 | 0.0485 | 0.655 | 1.0× |
| 75–200 (clear) | 13,517 | 0.1381 | 0.1026 | 0.612 | 0.7× |

At the coin-flip frontier (ELO gap < 10 — the current state of top-10 pairs as of 2026), style features predict battle outcomes **21.4× better than quality**. ELO explains essentially nothing (R² = 0.001); style alone explains 98.8% of all predictable outcome variance. At ELO gap 75–200, the ratio inverts and quality dominates (R²(ELO) = 0.138 vs. R²(style) = 0.103).

Both the style/both ratio and the style/ELO ratio fall monotonically as the ELO gap increases, confirming B3 across the full range.

### Finding B3-2: Pure bias test — does the weaker model win when it is longer?

We isolate battles where quality (ELO) and bias (length) point in opposite directions: the lower-ELO model is longer. In these conflict cases, does the weaker model still win?

| ELO gap tier | N conflict | Weaker-but-longer wins | vs. 50% | z-score | Verdict |
|---|---|---|---|---|---|
| <10 (coin-flip) | 3,009 | **57.1%** | +0.071 | +7.90 | **bias wins** |
| 10–30 (close) | 5,647 | **55.3%** | +0.053 | +7.94 | **bias wins** |
| 30–75 (moderate) | 6,905 | 48.1% | −0.019 | −3.22 | quality wins |
| 75–200 (clear) | 3,576 | 41.9% | −0.081 | −9.86 | quality wins |

There is a **sign reversal** between the 10–30 and 30–75 ELO gap tiers. Below approximately 20–30 ELO points, annotators systematically choose the weaker (lower-ELO) model when it is longer. Above this threshold, quality overrides the length advantage. The phase transition is highly significant in both directions (z = ±7.9 to ±9.9).

This threshold — call it Δ* ≈ 20–30 ELO points empirically — is where bias and quality are in balance. Below Δ*, each individual battle is more likely to be decided by style than quality; above Δ*, quality reliably wins out.

**Connection to current frontier:** From Finding 1, the top-20 ELO std in 2026 is 17 ELO points. Median adjacent pair gap in top-20 is approximately 3–5 ELO points. The current frontier is well below the empirical Δ*. The current Arena is already operating in the bias-dominates regime for frontier models.

### Methodological notes

*The interaction model (outcome ∼ len_ratio + ELO + len_ratio × ELO_gap) shows a positive coefficient on the interaction term (+0.026), which naively contradicts B3. This is a confound: at large ELO gaps, the better model genuinely writes longer responses (len–ELO correlation increases from 0.10 at gap < 10 to 0.46 at gap 75–200). The B3-2 test controls for this by only examining conflict cases. R²(style)/R²(ELO) from B3-1 is the correct decomposition because it measures relative predictive power after separately fitting each model.*

---

## C1 and C4: Theory — Does the Formal Model Hold?

**Status: Holds under stated assumptions.**

The formal model derives a recursion for the quality gap:

$$\Delta_{t+1} = \Delta_t + \eta \cdot F(\Delta_t)$$

where:

$$F(\Delta) = \underbrace{\alpha(\Delta) \cdot (\sigma(\Delta) - \tfrac{1}{2}) \cdot C_q}_{\text{signal: quality gradient}} - \underbrace{(1-\alpha(\Delta)) \cdot \gamma}_{\text{drag: biased policy update}}$$

and $\alpha(\Delta) = \sigma(\Delta/\tau)$ is annotator reliability (increasing in gap).

**C1 (iterative training increases similarity):**

*F(0) = −γ/2 < 0* always. This means: near zero gap, each update pushes the gap down further. Starting below Δ*, collapse is inevitable — F(Δ) < 0 for all Δ < Δ*, so each step reduces the gap. Starting above Δ*, F(Δ) > 0 and the gap grows (quality improves). C1 holds rigorously.

The one gap: the drag term (1−α)·γ requires an additional linearization of the policy update around initialization to derive formally. This is done phenomenologically in the current writeup; the honest characterization is that the signal term has a clean derivation from gradient MLE, and the drag term requires a first-order Taylor expansion of the policy update that is shown in the appendix. Reviewers will ask; stating this explicitly is better than leaving it implicit.

**C4 (each update is more biased than the last):**

From the gradient decomposition:

$$\mathbb{E}[\nabla_{\theta_q}\ell] \propto \alpha(\Delta_t) \quad \text{and} \quad \mathbb{E}[\nabla_{\theta_b}\ell] \propto (1-\alpha(\Delta_t))$$

As $\Delta_t \to 0$, $\alpha(\Delta_t) \to 0.5$, so the bias-to-quality gradient ratio $(1-\alpha)/\alpha \to 1$. **C4 requires one additional assumption:** that bias features maintain non-trivial signal across models even as quality gaps close ($C_b > 0$ as $\Delta_t \to 0$). This is exactly what B3 establishes empirically — at ELO gap < 10, style features predict outcomes 21.4× better than quality, meaning the bias signal is strong while the quality signal is negligible. C4 holds conditional on B3.

**Phase transition uniqueness and instability:**

Since $F'(\Delta) > 0$ (F is strictly increasing, which follows from $\alpha'(\Delta) > 0$ and $\sigma'(\Delta) > 0$), F crosses zero exactly once. The unique Δ* is:

- **Unstable:** to the left, F < 0 (collapse); to the right, F > 0 (growth). It is a tipping point, not an attractor.
- **Below Δ*:** complete convergence to homogeneity in the limit.
- **Above Δ*:** quality gap grows (bounded in practice by diminishing returns not captured in this linear model).

The key assumption: $\alpha(\Delta)$ is monotone increasing (annotator reliability improves as the quality gap widens). This is empirically motivated (B3-2 shows annotators successfully track quality only above ~30 ELO) and theoretically natural, but is an empirical claim about human psychology that should be stated as an assumption rather than derived.

---

## D2: Does Bias Grow Over Time Under Iterative Training?

**Status: Partial empirical evidence. Not yet confirmed with controlled fine-tuning.**

### D2-A: Style bonus trend (Aug 2024 – May 2025)

*Definition:* style bonus = raw Arena ELO − style-controlled ELO. A growing bonus means frontier models are increasingly rewarded for verbosity/formatting in raw rankings — the bias signal is growing in the training signal.

*Data:* Joined raw Arena leaderboard with style-control leaderboard (overall category) on model × date. 54 weekly snapshots, August 28, 2024 – May 11, 2025. (A methodology discontinuity occurred on May 19, 2025, where the SC algorithm changed and the mean bonus flipped from +46 to −15 in one week; data after May 11 excluded.)

*Result:* Mean style bonus for top-20 frontier models grew from **+24.3 ELO points (Aug 2024) to +47.0 ELO points (May 2025)** — a near-doubling in 9 months.

- Linear trend: +3.0 ELO pts/month
- R² = 0.909, p = 9.4×10⁻²⁹

Monthly breakdown:

| Month | Mean style bonus | Cumulative change |
|---|---|---|
| Aug 2024 | +24.3 | — |
| Sep 2024 | +25.2 | +0.9 |
| Oct 2024 | +30.6 | +6.3 |
| Nov 2024 | +30.5 | +6.2 |
| Dec 2024 | +30.4 | +6.1 |
| Jan 2025 | +33.9 | +9.7 |
| Feb 2025 | +40.0 | +15.7 |
| Mar 2025 | +45.0 | +20.7 |
| Apr 2025 | +46.7 | +22.4 |

*Confound:* Reasoning models (DeepSeek-R1, QwQ-32B) entered the top-20 in January 2025. These models have long chain-of-thought outputs that inflate raw ELO relative to style-controlled ELO. Restricting to the pre-reasoning window (Aug–Dec 2024), the trend remains significant: +1.6 ELO pts/month, R² = 0.619, p = 1.4×10⁻⁵, growing from +24.3 to +32.2 (+8 ELO points over 5 months). The pre-reasoning portion of the trend is cleaner evidence.

*Interpretation:* The bias premium in Arena ELO has been growing monotonically. Labs that use Arena-like preference signal in their training loops are training against an increasingly biased signal. This is the empirical pattern predicted by D2, though not a controlled experiment.

### D2-B: Simulation

We numerically integrate the discrete-time recursion with parameters calibrated to Arena ELO scale (τ = 0.3 logits ≈ 52 ELO, γ = 0.01, C_q = 0.5, η = 0.5). The theoretical phase transition is at:

$$\Delta^* \approx 0.065 \text{ logits} \approx 11.2 \text{ ELO points}$$

This is in the same order of magnitude as the empirical B3 phase transition (~20–30 ELO), providing quantitative consistency between the formal model and the data.

Simulated trajectories:

| Starting gap | Δ_final | β_init | β_final | Outcome |
|---|---|---|---|---|
| 3 ELO (below Δ*) | 0.0 | 0.486 | 0.500 | collapse |
| 8 ELO (below Δ*) | 0.0 | 0.462 | 0.500 | collapse |
| Δ* ≈ 11 ELO | 11.2 | 0.446 | 0.446 | unstable equilibrium |
| 25 ELO (above Δ*) | grows | 0.382 | → 0 | stable↑ |
| 60 ELO (above Δ*) | grows | 0.240 | → 0 | stable↑ |

Step-by-step for Δ₀ = 8 ELO (below Δ*):

| Round | Δ (ELO) | α_t | β_t (bias) | Quality signal | Bias drag |
|---|---|---|---|---|---|
| 0 | 8.00 | 0.538 | 0.462 | 0.00310 | 0.00462 |
| 5 | 7.29 | 0.535 | 0.465 | 0.00280 | 0.00465 |
| 10 | 6.42 | 0.531 | 0.469 | 0.00245 | 0.00469 |
| 20 | 4.09 | 0.520 | 0.480 | 0.00153 | 0.00480 |
| 40 | 0.00 | 0.500 | 0.500 | 0.00000 | 0.00500 |

The quality signal decays as the gap shrinks; the bias drag does not. Each round, β_t increases. The collapse is guaranteed once below Δ*: signal → 0 before drag does.

*Note on the "above Δ*" case:* In the linear model, Δ_t diverges above Δ* because there is no diminishing returns term. In practice, capability is bounded by compute and data. For the paper: the key asymmetric prediction is that below Δ* the collapse is inevitable, not that above Δ* a specific finite equilibrium is reached.

### D2-C: 2023 vs. 2025 comparison

We apply the B3-2 test (weaker-but-longer wins rate) to 2023 Arena battle data, using December 2023 ELO ratings and character-level response lengths extracted from conversation text.

| Year | ELO gap | N conflict | Weaker-but-longer wins | z vs. 50% |
|---|---|---|---|---|
| 2023 | <10 ELO | 296 | 53.4% | +1.17 (p ≈ 0.24) |
| **2025** | <10 ELO | 3,009 | **57.1%** | **+7.90 (p < 10⁻¹⁴)** |
| 2023 | <30 ELO | 1,910 | 49.2% | −0.73 (p ≈ 0.47) |
| **2025** | <30 ELO | ~8,000 | **56.1%** | **+7.90** |

In 2023 Arena data, there is **no detectable length bias** in coin-flip or close pairs (49–53%, all non-significant). By 2025, a strong and highly significant bias effect has emerged (56–57%). The directional change is consistent with two years of accumulated iterative training signal.

*Caveats:*
- The 2023 coin-flip sample (n=296) is underpowered; non-significance may partly reflect power.
- 2023 ELO estimates had wider confidence intervals (early Arena), so "coin-flip by ELO" pairs may have had real quality differences poorly captured by ELO.
- Character lengths (2023) vs. token counts (2025) introduce measurement noise.
- Despite these limitations, the 2023 <30 ELO tier (n=1,910) is adequately powered and shows 49.2% — essentially no bias — compared to 56.1% in 2025.

---

## Summary: Stage 2 Verification Status

| Claim | Status | Key evidence |
|---|---|---|
| A1: Win rates converging to 50% | ✅ Established | 6× variance compression, 2023–2026 |
| A3: Cross-model preference data used | ✅ Established | UltraFeedback, Nectar, Arena Learning |
| B2: Annotators have consistent biases | ✅ Literature | Zheng 2023, Sharma 2023, Shapira et al. |
| **B3: Bias dominates at small gaps** | ✅ **New** | 21.4× style/ELO at frontier; phase transition at ~20–30 ELO |
| C1: Iterative training → more similar | ✅ Theory | F(0)<0 rigorously; drag dominates near zero |
| C4: Each update more biased | ✅ Theory (conditional) | Gradient ratio (1−α)/α → 1; conditional on B3 |
| **D2: Bias grows over time** | ⚠️ Partial | Style bonus +2× in 9 months; 2023 vs. 2025 comparison |

**What remains for the paper:**

The weakest point is D2: the style bonus trend is suggestive but confounded by reasoning models and a methodology change. D2 would be substantially strengthened by a controlled fine-tuning experiment. The proposed options (in order of feasibility):

1. **Reward model contamination experiment** (2 days, no GPU): Take outputs from two similar HuggingFace models on a real prompt set, label preferences using a mixed oracle (α·quality + (1−α)·length at varying α), train reward models, show the learned length coefficient increases as α decreases. Isolates the reward model contamination mechanism directly.

2. **Pythia-70M DPO cycle** (3–5 days, 1 GPU): Two fine-tuned 70M variants, iterative DPO with mixed preference oracle, track quality and verbosity over 5–10 rounds. Most convincing controlled experiment at small scale.

The E claims (filtering to clear wins improves training) also remain entirely untested and would provide the constructive complement to D.

---

## Data and Code

**Analysis scripts:**
- `analysis_b3.py` — B3 results (R² decomposition, pure bias test, interaction model)
- `analysis_d2.py` — D2 results (style bonus trend, simulation, 2023 vs. 2025 comparison)
- Existing scripts in `empirical_findings.md` — A1 through Finding 10

**Key datasets:**
- `data/data/train-00000-of-00007.parquet` + `data/140k/data/train-0000{1-6}-of-00007.parquet` — 2025 Arena battles (135k rows)
- `data/data/train-00000-of-00001-cced8514c7ed782a.parquet` — 2023 Arena battles (33k rows)
- `data/arena_leaderboard.parquet` — 215 weekly ELO snapshots (May 2023–May 2026)
- `data/style_control/text_style_control/full-00000-of-00001.parquet` — Style-controlled ELO (Aug 2024–May 2026)

# Supplementary Analysis: Preference Heterogeneity and Capability Dimensionality

We ran three analyses to ask whether Arena battle outcomes contain evidence of pluralistic preferences — different annotators genuinely valuing different things — and whether the human evaluation signal decomposes into more than one capability dimension.

---

## A. Predictability Decomposition

We decompose battle outcome variance into three layers to distinguish signal from noise.

**Pair-level.** For each frontier model pair (ELO gap < 50 pts), we predict battle outcomes using the Bradley-Terry win probability derived from ELO. Pair-level R² = 0.73 — ELO reliably predicts which model wins *on average* across a batch of battles. We also detect a systematic position bias of approximately 7 ELO points favoring the model shown in position B; adjusting for this marginally improves pair-level fit.

**Battle-level.** For individual battles, R² = 0.07. For frontier pairs with |ΔELO| < 10 pts, predicting "the BT-favored model wins" is correct 50% of the time — indistinguishable from a coin flip. The prompt-specific outcome is almost entirely unpredictable from the ELO score of either model.

**Overdispersion test.** If annotators had systematically heterogeneous preferences — some raters reliably preferring Claude, others preferring Gemini — battle outcomes should overdisperse relative to the Bernoulli variance at the pair's win probability. We compute the residual variance of battle outcomes around BT-predicted win rates and compare it to the expected Bernoulli noise floor. Residual variance = 0.235; expected Bernoulli noise = 0.248. The residual is *below* the noise floor — no overdispersion. We find no evidence that annotators have preferences that deviate systematically from the aggregate win rate.

**Interpretation.** Preferences are not heterogeneous at the annotator level: different raters sampling the same pair produce outcomes consistent with a single shared win probability. The uncertainty in Arena rankings is driven by insufficient information (noisy individual battles), not by genuine disagreement across annotators. This has a direct implication for pluralistic evaluation: the pluralism, if it exists, operates at the *task* level rather than the *person* level.

---

## B. Task-Level Specialization: Language Permutation Tests

To test whether different tasks elicit genuinely different model strengths, we analyzed all 25,693 decisive coding battles from the 2025 Arena dataset (April–July 2025). We applied a regex-based language detector across 18 programming languages to each conversation, yielding the following distribution:

| Language | Battles |
|---|---|
| SQL | 5,696 |
| Python | 4,251 |
| JavaScript | 4,153 |
| Bash | 1,575 |
| CSS | 920 |
| TypeScript | 678 |
| PHP | 577 |
| HTML | 523 |
| C++ | 493 |
| Java | 465 |
| Other | 2,312 |

For each frontier model, we test whether win rates vary significantly across programming languages using a weighted between-language variance statistic compared against a permutation null (20,000 permutations per model). With 51 frontier models tested, we apply both Bonferroni (α/51 = 0.00098) and Benjamini-Hochberg FDR (5%) corrections.

| Model | Observed variance | p-value | BH significant |
|---|---|---|---|
| Claude Opus 4 Thinking | 0.0142 | 0.001 | Yes |
| Claude Opus 4 | 0.0118 | 0.001 | Yes |
| Claude 3.7 Sonnet Thinking | 0.0091 | 0.003 | Yes |
| Gemini 2.5 Flash | 0.0064 | 0.091 | No |
| GPT-4.1 | 0.0031 | 0.247 | No |
| Qwen3-235B | 0.0028 | 0.311 | No |

Three models — all Anthropic thinking models — survive multiple testing correction. All other frontier models (OpenAI, Google, Alibaba, DeepSeek, Mistral) are language-generalists at p > 0.09. The Claude thinking models show higher win rates in Python and lower win rates in SQL relative to their baseline; the effect is real but small (±6–8 percentage points). Most frontier models are indistinguishable across programming languages, meaning task-level pluralism in this dimension is modest.

---

## C. Dimensionality of Human Preferences: Per-Category ELO PCA

The most direct test of preference pluralism is whether Arena's per-category ELO ratings reduce to a single dimension or span multiple. We use the full Arena leaderboard dataset (368 models, 215 weekly snapshots, `lmarena-ai/leaderboard-dataset`) and take the most recent ratings for the five categories where Arena assigns per-category ELO: math, coding, creative_writing, instruction_following, and hard_prompts_english. We retain the 347 models with complete data across all five categories.

For each category, we compute residuals from overall ELO to partial out raw quality:

$$r_{i,c} = \text{ELO}_{i,c} - \text{ELO}_{i,\text{overall}}$$

We then standardize and apply PCA to the 5-dimensional residual matrix (347 models × 5 residuals).

### Variance explained

| Component | Variance explained | Cumulative |
|---|---|---|
| PC1 | 59.3% | 59.3% |
| PC2 | 22.4% | 81.7% |
| PC3 | 10.1% | 91.8% |

### PC1 — Technical specialization (59.3%)

| Category residual | Loading |
|---|---|
| hard_prompts_english | +0.540 |
| coding | +0.531 |
| math | +0.421 |
| instruction_following | +0.420 |
| creative_writing | −0.268 |

PC1 separates models that over-perform relative to their overall ELO on technical tasks (hard prompts, coding, math, instruction following) from models that over-perform on creative writing. It represents a **technical specialization vs. creative quality** axis.

### PC2 — Creative and IF breadth (22.4%)

| Category residual | Loading |
|---|---|
| creative_writing | +0.796 |
| instruction_following | +0.576 |
| coding | +0.047 |
| hard_prompts_english | +0.035 |
| math | −0.173 |

PC2 captures models that simultaneously over-perform on both creative writing and instruction following, with math and hard prompts loading near zero. This dimension is dominated by a single provider: the 10 highest-scoring models on PC2 are all Claude models (Claude 3.7 Sonnet Thinking at +3.70, Claude Opus 4 Thinking at +3.19, Claude Sonnet 4 Thinking at +2.54). The lowest-scoring models are GPT-OSS variants (−3.73, −3.69), Nvidia Nemotron (−2.92), and Amazon Nova experimental models — all characterized by strong technical ELO but creative writing residuals of −75 to −92 ELO points.

### Category cross-correlations (n = 347 models)

| Category pair | Spearman ρ | Interpretation |
|---|---|---|
| coding × hard_prompts_english | +0.824 | Near-redundant |
| coding × instruction_following | +0.633 | Correlated cluster |
| math × hard_prompts_english | +0.563 | Correlated cluster |
| math × creative_writing | −0.359 | Genuine tradeoff |
| coding × creative_writing | −0.410 | Genuine tradeoff |
| creative_writing × instruction_following | +0.030 | **Orthogonal** |

**Creative writing and instruction following are statistically orthogonal** across the full model population (ρ = 0.030). Models that over-perform at creative writing are neither more nor less likely to over-perform at instruction following. Meanwhile, coding and hard prompts are nearly redundant (ρ = 0.824), and math/coding are negatively correlated with creative writing. The 5-category space effectively collapses to two independent dimensions: a **technical cluster** (coding, hard_prompts, math, and partly IF) and a **creative quality axis**.

### Interpretation

Human preferences over frontier models contain genuine two-dimensional structure. The scalar Arena ELO collapses both dimensions into a single number, which explains why: (1) 95% of frontier model pairs have overlapping confidence intervals at the scalar level, yet (2) 92–95% of those same pairs have a category-level reversal in at least one dimension (Finding 8b). The pluralism in preferences is real — it operates at the task-type level, not the annotator level — and it is concentrated in the math vs. creative writing tradeoff, the only category pair whose residuals are genuinely orthogonal after conditioning on overall capability.

---

## Summary

| Question | Finding |
|---|---|
| Do annotators have heterogeneous preferences? | No — no overdispersion; outcomes consistent with a single shared win probability per pair |
| Do different programming languages predict battle outcomes? | Mostly no — only 3 models (all Claude thinking) survive multiple testing correction |
| Do human preferences decompose into multiple dimensions? | Yes — two dimensions explain 81.7% of residual variance |
| What are those dimensions? | (1) Technical specialization (coding/math/hard_prompts) vs. creative writing; (2) Creative+IF breadth simultaneously |
| Where is the pluralism? | At the task level: math and creative writing are orthogonal capability dimensions that Arena's scalar ELO collapses |

The evidence points toward a specific failure mode: Arena's pluralism problem is not that different users want different things (annotator-level heterogeneity is undetectable), but that different *tasks* elicit genuinely different model strengths, and the scalar ELO averages over these differences rather than surfacing them. A multi-dimensional leaderboard with separate ratings for math and creative writing would recover real differentiation that is currently invisible.

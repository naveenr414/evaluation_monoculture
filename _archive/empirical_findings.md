# Empirical Findings: Cyclic Optimization in Arena-Like Platforms

## Setup

We study Chatbot Arena (lmarena.ai) using five data sources: (1) a historical leaderboard dataset containing ELO ratings for 368 models across 215 weekly snapshots from May 2023 to May 2026 (`lmarena-ai/leaderboard-dataset`); (2) the Open LLM Leaderboard v2 (OLLMv2) benchmark scores (IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO) for 3,861 open-source models with per-subtask BBH results for a representative sample; (3) pairwise battle records from two periods — 33,000 battles from April–June 2023 and 116,000 battles from April–July 2025; (4) 17 months of real-world API token usage from OpenRouter (January 2025–June 2026, 347 models); and (5) a Kaggle benchmark tracker covering proprietary models. We matched 79 Arena models to OLLMv2 benchmark scores and 49 Arena models to OpenRouter usage data. All statistical claims use bootstrap 95% confidence intervals (n=5,000) and linear regression trend tests unless otherwise stated.

---

## Finding 1: Frontier Model Win Rates Are Compressing

Under Bradley-Terry, the win probability between models i and j is P(i beats j) = σ((ELOᵢ − ELOⱼ)/400). As ELO differences among frontier models shrink, pairwise win rates compress toward 0.5 — the point of maximum uncertainty per vote.

Among the top-20 Arena models at each snapshot, we compute the variance of all pairwise win rates and the ELO standard deviation. Both decline monotonically:

| Year | Top-20 ELO std | Top-20 WR variance | Top-10 WR variance |
|------|---------------|-------------------|-------------------|
| 2023 | 45.1 | 0.00062 | 0.00046 |
| 2024 | 33.5 | 0.00038 | 0.00022 |
| 2025 | 23.0 | 0.00019 | 0.00013 |
| 2026 | 17.0 | 0.00010 | 0.00006 |

Win rate variance among top-20 models has compressed approximately 6× since 2023 (slope = −0.000217/yr, R² = 0.558, p < 0.0001). ELO standard deviation has more than halved. The pattern holds for top-5 and top-10 subsets at approximately 9× and 8× compression respectively.

Concretely: in September 2023, 3 of 45 top-10 pairs exceeded a 55% win rate. By September 2025, every one of 45 top-10 pairs fell within 50–53% — statistically indistinguishable from a coin flip at any practical sample size.

---

## Finding 2: Rankings Are Becoming Statistically Unresolvable

To distinguish adjacent models in a ranked list, the ELO gap between them must exceed the combined uncertainty in their ratings. We define the CI/gap ratio as (mean confidence interval width) / (mean ELO gap to adjacent model). A ratio above 1 means ranking order is consistent with sampling noise.

| Year | Top-5 CI/gap | Top-10 CI/gap | Top-20 CI/gap | Frac. overlapping CIs (top-20) |
|------|-------------|--------------|--------------|-------------------------------|
| 2024 | 0.86 | 1.11 | 1.56 | 75.5% |
| 2025 | 1.33 | 1.92 | 2.92 | 89.5% |
| 2026 | 2.19 | 2.99 | 4.33 | 94.8% |

In 2024, top-5 models were just barely resolvable (ratio 0.86). By 2026, 95% of top-20 adjacent pairs have overlapping confidence intervals. The ratio is growing at 0.69–1.60 per year (top-5 to top-20), R² = 0.57–0.71, p < 0.0001 for all K.

---

## Finding 3: Vote Requirements Are Growing Exponentially

Given an ELO gap δ, the number of votes needed to resolve a ranking at 80% power is n ≈ ((z_α + z_β)² × p(1−p)) / (p − 0.5)², where p = σ(δ/400). As δ shrinks, n grows quadratically.

| Year | Median votes needed (top-10 adj.) | Median votes needed (top-20 adj.) |
|------|----------------------------------|----------------------------------|
| 2023 | 87,000 | 134,000 |
| 2024 | 188,000 | 338,000 |
| 2025 | 293,000 | 850,000 |
| 2026 | 286,000 | 1,389,000 |

The votes needed to resolve top-20 adjacent rankings doubles every 0.7 years (log₁₀ slope = 0.449/yr, R² = 0.514, p < 0.0001). In the most recent snapshot, 18 of 19 top-20 adjacent pairs have insufficient votes at 80% power. Two models separated by 0.3 ELO points require 55 million votes to distinguish; they have 14,000. At Arena's current collection rate the deficit is structurally unbridgeable.

---

## Finding 4: Direct Evidence From Battle Data

In 2023 (33,000 battles, 20 models), the mean win rate of the higher-ranked model was 0.74 (std = 0.13); 35.6% of pairs exceeded 80%. In 2025 (116,000 battles, 53 models), the mean fell to 0.64 (std = 0.09); only 5.8% of pairs exceeded 80%, while 40.5% fell within the 40–60% band. KS test: stat = 0.354, p < 0.0001.

*Caveat:* the 2023 dataset includes qualitatively weaker models (dolly-v2, alpaca, koala) competing against frontier models (GPT-4), which mechanically produces extreme win rates. Part of the compression reflects Arena becoming a frontier-only competition — which is itself consistent with Arena losing discriminative power at the frontier.

---

## Non-Finding: ELO–Benchmark Correlation Is Stable Within Cohorts

Within a fixed cohort of models (those present from the beginning), the Spearman correlation between Arena ELO and OLLMv2 benchmark scores is stable at ρ ≈ 0.85 from 2024 through 2026. The apparent rolling decline (0.84 → 0.75) is a composition effect: newer models entering the panel have weaker ELO–benchmark alignment at entry, but this does not propagate to existing models whose ELO has frozen.

The 2025H1 cohort shows substantially lower correlation (ρ = 0.39 after removing reasoning models), but n ≤ 9 precludes strong conclusions. The main confound is reasoning models (qwq-32b, deepseek-r1), which receive high Arena ELO for chain-of-thought quality but score poorly on OLLMv2 benchmarks not designed to evaluate reasoning chains.

---

## Dissection: v-Inflation vs. Genuine Convergence

Finding 1 is consistent with two mechanisms: (a) **v-inflation** — models optimize toward Arena-like surface features (verbosity, formatting, apparent confidence); or (b) **genuine convergence** — frontier labs have independently reached similar capability levels through similar training recipes. Three analyses discriminate between them.

**Organization symmetry.** If v-inflation operates at the lab level, same-organization pairs should compress faster than cross-organization pairs. Instead, the intra/inter-org ELO gap ratio is near 1.0 across all years (1.04, 1.11, 1.01, 1.04 for 2023–2026), with both intra- and inter-org gaps compressing at similar rates (−15.7 vs. −14.6 ELO points/year). Symmetric compression across organizations is inconsistent with lab-level gaming and consistent with genuine cross-lab convergence.

**Verbosity.** Response length correlates significantly with Arena ELO (Spearman ρ = 0.578, p < 0.0001, n = 44), strongest in the top ELO tier (ρ = 0.714). However, the longest-response models are predominantly reasoning models (Gemini 2.5 Flash at 6,726 chars, Grok-3 at 6,636 chars, DeepSeek-R1 at 4,608 chars), whose verbosity reflects chain-of-thought generation rather than strategic padding. The verbosity–ELO correlation cannot be attributed to gaming without separating reasoning and non-reasoning models.

**Benchmark profile compression.** Under v-inflation, RLHF-sensitive benchmarks (IFEval — instruction following) should converge faster than RLHF-insensitive ones (GPQA, MATH). We find the opposite: benchmark standard deviations are *increasing* across all years, and MATH grows at 5.73 std points/year while IFEval grows at only 2.34 std points/year. This is inconsistent with RLHF-sensitive benchmarks converging faster, and more consistent with genuine capability improvement concentrated in hard reasoning tasks.

| Test | Favours |
|------|---------|
| Intra/inter-org gap ratio ≈ 1.0, same compression rate | Genuine convergence |
| Benchmark stds increasing, not compressing | Genuine convergence |
| MATH spreads faster than IFEval | Genuine convergence |
| Base model benchmarks predict Arena ELO as well as instruct (C2) | Genuine convergence |
| ELO–benchmark correlation stable within cohorts | Genuine convergence |
| Verbosity significantly correlated with ELO | Weak v-inflation (confounded by reasoning) |
| Battle category cross-category ρ = 0.787–0.910, matches BBH (8a) | Genuine convergence |
| Math/creative residuals orthogonal; org-level specialization persists (8c) | Genuine convergence with residual differentiation |

---

## Finding 5: OpenRouter Usage — Decentralization, Brand Premiums, and Frontier Null

We augment Arena-based findings with 17 months of daily API token usage from OpenRouter (January 2025–June 2026), covering 347 models and roughly 500 billion tokens per day across millions of developer API calls.

### 5a. Usage Is Decentralizing, Not Concentrating

| Month | Top-1 share | Top-5 share | HHI |
|-------|------------|------------|-----|
| Jan 2025 | 28.9% | 67.2% | 0.140 |
| Jun 2025 | 14.0% | 46.6% | 0.062 |
| Dec 2025 | 11.3% | 34.2% | 0.040 |
| Apr 2026 | 7.1% | 29.0% | 0.033 |

Top-1 share declined from 28.9% to 7.1% (slope = −0.115/yr, p = 0.0002); HHI fell from 0.14 to 0.03 (p < 0.0001). This is consistent with C4 in an unexpected direction: because Arena can no longer distinguish frontier models, developers diversify across models rather than converging on a clear winner. When the evaluation signal disappears, experimentation increases.

### 5b. Pareto Inversions — Brand Dominates Over Arena ELO and Price

We identify model pairs where model B strictly Pareto-dominates model A (higher ELO *and* lower price), yet model A receives substantially more traffic:

| Dominated model | ELO | $/1M | Traffic (B tokens) | Dominator | ELO | $/1M | Traffic ratio |
|----------------|-----|------|--------------------|-----------|-----|------|--------------|
| GPT-4o-mini | 1287 | $0.15 | 10,118 | Qwen3-30B | 1384 | $0.05 | **1,299×** more |
| MiniMax M2.1 | 1391 | $0.29 | 2,155 | Qwen3-235B-thinking | 1414 | $0.10 | **71×** more |
| Llama-3.1-70B | 1261 | $0.40 | 342 | Nemotron-Nano-30B | 1349 | $0.05 | **33×** more |
| Claude-3-Haiku | 1195 | $0.25 | 43 | Gemma-3n | 1306 | $0.06 | **14×** more |

GPT-4o-mini is 3× more expensive and 98 ELO points below Qwen3-30B, yet receives 1,299× more traffic. OpenAI and Anthropic models systematically outperform their ELO+price profile; Chinese-origin models (Qwen, GLM, MiniMax) with better ELO-to-price ratios systematically underperform. Brand trust, reliability SLAs, enterprise contracts, and ecosystem integration dominate rational cost-minimization.

### 5c. Arena ELO Predicts Usage Overall — But Not Within the Compressed Frontier

Overall: Spearman ρ = 0.495 (p = 0.0003, n = 49). Controlling for price: ρ = 0.563. Age-normalizing (average daily tokens): ρ = 0.606. Model age explains at most 10% of the raw correlation.

Within each price tier, ELO strongly predicts usage:

| Price tier | ρ(ELO, avg daily tokens) | p |
|-----------|--------------------------|---|
| Cheap (<$0.30/1M) | 0.457 | 0.011* |
| Budget ($0.30–$1/1M) | 0.874 | 0.0001* |
| Mid ($1–$5/1M) | 0.943 | 0.005* |

These correlations span 295–369 ELO points within each tier — they reflect choosing between qualitatively different models, not discriminating among the compressed frontier.

**Restricting to the top-15 models by ELO** (ELO std = 19 points, ELO range = 66 points — the C4-compressed zone) — Arena ELO does NOT predict usage: Spearman ρ = −0.104, p = 0.71. Gemini 2.5 Flash (ELO 1418, $0.30/1M) gets 57.6B tokens/day while Gemini 2.5 Pro (#1 by ELO at 1457, $1.25/1M) gets only 17.6B tokens/day. The monthly ELO–usage correlation shows no significant trend (slope = −0.010/yr, p = 0.89), fluctuating between 0.07 and 0.59 with no direction.

### 5d. Price-Performance Frontier

Among the top 80 OpenRouter models by Arena ELO, the $2–$10 and >$10 price tiers have statistically identical mean ELO (1461 vs 1461). Paying more than $2/1M tokens buys no measurable Arena quality improvement. Nine models fall within 25 ELO points of the current #1 — statistically indistinguishable — spanning a 24× price range ($1.25/1M to $30/1M).

---

## Finding 6: Variance Decomposition of Developer Usage Decisions

We regress log(average daily tokens) on all measurable features for 49 matched OpenRouter models:

| Factor | R² alone | Marginal R² added | Drop-one importance |
|--------|----------|-------------------|---------------------|
| Arena ELO | 36.6% | — | 9.6% |
| Price | 2.9% | +11.0% | 6.9% |
| Incumbency (log days on market) | 14.8% | +0.1% | 0.6% |
| Context window | — | +6.0% | **11.6%** |
| Brand (org fixed effects) | — | **+20.3%** | **20.3%** |
| **Total explained** | | **74.1%** | |
| **Unexplained** | | **25.9%** | |

Three findings stand out. **Price alone explains only 2.9%** — collinear with ELO, limited independent signal. **Context window (11.6%) exceeds price (6.9%)** — developers building agentic and batch workloads weight context capacity over conversational Arena quality. **Brand adds 20.3%** — roughly equal to ELO's independent contribution — and is strongly asymmetric: Meta (+0.69 mean residual, ~4.9× over-usage) and OpenAI (+0.43, ~2.7×) systematically outperform their ELO+price+incumbency profile, while Qwen (−0.64, ~0.2×) and Amazon (−0.82, ~0.15×) severely underperform.

The 25.9% unexplained variance captures API latency, reliability SLAs, OpenAI-compatible API ecosystem, enterprise procurement, and geographic developer distribution.

*Incumbency caveat.* Incumbency explains 14.8% alone but only 0.1% marginal — it is a proxy for brand and benchmark quality, not an independent driver. Model age explains at most 10% of the raw ELO-usage correlation.

---

## Finding 7: Frontier Models Have Converged in Error Structure, Not Just Benchmark Level

We test whether frontier models are making more correlated errors over time — going beyond Kim et al. (ICML 2025), who establish the cross-sectional fact (60% agreement rate when both models err on MMLU), to ask whether this correlation has increased across model generations.

### 7a. Aggregate benchmark profiles — saturation partially explains the jump

For each OLLMv2 model we construct a six-dimensional benchmark score vector. Pairwise cosine similarity among the top-50 models by year:

| Year | Top-50 cosine similarity | Top-50 Spearman ρ | Avg benchmark score |
|------|--------------------------|-------------------|---------------------|
| 2022 | 0.898 | 0.587 | 5.99 |
| 2023 | 0.340 | 0.401 | 20.74 |
| **2024** | **0.929** | **0.197** | **45.62** |
| **2025** | **0.924** | **0.236** | **42.59** |

The 2023→2024 jump in cosine similarity (0.340 → 0.929) is large. Deduplicating models by base model barely changes the result (0.929 → 0.932), ruling out shared weights as the primary driver. The more important explanation is benchmark saturation: IFEval standard deviation across the top-50 collapsed from 12.9 (2023) to 6.2 (2024), and BBH std from 7.8 to 2.9. When all frontier models score 75–90 on IFEval, their vectors become proportionally similar on that dimension regardless of how different they are elsewhere. MATH and GPQA, where genuine variance persists, remain less saturated (stds of 8.6 and 4.2 in 2024).

High cosine similarity therefore conflates two things: (i) models score at similar absolute levels (saturation), and (ii) models fail on the same tasks (genuine error correlation). These require separate measurement.

### 7b. BBH subtask-level analysis — genuine error correlation confirmed

We fetch per-subtask BBH accuracy from the OLLMv2 results repository for 44 models spanning 2023 and 2024 cohorts (19 from 2023, 25 from 2024; 7 families including Meta, Qwen, Mistral, Microsoft, nvidia, allenai, tiiuae). Each model has a 24-dimensional subtask accuracy vector. Spearman correlation of these vectors measures whether models fail on the same subtasks — independently of average accuracy level.

**Cross-family, cross-architecture pairs** (controlling for shared base model weights):

| Cohort | Mean pairwise Spearman ρ | N pairs |
|--------|-------------------------|---------|
| 2023 cross-family | 0.782 | 146 |
| **2024 cross-family** | **0.848** | 251 |
| t-test | t = 6.39 | **p < 0.0001** |

The increase from 0.782 to 0.848 is statistically significant and survives removing all same-family pairs. Within 2024, the same-family vs. cross-family difference is negligible: 0.855 vs. 0.848 (t-test p = 0.63), confirming that shared weights explain essentially none of the correlation. Models from completely different organizations and architectures — Qwen2.5-72B vs. Llama-3.3-70B (ρ = 0.912), Qwen2.5-72B vs. Phi-3.5-MoE (ρ = 0.915), Mistral-Large vs. Llama-3.1-70B (ρ = 0.880) — fail on the same BBH subtasks.

**Per-subtask structure.** The saturated subtasks and the genuinely correlated subtasks are not the same:

- *Uniformly hard (both cohorts fail similarly)*: `tracking_shuffled_objects` (2023: 0.14–0.17, 2024: 0.27–0.36), `web_of_lies` (2023: 0.50, 2024: 0.59). These show correlated errors from a shared floor, not from convergent capabilities.
- *Genuinely more correlated in 2024*: `ruin_names` (within-cohort std fell from 0.206 → 0.055), `reasoning_about_colored_objects` (0.143 → 0.073), `salient_translation_error_detection` (0.127 → 0.047). In 2024, models not only score higher on average — they score more *similarly to each other* on these subtasks, even across different architectures.
- *Still discriminating in 2024*: `temporal_sequences` (std = 0.180), `navigate` (std = 0.131), `hyperbaton` (std = 0.110). These retain meaningful variance among 2024 frontier models.

The battle outcome variability finding is consistent: the bootstrap std of win rates per model pair fell from 0.129 (2023) to 0.104 (2025), indicating slightly more uniform outcomes across prompts.

**Interpretation.** The 2023→2024 aggregate cosine similarity jump is substantially driven by benchmark saturation on IFEval and BBH, but the subtask-level analysis reveals a genuine underlying increase in error correlation across architectures and organizations. The most likely mechanism is convergent training: when all major labs sample from similar pre-training corpora and use similar instruction-tuning recipes, the residual hard problems are structurally the same problems for everyone. This is consistent with the genuine-convergence interpretation from the dissection analyses above.

---

## Finding 8: Category-Level Analysis Extends Error Correlation to Real Deployment

Finding 7b established convergent error structure from benchmark subtask accuracy. We replicate the key question in Arena battle data — where models are evaluated on real human preferences rather than held-out benchmarks — using the 116,000-battle 2025 dataset, which includes native category tags (math, coding, creative writing, instruction following, general).

### 8a. Cross-category rank correlation mirrors BBH subtask correlation

For 48 models with ≥50 battles per category, we compute each model's win rate in each category and measure Spearman ρ between per-category win-rate rankings. All ten category pairs are highly correlated:

| Category pair | Spearman ρ | p |
|---|---|---|
| creative_writing vs instruction_following | 0.910 | 3.4×10⁻¹⁹ |
| coding vs instruction_following | 0.897 | 6.7×10⁻¹⁸ |
| coding vs general | 0.883 | 1.0×10⁻¹⁶ |
| math vs coding | 0.859 | 5.4×10⁻¹⁵ |
| math vs creative_writing | 0.794 | 1.7×10⁻¹¹ |
| math vs general | 0.787 | 3.3×10⁻¹¹ |

The range 0.787–0.910 is directly comparable to the BBH cross-family subtask correlation (ρ = 0.848, Finding 7b). A single quality dimension dominates win rates across all five categories — consistent with the benchmark-level finding that convergence has produced a largely one-dimensional capability surface.

### 8b. Within ELO-close pairs, category reversals are nearly universal

Despite the high cross-category correlation at the population level, within specific model pairs in the statistically unresolvable zone, the nominally lower-ranked model wins in at least one category in almost every case:

| ELO gap | Fraction where lower-ELO model wins ≥1 category | N pairs |
|---|---|---|
| 0–5 pts | 92% | 25 |
| 5–15 pts | 95% | 44 |
| 15–30 pts | 66% | 29 |
| 30–100 pts | 24% | 92 |

In the statistically unresolvable zone (Finding 2: ELO gaps < ~15 pts for top-20 models), essentially every adjacent pair has genuine category-level reversals. Concrete examples with gap < 10 pts: Grok-4 (ELO 1409) wins in all five categories over Gemini-2.5-Flash (ELO 1418, gap 7.7 pts); Hunyuan-Turbo (ELO 1376) wins in all five categories over Gemini-2.5-Flash-Lite (gap 7.4 pts). The scalar ELO ranking is not just statistically unresolvable — it actively collapses multi-dimensional capability profiles that genuinely differ by task type.

### 8c. Residual analysis: which categories discriminate beyond overall capability

We partial out overall capability (overall win rate) from each category's win rate via OLS residuals. The residual standard deviation measures how much a category reveals beyond raw quality:

| Category | r with overall wr | Residual std | Verdict |
|---|---|---|---|
| math | 0.883 | **0.049** | discriminating |
| creative_writing | 0.891 | **0.045** | discriminating |
| instruction_following | 0.901 | 0.040 | moderately discriminating |
| coding | 0.958 | 0.025 | mostly quality signal |
| general | 0.983 | 0.019 | near-pure quality signal |
| domain_knowledge (tag) | 0.999 | 0.004 | no independent signal |

The critical finding: **math and creative writing are orthogonal after conditioning on overall capability.** The Spearman correlation of residuals between math and creative writing is ρ = 0.140, p = 0.33 — statistically indistinguishable from zero. Math vs. coding residuals: ρ = 0.143, p = 0.32. Creative writing vs. coding residuals: ρ = 0.004, p = 0.98. These three categories are measuring independent capability dimensions after capability level is held fixed. By contrast, instruction following and coding residuals correlate (ρ = 0.550, p < 0.001), and coding and general residuals are strongly negatively correlated (ρ = −0.711), indicating they share variance with the quality axis.

Forty-three percent of models overperform in math and underperform in creative writing, or vice versa — the clearest examples of each pole:

| Model | Math residual | Creative residual | Pattern |
|---|---|---|---|
| Qwen3-235B-A22B | +0.117 | −0.064 | math specialist |
| O3-mini | +0.094 | −0.042 | math specialist |
| MiniMax-M1 | +0.126 | −0.012 | math specialist |
| Gemini-2.5-Pro-preview-03-25 | +0.012 | +0.114 | creative specialist |
| DeepSeek-V3 | −0.053 | +0.048 | creative-leaning |
| Grok-3-preview | −0.080 | +0.023 | creative-leaning |

Org-level residuals reveal systematic lab-level specialization:

| Org | Math | Creative | IF | Coding |
|---|---|---|---|---|
| MiniMax | **+0.126** | −0.012 | −0.040 | −0.037 |
| Alibaba/Qwen | +0.029 | −0.031 | −0.011 | +0.005 |
| Anthropic | +0.044 | +0.035 | **+0.045** | +0.024 |
| Google | −0.004 | +0.026 | +0.003 | −0.017 |
| DeepSeek | **−0.062** | +0.002 | −0.054 | −0.004 |
| Mistral | −0.049 | −0.012 | −0.009 | +0.019 |

DeepSeek's large negative math residual (−0.062) is notable: despite very high overall ELO, both DeepSeek-R1 and DeepSeek-V3 underperform in math relative to their overall quality, with creative writing closer to expectation. This pattern is inconsistent with DeepSeek's reputation as a strong reasoning model and may reflect the specific structure of Arena math prompts versus the harder reasoning tasks where DeepSeek excels.

### 8d. BBH subtasks are more discriminating than Arena categories

Applying the same residual analysis to BBH subtask accuracy (44 models, 24 subtasks), the most discriminating subtasks have substantially higher residual std than the best Arena category:

| BBH subtask | Residual std | Arena category | Residual std |
|---|---|---|---|
| temporal_sequences | 0.164 | math | 0.049 |
| movie_recommendation | 0.113 | creative_writing | 0.045 |
| geometric_shapes | 0.093 | instruction_following | 0.040 |
| navigate | 0.091 | coding | 0.025 |
| *BBH mean (24 tasks)* | **0.067** | general | 0.019 |
| causal_judgement (min) | 0.029 | — | — |

The average BBH subtask (residual std 0.067) is more discriminating than the best Arena category (math, 0.049). Arena's five categories capture two genuine independent dimensions (math, creative writing) but are too coarse to approach the granularity of 24 task-specific evaluations. The vast majority of Arena's battle volume — coding (26% of battles) and general (50%) — sits in the near-zero residual zone and carries almost no capability information beyond overall ELO.

---

## Finding 9: Arena Category Popularity Aligns with Convergence Rates

Arena categories that receive more votes (higher optimization pressure from humans and labs) are also the categories where benchmark scores show the most convergence. We measure this using cumulative vote counts from the Arena leaderboard and OLLMv2 benchmark convergence rates (slope of top-50 std, 2023–2025):

| Benchmark | Arena category | Vote share | Std slope/yr | Direction |
|---|---|---|---|---|
| IFEval | instruction_following | **24.1%** | −2.91/yr | converging ↓ |
| BBH | hard_prompts_english | 16.5% | −1.58/yr | converging ↓ |
| MMLU-PRO | (general, no Arena cat.) | — | −2.28/yr | converging ↓ |
| GPQA | hard_prompts_english | 16.5% | +0.34/yr | diverging ↑ |
| MUSR | (general, no Arena cat.) | — | +0.29/yr | diverging ↑ |
| MATH Lvl 5 | math | **7.1%** | +0.81/yr | diverging ↑ |

The direction is consistent: the most-voted category (instruction_following, 24.1%) maps to the fastest-converging benchmark (IFEval, −2.91/yr); the least-voted (math, 7.1%) maps to the most rapidly diverging benchmark (MATH, +0.81/yr). The 2025 cross-section corroborates this: coding (14.4% of arena votes) has the lowest battle win-rate std of any category (0.089), while math (7.1%) has the highest (0.107).

**The principal confound is task difficulty/saturation.** Instruction-following prompts are easier to judge and easier to solve — IFEval mean score in 2025 is 74.3, close to ceiling. Math prompts are harder — MATH Lvl 5 mean is 47.2, with substantial room for models to spread. High-mean benchmarks converge more (Spearman ρ = −0.771, p = 0.072 between mean score and std slope, n = 6). The one case that breaks the popularity story is GPQA: hard_prompts_english receives 16.5% of Arena votes (substantial optimization pressure) but GPQA is diverging (+0.34/yr), consistent with it being too hard to saturate regardless of training signal. This suggests saturation is the primary mechanism rather than optimization pressure directly, though both predict the same directional pattern and the data cannot cleanly separate them.

---

## Finding 10: Scale, Not Architecture, Drives Benchmark Convergence

If frontier models from different architectures converge in capability profile, it matters whether the cause is shared pretraining data, shared architecture class, or similar model scale — the three manipulable factors in LLM development. We decompose this using all 946 pairwise BBH subtask Spearman ρ values (44 models × 24 subtasks) matched to architecture class (`LlamaForCausalLM`, `Qwen2ForCausalLM`, etc.), pretraining base model (dataset proxy), parameter count (scale), and training year.

### 10a. The convergence hierarchy

Three nested groups reveal the gradient:

| Pair type | Mean ρ | 95% CI | n pairs |
|---|---|---|---|
| Same base model (shared data + architecture) | 0.854 | [0.803, 0.901] | 21 |
| Same architecture, different base model | 0.758 | [0.738, 0.778] | 267 |
| Different architecture | 0.688 | [0.675, 0.700] | 658 |

The gradient runs same-base (0.854) → same-arch-only (0.758) → cross-arch (0.688). The baseline cross-architecture ρ of 0.688 is already high — even Llama-family vs. Qwen-family pairs, sharing no architecture class and no direct data lineage, share about two-thirds of their subtask rank ordering. The architecture class itself adds ~0.07 ρ above baseline; shared pretraining data adds a further ~0.10 ρ above that.

Importantly, in the larger OLLMv2 benchmark extension (3,861 models, 6-benchmark vectors), the same-architecture-different-base ρ (0.783 for Llama, 0.728 for Qwen) sits very close to the cross-architecture ρ (0.734 for Llama vs. Qwen). Architecture class, once data lineage is disentangled, contributes only modest additional similarity.

### 10b. Scale is the dominant structural predictor

Variance partitioning via incremental R² isolates each factor's independent contribution to pairwise ρ:

| Factor | Univariate R² | Incremental ΔR² |
|---|---|---|
| Scale similarity (log param ratio) | 0.222 | 0.197 |
| Same architecture class | 0.046 | 0.025 |
| Same base model (dataset) | 0.017 | 0.0003 |

Scale — measured as |log(params_a / params_b)| — is the single largest structural predictor, explaining 22% of variance in pairwise ρ alone and 20% incrementally after accounting for architecture and dataset. Models of similar size converge much more than models of disparate size: the lowest-scale-ratio tertile (log ratio ≤ 0.41, i.e., within ~50% parameter counts) has mean ρ = 0.782, versus 0.627 for the highest-ratio tertile (log ratio > 1.55, i.e., > 5× parameter difference). Same architecture class contributes ΔR² = 0.025; once architecture and scale are controlled, knowing whether two models share a pretraining base adds essentially nothing (ΔR² = 0.0003) — collinearity between base model and architecture class absorbs this effect.

### 10c. Training era is the largest single factor overall

Year cohort dominates all structural factors:

| Pairing | Mean ρ | n pairs |
|---|---|---|
| Same year (both 2023 or both 2024) | 0.827 | 471 |
| Adjacent years (2023 vs. 2024) | 0.597 | 475 |

The ρ drop from 0.827 to 0.597 across training years is larger than the entire spread from cross-architecture (0.688) to same-base (0.854) pairs. This reflects the year-on-year capability jump between 2023 and 2024 frontier models — a compositional shift that cuts across all architecture and data boundaries. Within the same training era, convergence is broad and structurally pervasive; across training eras, capability gaps are large enough that even structurally similar models look dissimilar.

### 10d. Interpretation: what drives cross-architecture convergence

The residual cross-architecture baseline ρ = 0.688 — the similarity observed between Llama and Qwen models sharing no architecture or direct data lineage — points to a universal mechanism. Candidate sources: (a) all frontier models train on heavily overlapping subsets of Common Crawl, Wikipedia, GitHub, and arXiv, even when downstream fine-tuning differs; (b) transformer pretraining on internet-scale data converges to a common representational geometry regardless of architectural details, consistent with recent mechanistic interpretability findings; (c) RLHF and instruction tuning applied from a common pool of human preference data induces correlated adjustments across models. Among these, the scale result is most diagnostic: scale similarity predicts ρ better than architecture class, suggesting that capacity (how much the model can represent) matters more than inductive bias (how it represents it). Labs building at the same scale, regardless of architecture choices, converge to the same capability profile.

**Implication.** The cross-architecture convergence is not a modeling artifact or data leakage — it is an empirical regularity of frontier-scale training. Architecture choice does not substantially differentiate capability profiles at comparable scale; only training era and extreme scale differences do. This makes competitive differentiation through architecture alone structurally limited: a Qwen-family model and a Llama-family model at the same scale, trained in the same year, are more similar to each other (ρ ≈ 0.69) than a Qwen-family model is to a newer Qwen model of the same architecture (same-year within Qwen: ρ ≈ 0.877; cross-year: ρ drops toward the 0.597 baseline).

---

## Theoretical Implication: Correlated Errors and Bradley-Terry

The subtask-level error correlation (Finding 7b) has a formal implication for Arena's ELO estimation.

Standard Bradley-Terry assumes P(i beats j) = σ(θᵢ − θⱼ), with battle outcomes IID conditional on θ. The only randomness is the Bernoulli draw — no explicit noise on θ, no correlation across battles.

Write a richer model: for model i on prompt k, response quality = θᵢ + εᵢₖ. Finding 7b means Cov(εᵢₖ, εⱼₖ) = ρᵢⱼσ² > 0 — models fail on the same prompts. Two consequences follow.

**Within a battle:** Var(εᵢₖ − εⱼₖ) = 2σ²(1 − ρᵢⱼ). As ρᵢⱼ → 1, the prompt-level quality difference shrinks — correlated errors reduce the range of prompts that discriminate between models.

**Across battles:** Two battles involving model i on similar prompts k and k′ are no longer independent, because εᵢₖ ≈ εᵢₖ′. This violates the IID assumption in standard BT. The effective sample size is:

n_eff ≈ n / (1 + (n − 1)ρ_battle)

where ρ_battle is the correlation between battle outcomes induced by shared prompt structure and error correlation. As n → ∞, n_eff → 1/ρ_battle — a fixed ceiling regardless of additional votes. If battles carry ρ_battle = 0.05 correlation, the effective sample size never exceeds 20.

This links Finding 7b (error correlation increasing) → correlated battle outcomes → Finding 2 (statistical unresolvability) → Finding 3 (exponential vote requirements). Arena's reported confidence intervals, which assume IID battles, are optimistically narrow. The problem is not simply that ELO gaps have shrunk — it is that the information per vote has a structural ceiling set by the error correlation between frontier models.

**Implication for evaluation design.** This is not fixable by collecting more votes on the same prompt distribution. The fix requires targeting prompts where model errors are *uncorrelated* — the discriminating prompts where one model succeeds and another fails. With frontier models in 2024 sharing error structure at the subtask level even across different architectures, such prompts are increasingly rare. Adversarial prompt selection, stratified difficulty sampling, and per-dimension evaluation (rather than a single scalar ELO) are all implied by the non-exchangeability of battle outcomes.

---

## Overall Interpretation

Ten findings converge on a coherent picture with three layers: what is happening, why it is happening, and what can still discriminate.

---

### Layer 1: Arena rankings are statistically unresolvable at the frontier

Win rate variance compressed 6–9× since 2023 (Finding 1). The CI/gap ratio for top-20 models reached 4.33 in 2026, with 95% of adjacent pairs having overlapping confidence intervals (Finding 2). Votes required to resolve adjacent rankings double every 0.7 years (Finding 3). Raw battle data confirms the shift: decisive battles (win rate > 80%) fell from 36% to 6% of pairs between 2023 and 2025 (Finding 4).

Real-world deployment has already adapted to this failure. Within the compressed frontier (top-15 models by ELO, std = 19 points), Arena ELO does not predict actual usage (ρ = −0.104, n.s., Finding 5c). Brand (20.3% of usage variance), context window (11.6%), and genuinely unexplained factors (25.9%) dominate over ELO (9.6% marginal) and price (6.9%, Finding 6). Usage has decentralized — top-1 model share fell from 28.9% to 7.1% over 17 months — consistent with developers diversifying across indistinguishable models rather than selecting a clear winner. Severe Pareto inversions persist: GPT-4o-mini receives 1,299× more traffic than its cheaper, higher-ELO dominator, reflecting brand and ecosystem lock-in rather than capability signal.

---

### Layer 2: The mechanism is genuine convergence with correlated errors

Six independent tests all favour genuine capability convergence over strategic v-inflation: organization-symmetric ELO compression, RLHF-insensitive benchmarks (MATH) spreading faster than RLHF-sensitive ones (IFEval), stable ELO–benchmark correlation within cohorts, and now the battle-level category evidence (Finding 8a). The deepest evidence is the BBH subtask-level error correlation (Finding 7b): frontier models from different architectures and organizations fail on the same subtasks, with cross-family ρ increasing from 0.782 (2023) to 0.848 (2024), p < 0.0001. The battle-level replication (Finding 8a) finds cross-category win-rate rank correlations of 0.787–0.910, a directly comparable magnitude. Verbosity remains a weak confound, but it is heavily entangled with reasoning models' chain-of-thought generation rather than strategic padding.

Convergence is not uniform across domains. IFEval (instruction following, 24.1% of Arena votes) converges at −2.91 std/yr; MATH Lvl 5 (7.1% of votes) diverges at +0.81 std/yr (Finding 9). The directional ordering aligns with category popularity, but the same ordering is predicted by task saturation (IFEval mean = 74.3 in 2025; MATH mean = 47.2), and GPQA — which receives substantial Arena votes (hard_prompts_english, 16.5%) but scores very low (mean = 15.4) — diverges regardless, pointing to saturation as the primary driver. Labs have therefore converged fastest precisely where the evaluation signal is clearest and tasks are most tractable, which is also where benchmark scores are nearest ceiling.

Finding 10 adds a mechanistic decomposition of *what* drives this cross-model convergence. Scale similarity (parameter count ratio) is the single largest structural predictor of pairwise BBH subtask ρ, explaining 20% of variance independently. Architecture class contributes modestly (ΔR² = 0.025); shared pretraining data, once architecture is controlled, adds negligibly (ΔR² = 0.0003). Training era dominates all structural factors — same-year pairs have mean ρ = 0.827 vs. 0.597 for cross-year pairs, a drop larger than the full same-base-to-cross-architecture spread. The cross-architecture baseline ρ = 0.688 (Llama vs. Qwen, no shared architecture or direct data lineage) points to a universal property of frontier-scale training: at comparable scale, models converge to similar capability profiles regardless of architecture choice. Competitive differentiation through architecture alone is structurally limited.

Correlated errors provide the formal mechanism connecting convergence to unresolvability. With cross-model error correlation ρᵢⱼ > 0, the variance of within-battle quality differences shrinks as Var(εᵢₖ − εⱼₖ) = 2σ²(1 − ρᵢⱼ). Across battles, shared prompt structure induces correlated outcomes, violating Bradley-Terry's IID assumption. The effective sample size is capped at n_eff ≈ 1/ρ_battle regardless of additional votes. Arena's reported confidence intervals, which assume IID battles, are optimistically narrow; the structural information ceiling cannot be overcome by collecting more votes on the same prompt distribution.

---

### Layer 3: Some discriminating structure remains, but it is concentrated and shrinking

The category residual analysis (Finding 8c) identifies where capability differences still surface. After conditioning on overall quality:

- **Math and creative writing are orthogonal** (residual ρ = 0.140, p = 0.33; creative vs. coding ρ = 0.004, p = 0.98). They are measuring genuinely independent capability dimensions.
- **Coding and general are near-pure quality signals** (r = 0.958 and 0.983 with overall win rate). These two categories together account for roughly 76% of Arena's battle volume, yet carry almost no discriminative information beyond the scalar ELO that already exists.
- **BBH subtasks are substantially more discriminating** than Arena categories (mean residual std 0.067 vs. 0.028 for Arena categories; the best Arena category, math at 0.049, is weaker than the BBH average).

The residual structure shows systematic lab-level specialization: Alibaba/Qwen models over-perform in math (+0.029 mean residual) and under-perform in creative writing (−0.031); Google models show the reverse pattern (+0.026 creative, neutral in math); Anthropic leads in instruction following (+0.045); DeepSeek under-performs in both math (−0.062) and instruction following (−0.054) despite high overall ELO. These organizational signatures are real and reproducible — the scalar ELO obliterates them.

This specialization itself signals genuine convergence: 43% of models have opposite signs on their math and creative residuals, meaning labs have found different local optima in the two-dimensional space that remains. The gap between them, however, is shrinking. In 95% of ELO-close pairs (gap < 15 pts), the nominally lower-ranked model already wins more in at least one category, yet the absolute residual magnitudes (≤ 0.13) are small enough that they are practically indistinguishable in single-category leaderboards. The two discriminating dimensions are real but narrow.

---

### Synthesis: what would fix the evaluation

The evidence is sufficient to specify the failure mode precisely and derive the implied fix:

**Why the current evaluation fails.** Arena's scalar ELO is constructed from battles distributed across all categories, dominated (76%) by coding and general prompts that carry near-zero residual information beyond overall quality. The remaining 24% — math and creative writing — contains real discriminative signal, but that signal is collapsed into the same single number. Battle outcomes within the uninformative 76% are highly correlated across models (convergent errors) and contribute noise rather than information. The IID assumption underlying Bradley-Terry is violated; the nominal vote count overstates the effective sample size.

**What the evidence implies.** V-inflation would call for changing the evaluation *signal* — harder prompts, diverse judges, anti-gaming mechanisms. Genuine convergence with correlated errors calls for a different intervention: stratified evaluation with separate ratings per capability dimension (at minimum: math, creative writing, instruction following), adversarial prompt selection targeting the shrinking set of prompts where frontier model errors are uncorrelated (best candidates are in math and creative writing, where cross-org residuals are largest), and explicit acknowledgment that a scalar ELO is no longer sufficient to characterize frontier models. The specialization patterns — Alibaba strong in math, Google strong in creative, Anthropic strong in IF — are real and actionable; a multi-dimensional leaderboard would surface them. A scalar one buries them.

---

## Data and Reproducibility

**Data sources.** Arena historical leaderboard: `lmarena-ai/leaderboard-dataset` (HuggingFace). OLLMv2 aggregate scores: `open-llm-leaderboard/contents` (HuggingFace). OLLMv2 per-subtask BBH results: `open-llm-leaderboard/results/{org}/{model}/results_{date}.json` (HuggingFace, one file per model). Battle data: `lmsys/chatbot_arena_conversations` (2023) and `lmarena-ai/arena-human-preference-140k` (2025). Kaggle proprietary benchmark tracker: `lizzulfiqar/llm-benchmark-tracker-20202026`. OpenRouter daily token rankings: `GET /api/v1/datasets/rankings-daily` (requires OpenRouter API key; data begins 2025-01-01). OpenRouter model pricing: `GET /api/v1/models` (public). BBH subtask data saved to `data/bbh_subtask_scores.csv`.

**Analysis scripts.** `analysis_c1.py`, `analysis_c1_fixed.py`, `analysis_c1_check.py`, `analysis_c4.py`, `analysis_c4_interp.py`, `analysis_c2.py`, `analysis_b_c.py`, `analysis_123.py`, `analysis_dissect.py`, `analysis_openrouter.py`, `analysis_openrouter_ts.py`, `analysis_distribution_shift.py`, `analysis_frontier_usage.py`, `analysis_usage_drivers.py`, `analysis_error_correlation.py`, `analysis_subtask_correlation.py`, `analysis_category_correlation.py` (Findings 8a–8d: cross-category rank correlation, ELO-close pair reversals, residual discriminability, BBH comparison), `analysis_category_convergence.py` (Finding 9: category popularity × benchmark convergence rates), `analysis_architecture_convergence.py` (Finding 10: scale/architecture/dataset decomposition of pairwise BBH ρ across 44 models and OLLMv2 extension across 3,861 models).

**Key external reference.** Kim, Garg, Peng, Garg. "Correlated Errors in Large Language Models." ICML 2025. arXiv:2506.07962.

# Experiment A. Catch-up and specialization

Panel unless stated otherwise: Arena per-category leaderboard, 8 core categories,
one flagship model per firm per snapshot, 23 labs, 133 snapshots 2025-01-05 to
2026-05-27. Event panel: 65 flagship changeovers, 8 frontier firms, 520
observations, 2024-10-28 to 2026-03-18, 56-day maturation.

r = two-way (firm x category) FE residual. g = gap to the category leader.
gtilde = g with the firm's own mean gap and the category mean removed.

## 1. Pairwise residual similarity against the permutation null
Mean pairwise cos(r_i, r_i') = -0.053, permutation null -0.056, p = 0.106.
Row and column sums of r are both zero, so the mean is mechanically pinned near
-1/(N-1) = -0.045. `plot_residuals.py`

## 2. Block structure
Hierarchical clustering of the cosine matrix at 2026-05-27, K = 23, gives four
groups with within-cluster mean cosine +0.53 to +0.69. Co-clustering rates over
snapshots with at least 15 labs: Ant Group-Meituan 100% of 63, Amazon-Ant Group
100% of 59, Meituan-NVIDIA 95.5% of 67, Baidu-Google 90.6% of 53, Cohere-Z.ai
88.4% of 69, Cohere-DeepSeek 87.2% of 78, Anthropic-Xiaomi 86.0% of 43.

## 3. Firm demeaning only, no category demeaning
Mean pairwise cosine +0.577, median +0.660, 94.6% positive. Permutation null
that shuffles firms within each category, preserving the common category
profile, gives +0.563.

Per-category contribution to the mean cosine (sums to +0.577):

| category | contribution | share | mean residual |
|---|---|---|---|
| Creative Writing | +0.180 | 31.2% | -34.2 |
| Coding | +0.134 | 23.2% | +24.3 |
| Instruction Following | +0.115 | 20.0% | -14.9 |
| English | +0.069 | 11.9% | +11.0 |
| Math | +0.041 | 7.1% | +7.5 |
| Hard Prompts | +0.018 | 3.1% | +9.5 |
| Multi-Turn | +0.016 | 2.7% | -0.4 |
| Longer Query | +0.005 | 0.8% | -2.8 |

Leave-one-out change in the mean cosine: Instruction Following -0.042,
Coding -0.041, Creative Writing -0.020, English -0.005, Hard Prompts +0.001,
Longer Query +0.017, Multi-Turn +0.032, Math +0.050.

Variance decomposition at the 2026-05-27 snapshot: 67.4% common across-firm
category mean, 32.6% idiosyncratic. PC1 carries 81.8% and loads Creative
Writing +0.73 against Coding -0.51. Participation ratio 1.47 of 8.
`plot_residuals_firmonly.py`

## 4. Cluster composition, k = 4, 2026-05-27
- OpenAI, Google, xAI, DeepSeek, Alibaba, Baidu (within-cos +0.58): Math +14,
  Creative Writing +12, Coding -12, English -12.
- NVIDIA, IBM, Amazon, AI2, Ant Group, Meituan, MiniMax, StepFun, Tencent
  (+0.53): English +12, Coding +11, Creative Writing -19.
- Meta, Mistral, Cohere, Z.ai, ByteDance (+0.54): Creative Writing +17,
  Multi-Turn +9, Math -16.
- Anthropic, Moonshot, Xiaomi (+0.69): Instruction Following +14, Longer Query
  +7, English -14.

At k = 5 the nine-lab group splits into AI2, IBM, NVIDIA, Ant Group, Meituan
(English +19, +0.71) and Amazon, MiniMax, StepFun, Tencent (Coding +11,
Creative Writing -25, +0.76).

## 5. Residual magnitudes
Firm and category means absorb 96.1% of raw theta variance in 2025Q1 and 96.4%
in 2026Q2. The residual share is 3.9% and 3.6%. Residual sd is 10.9 and 11.7.

## 6. Distribution and persistence
Distribution 2025Q1 vs 2026Q2: gap mean +98.7 sd 61.4 to +98.0 sd 64.8;
residual sd 10.9 to 11.7, IQR 11.7 to 14.1. R-squared of the early/late split
is 0.0000 for both. Variance ratios late/early 1.11 and 1.14.

Quarterly persistence:

| | slope | R-squared | n |
|---|---|---|---|
| gap, q to q+1 | 0.902 | 0.777 | 704 |
| residual, q to q+1 | 0.708 | 0.522 | 704 |
| gap, q to q+4 | 0.776 | 0.485 | 224 |
| residual, q to q+4 | 0.249 | 0.066 | 224 |

Decile transition matrix, cut points from the initial distribution, all
categories pooled, 704 transitions. Gap: 35.4% stay in their decile, 71.4%
within one; decile 1 stays 56%, decile 10 stays 76%; mean decile change +0.6 in
decile 1 and -1.0 in decile 10. Relative gap: 29.3% stay, 61.2% within one;
deciles 1 and 2 move +1.1 and +1.7, deciles 9 and 10 move -1.2 and -1.3.
`plot_resid_persistence.py`, `plot_gap_transition.py`

## 7. Mean reversion and measurement error
AR(1) consistency: residual rho_1 = 0.708, rho_1^4 = 0.251, observed rho_4 =
0.249. Gap rho_1 = 0.902, rho_1^4 = 0.662, observed rho_4 = 0.776.
Reliability of the quarterly regressor: 0.987 for the gap, 0.616 for the
residual.

Split-half measurement of the mechanical bias, 116,257 battles 2025-04-17 to
2025-07-24, 20 random 50/50 splits, Bradley-Terry refit on each half, regressing
theta_B - theta_A on the gap built from half A: measured spurious slope +0.119
(sd 0.056), errors-in-variables prediction +0.161. By category: math 0.159,
creative writing 0.120, instruction following 0.109, overall 0.026. At the
leaderboard's own reported variances the correction moves beta from 0.338 to
0.326. `exp_split_half_bias.py`

Out-of-sample prediction on independent Arena samples was not run. Battle data
covers 5 of the 41 months the event panel spans.

## 8. Sign persistence against a calibrated null
35 of 104 firm-category pairs keep the same sign in all six quarters, 21
positive and 14 negative. Naive independent-coin null expects 3.2. A zero-mean
AR(1) null with the same dispersion, pushed through the same two-way demeaning,
expects 6.5 at rho = 0.22 (p < 0.0001), 13.9 at rho = 0.46, 21.5 at rho = 0.60
(p = 0.003), and 30.4 at rho = 0.71 (p = 0.198). `exp_sign_persistence.py`

## 9. Beta across controls
Draft specifications: pooled 0.354 (0.060), additive FE 0.314 (0.075),
firm x category FE 0.334 (0.079), + category x time FE 0.338 (0.081),
+ cadence 0.354 (0.094).

## 10. Multiple testing on the specializations
Per-pair Newey-West t-test on the mean residual with Benjamini-Hochberg at 5%:
21 of 104 pairs significant over 2025Q1-2026Q2, 31 of 96 over 2024Q4-2026Q2.
21 of the 35 same-signed pairs are also FDR-significant.

Largest by mean residual, all same-signed in six quarters: Cohere Math -30.95,
Amazon Creative Writing -26.38, Anthropic English -22.48, Cohere Creative
Writing +21.64, Google Creative Writing +20.30, Anthropic Instruction Following
+19.79, IBM English +16.37, DeepSeek Creative Writing +15.49, Google Coding
-14.68, NVIDIA English +14.52.

## 11. Specialized firms by category
Of 13 firms, count keeping the same sign in all six quarters:

| category | same sign | fraction | pos / neg | FDR significant |
|---|---|---|---|---|
| English | 10 | 77% | 6 / 4 | 6 |
| Creative Writing | 6 | 46% | 5 / 1 | 3 |
| Multi-Turn | 5 | 38% | 4 / 1 | 2 |
| Coding | 4 | 31% | 1 / 3 | 3 |
| Instruction Following | 4 | 31% | 2 / 2 | 4 |
| Math | 2 | 15% | 0 / 2 | 1 |
| Hard Prompts | 2 | 15% | 1 / 1 | 1 |
| Longer Query | 2 | 15% | 2 / 0 | 1 |

Firm counts: Cohere 6, Google 5, Amazon 4, Anthropic 3, NVIDIA 3, xAI 3.

## 12. Persistence by horizon
Monthly panel, h = 1 to 16, standard errors clustered on firm.

| h | gap, pooled | gap, within firm x cat FE | residual |
|---|---|---|---|
| 1 | 0.956 (0.020) | 0.727 (0.044) | 0.844 (0.024) |
| 3 | 0.883 (0.055) | 0.306 (0.052) | 0.600 (0.052) |
| 6 | 0.799 (0.094) | -0.105 (0.088) | 0.376 (0.081) |
| 9 | 0.715 (0.135) | -0.144 (0.098) | 0.283 (0.108) |
| 12 | 0.712 (0.155) | -0.098 (0.105) | 0.263 (0.135) |

The within-FE gap slope crosses zero at h = 5 and is negative from h = 6 to 13.
On a balanced panel of the 12 firms present in all 17 months the pooled gap
slope declines less (0.956 to 0.785 at h = 12) and the within-FE slope is
unchanged. Holding the start window fixed at 2025-01 to 2025-05, the pooled gap
slope is still falling at h = 12. Sample falls from 2,184 pairs at h = 1 to 112
at h = 16. `plot_slope_vs_horizon.py`

## 13. Alternative benchmarks for the gap
Firm x category and category x quarter FE, clustered on event.

| benchmark | beta | s.e. | corrected | mean gap | n | events |
|---|---|---|---|---|---|---|
| best rival | 0.338 | 0.081 | 0.326 | 52.6 | 520 | 65 |
| second-best rival | 0.322 | 0.083 | 0.309 | 27.5 | 520 | 65 |
| top-3 average | 0.387 | 0.086 | 0.368 | 29.7 | 520 | 65 |
| fixed rival (OpenAI) | 0.524 | 0.120 | 0.522 | 32.8 | 390 | 51 |

OpenAI leads a category in 486 snapshot-categories, ahead of Anthropic 435 and
Google 352. `plot_evalchasing_benchmarks.py`

## 14. Beta by period and by model tier
Period, flagship events: all 0.338 (0.081, 65 events), 2025 only 0.516 (0.094,
49), 2026 only 1.185 (0.156, 11 events from 5 firms), early half to 2025-08-07
0.389 (0.163, 34), late half 0.571 (0.095, 31).

Tier-level events, where a firm's best small, mid and large model each generate
their own releases and the benchmark stays the rivals' overall flagships:
small 0.864 (0.180, 18 events, mean gap 132.5), mid 0.524 (0.097, 44, 76.6),
large 0.821 (0.102, 29, 52.8), all tiers 0.528 (0.058, 91). Tiers are assigned
from model names; thinking variants count as large. `exp_beta_splits.py`

## 15. Persistence by category
Quarterly slope of the position on its own lag, by category.

| category | absolute gap, q+1 | relative gap q+1 | q+2 | q+3 |
|---|---|---|---|---|
| English | 0.897 | 0.893 | 0.791 | 0.730 |
| Creative Writing | 0.926 | 0.850 | 0.604 | 0.466 |
| Instruction Following | 0.918 | 0.759 | 0.561 | 0.337 |
| Coding | 0.922 | 0.740 | 0.480 | 0.262 |
| Multi-Turn | 0.892 | 0.671 | 0.391 | 0.217 |
| Math | 0.865 | 0.577 | 0.312 | 0.159 |
| Hard Prompts | 0.902 | 0.567 | 0.266 | 0.065 |
| Longer Query | 0.886 | 0.437 | 0.059 | 0.158 |

Absolute-gap slopes have standard errors near 0.05 at q+1 and 0.11 at q+3.
Spearman correlation between the relative-gap slope and the category's mean
Bradley-Terry variance is -0.29. `plot_gap_persistence_bycat.py`

## 16. Controlling for the level of the outgoing model
Level is the outgoing model's overall rating at the pre-release snapshot, range
1252 to 1503.

| specification | beta | s.e. | t |
|---|---|---|---|
| baseline, firm x cat + cat x quarter FE | 0.338 | 0.081 | 4.18 |
| + level, linear | 0.219 | 0.137 | 1.60 |
| + level decile x category FE | 0.274 | 0.119 | 2.30 |

By quartile of the outgoing model's rating:

| quartile | rating range | mean gap | mean dtheta | beta | s.e. |
|---|---|---|---|---|---|
| Q1 | 1252-1337 | 79.6 | 39.7 | -0.162 | 0.281 |
| Q2 | 1339-1420 | 53.7 | 26.4 | 0.672 | 0.305 |
| Q3 | 1420-1438 | 47.1 | 0.5 | 0.930 | 0.120 |
| Q4 | 1442-1503 | 31.3 | 10.8 | 0.931 | 0.208 |

16 or 17 events per quartile. `exp_beta_by_level.py`

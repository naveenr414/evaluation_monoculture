# Experiment B. The shrinking top

Two panels of data. Arena per-category leaderboard, 8 core categories, all rated
models, 2023-05 to 2026-05. OpenRouter daily token volume for the top-50 models,
2025-01 to 2026-06, 347 slugs, 192 matched to Arena names covering 83.5% of
pooled tokens. Firm-keyed leaderboard series stop at 2026-03-19, because the
`organization` field goes blank for 50-61 tail models from 2026-04-20 onward
(distinct orgs fall 56 to 25 with no change in the model count).

## 1. Where usage sits relative to the evaluated frontier
Average OpenRouter token share served by models within x% of that month's top
Arena `overall` rating:

| x | 1 | 2 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|
| share | 0.040 | 0.061 | 0.306 | 0.699 | 0.778 | 0.783 | 0.783 |

The curve saturates at 0.792, the matched-token ceiling, so essentially all
matched usage is inside 20%. The inflection is at x = 10%. Unmatched volume is
treated as outside every threshold, making each figure a floor.

The KL divergence between usage shares and Arena vote attention was not re-run
with the current matcher.

## 2. Number of models near the top
Models within 10% of the category top, mean per snapshot:

| category | 2025Q1 | 2026Q2 |
|---|---|---|
| English | 44.9 | 151.0 |
| Math | 39.5 | 139.5 |
| Multi-Turn | 32.3 | 130.6 |
| Coding | 44.6 | 124.5 |
| Longer Query | 42.8 | 117.6 |
| Hard Prompts | 39.6 | 116.4 |
| Creative Writing | 42.8 | 110.3 |
| Instruction Following | 39.4 | 104.1 |

The rated pool grows from 167 to 345 models per snapshot over the same window.

Fixed-rank spreads, pooled over categories and measured on the snapshot panel:
leader minus 10th best falls 92.6 to 46.1 points, trend -16.2 per year, r =
-0.49; leader minus 20th -29.4 per year, r = -0.69; leader minus 50th falls 268
to 92 points, -65.1 per year, r = -0.89.

Percentile spreads do not fall, because the band absorbs more models as the pool
grows: top 1% +7.2 points per year (r = +0.31), top 5% +6.9 (r = +0.22), top 10%
+0.35 (r = +0.01). Models covered by those bands go from 2, 4 and 7 in 2024Q1 to
4, 18 and 35 in 2026Q2, with correlations against time of +0.88, +0.96 and +0.96.

## 3. Share of models within x% of the top
Share of rated models within 10% of the category top:

| category | 2025Q1 | 2026Q2 |
|---|---|---|
| English | 27.1% | 43.2% |
| Math | 23.6% | 41.1% |
| Multi-Turn | 19.7% | 37.6% |
| Coding | 26.9% | 36.1% |
| Longer Query | 25.8% | 35.9% |
| Hard Prompts | 23.9% | 33.3% |
| Creative Writing | 25.9% | 31.7% |
| Instruction Following | 23.7% | 29.8% |

In rating points the same comparison is flat at tight thresholds and rising at
loose ones. Share within 50 points of the top falls in five of eight categories
(Coding 3.8 to 3.1, Instruction Following 2.6 to 1.8); share within 100 points
rises in seven of eight (Math 9.3 to 21.6, Multi-Turn 8.4 to 21.7).

## 4. Quality distribution, frontier-firm models, 2025Q1 vs 2026Q2
Gap to the category leader, measured over the full field, for models belonging to
the 8 frontier firms:

| | n (model x cat) | mean | median | sd | p90 | within 50 pts |
|---|---|---|---|---|---|---|
| 2025Q1 | 16,456 | 197.4 | 177.9 | 113.7 | 349.9 | 6.1% |
| 2026Q2 | 33,638 | 206.3 | 171.5 | 132.2 | 407.4 | 5.2% |

Kolmogorov-Smirnov D = 0.076, p = 1.8e-55. Distinct frontier-firm models rise
from 95 to 208. The median falls 6 points while the mean, sd and p90 rise: the
distribution lengthens its lower tail rather than compressing.

Among a fixed cohort of 11 firms present every month with at least three distinct
flagships, the spread of flagships is flat: sd +2.2 per year (r = +0.08), leader
minus median +6.2 (r = +0.13), leader minus worst -44.6 (r = -0.39). For
Anthropic, OpenAI and Google alone the trend is +0.0 per year on both sd and
range, with the range averaging about 60 points.

## 5. Usage distribution, 2025 H1 vs 2026 H1

| | models | firms | eff. models | eff. firms | top-1 model | top-3 models | top-10 models | top-1 firm | top-3 firms | Gini |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025 H1 | 172 | 25 | 18.5 | 4.23 | 12.8% | 34.0% | 58.7% | 34.9% | 75.9% | 0.839 |
| 2026 H1 | 161 | 24 | 41.8 | 10.41 | 5.3% | 15.6% | 38.3% | 16.6% | 43.4% | 0.723 |

The number of models and firms appearing is unchanged; the concentration among
them falls on every measure.

## 6. Entry gap of new frontier-firm releases, relative
Gap at first appearance, for releases by the 8 frontier firms, expressed three
ways:

| quarter | points | as % of leader rating | as % of leader-to-floor span | n |
|---|---|---|---|---|
| 2024Q2 | 79.6 | 6.26 | 15.43 | 51 |
| 2024Q4 | 92.3 | 6.84 | 15.75 | 129 |
| 2025Q2 | 89.6 | 6.29 | 13.53 | 192 |
| 2025Q4 | 52.1 | 3.51 | 7.44 | 161 |
| 2026Q2 | 62.6 | 4.12 | 8.74 | 120 |

Trends per year: points -7.73 (r = -0.077), fraction of leader -0.0101
(r = -0.142), fraction of span -0.0188 (r = -0.098). All three are weakly
negative and none is well determined.

Over the wider firm set the same measure is flat. Mean entry gap by category
runs 79 to 105 points with trends between -1.5 and -29.2 per year, and 2.6% to
5.7% of entries take the category lead outright.

## 7. Time behind the frontier
For a model entering category k with rating theta, the age is the time since the
running-maximum frontier was last at or below theta. Zero means the entrant takes
the lead. Frontier firms only, releases from 2024-06, 154 releases and 1,037
model-category rows.

Pooled over categories: 1.9 months in 2024Q3, 4.4 in 2024Q4, 5.9 in 2025Q2, 6.3
in 2025Q4, 8.9 in 2026Q1, 9.6 in 2026Q2.

| category | mean | median | % at the frontier | trend per year | corr |
|---|---|---|---|---|---|
| English | 5.71 | 5.29 | 7.3 | +2.50 | +0.39 |
| Math | 5.85 | 5.58 | 7.9 | +5.03 | +0.67 |
| Coding | 5.97 | 5.52 | 8.9 | +2.98 | +0.43 |
| Hard Prompts | 6.12 | 5.88 | 10.3 | +4.25 | +0.53 |
| Longer Query | 6.14 | 5.91 | 9.0 | +3.72 | +0.46 |
| Instruction Following | 6.87 | 7.06 | 10.2 | +4.08 | +0.45 |
| Creative Writing | 6.94 | 7.16 | 8.7 | +8.17 | +0.78 |
| Multi-Turn | 8.28 | 8.61 | 6.4 | +6.33 | +0.65 |

Every category rises. The point gap at entry is roughly flat over the same
window, so the rise reflects the frontier advancing more slowly in rating points,
which stretches a fixed point gap into more months.

## 8. Effective number of firms
Inverse Herfindahl of the weight shares.

Deployment, OpenRouter monthly tokens: effective firms 3.25 to 9.19, +4.92 per
year, r = +0.93. Effective models 7.16 to 22.34, +13.02 per year, r = +0.85.
Top-3 firm share 0.76 to 0.47, -0.293 per year, r = -0.92.

Evaluation, Arena, averaged over the 8 categories. The answer depends on the
weight:

| weight | start | end | trend per year | corr |
|---|---|---|---|---|
| top-10 presence | 3.45 | 2.82 | +0.19 | +0.13 |
| top-20 presence | 8.05 | 4.64 | -0.52 | -0.34 |
| top-50 presence | 16.03 | 8.53 | -3.09 | -0.82 |
| softmax, tau = 50 | 3.50 | 8.19 | +2.45 | +0.86 |
| firms rated | 32.0 | 50.9 | +9.72 | +0.98 |

Fixed-rank presence concentrates; rating-weighted attention disperses and tracks
the OpenRouter series. Top-10 presence is flat near 3.

Scripts: `plot_openrouter_topshare.py`, `plot_near_top_by_cat.py`,
`plot_top_spread.py`, `plot_cohort_spread.py`, `expB_distributions.py`,
`plot_entrant_gap.py`, `plot_months_behind.py`, `plot_effective_firms.py`,
`plot_release_cadence.py`.

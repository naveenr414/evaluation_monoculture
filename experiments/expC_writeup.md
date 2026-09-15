# Experiment C. Production functions

Setup throughout: theta_i,k = a_i,k phi(x_i,k), investment x unobserved, budget
sum_k x_i,k = c_i, firm maximises sum_k w_k theta_i,k.  The first-order condition
gives psi(x_k) = phi'(x_k)/phi(x_k) = lambda / (w_k theta_i,k), so
x_k = psi^{-1}(lambda/(w_k theta_k)) with lambda fixed by the budget, and
a_i,k = theta_i,k / phi(x_i,k).  For every (c_i, w) the matrix is point
identified; the identified set is swept out by varying c_i and w.

Data: 8 frontier firms, flagship at 2026-03-19 (before the organization-field
break), 8 core categories, theta measured above the weakest rated model in each
category, w from Arena vote shares.  Event panel where noted: 65 flagship
changeovers, 520 observations.

## 1. Which production functions survive
Draft Table 2 lists four families by the features they can represent: scalar
a_i, diagonal A_i, general A_i, and unrestricted F_i.  Two results bear on the
choice.

Out-of-sample model comparison on the event panel, leave-one-event-out
prediction of a held-out release's full 8-category dtheta vector
(`expR_structural_Fi.py`):

| model | params per firm | CV R^2 |
|---|---|---|
| Model B: a_i scales the shared B_ridge | 1 | 0.3633 |
| Model 1: a_i on isotropic, SVD basis | 1 | 0.3311 |
| Model 0: F_i common, SVD basis | 0 | 0.3069 |
| Model 2: a_i, b_i, SVD basis | 2 | 0.3069 |
| Model C: beta_i only | 1 | 0.1917 |
| Model C: beta_i + gamma_i | 2 | 0.1917 |

In Model C the selected ridge penalty is 1e9, i.e. complete shrinkage to the
pooled value, so the firm-specific fit is numerically identical to the common
one.  Model 2 reaches the same 0.3069 as Model 0 for the same reason.  Only
Model B shows a firm-specific gain, +0.056.

Second, the recovered A_i carries no firm-specific shape (item 3 below), so
diagonal and general A_i are not separated by these data.  The families are
eliminated by identification, not by fit.

## 2. Identified set of a_i,k
Budget swept over c in [0.1, 1000] at fixed vote-share w.  Every entry falls
monotonically in c.

| firm | Coding min -> max | Math min -> max | English min -> max |
|---|---|---|---|
| Anthropic | 181.6 -> 80,502 | 166.5 -> 141,446 | 122.3 -> 27,078 |
| Google | 173.6 -> 77,519 | 163.3 -> 136,232 | 118.2 -> 26,079 |
| OpenAI | 174.1 -> 76,192 | 162.8 -> 133,875 | 115.3 -> 25,621 |
| Alibaba | 168.3 -> 75,301 | 160.2 -> 132,345 | 116.7 -> 25,346 |
| xAI | 164.8 -> 73,549 | 150.1 -> 129,228 | 113.6 -> 24,753 |
| Mistral | 161.0 -> 70,326 | 142.4 -> 123,525 | 109.4 -> 23,666 |
| DeepSeek | 156.9 -> 68,690 | 139.7 -> 120,659 | 107.5 -> 23,121 |
| Meta | 129.8 -> 54,893 | 113.9 -> 96,390 | 84.0 -> 18,456 |

Across all 64 entries the max/min ratio has median 473 and ranges from 215 to
868.  A three-order-of-magnitude budget range moves each entry by two to three
orders of magnitude.  The floor used to make theta positive is a further free
normalisation.

## 3. Shape of A_i and what it represents
Normalising each firm's row by its own mean, the variance of the normalised
entries decomposes as:

| c | category | firm | firm x category |
|---|---|---|---|
| 1 | 100.0% | 0.0% | 0.0% |
| 10 | 99.8% | 0.0% | 0.2% |
| 100 | 98.6% | 0.0% | 1.4% |

The largest spread across firms within a category is 0.012 at c = 1 and 0.106 at
c = 100, against a category range of roughly 0.4 to 1.8.

Common profile at c = 10: Math 1.48, Creative Writing 1.17, Coding 1.13,
Multi-Turn 1.14, Longer Query 1.00, Instruction Following 0.76, Hard Prompts
0.75, English 0.57.  The ordering is close to the inverse of the vote-share
weights, which is what the first-order condition produces.

Firms differ by a scalar.  Mean a over categories at c = 10: Anthropic 1003,
Google 968, OpenAI 948, Alibaba 938, xAI 916, Mistral 872, DeepSeek 852, Meta
682, a 1.47x spread that reproduces the ranking by overall rating.

## 4. Non-logarithmic families
The construction needs only phi increasing and concave with invertible
phi'/phi.  For the power family phi = x^alpha, psi = alpha/x, giving a closed
form:

    x_k = c w_k theta_k / sum_j w_j theta_j        a_k = theta_k^(1-alpha) (S/(c w_k))^alpha

Two consequences.  The allocation does not depend on alpha: for OpenAI at c = 10
the vector x is 1.086, 0.497, 2.079, 1.669, 0.642, 0.630, 0.913, 2.484 for
alpha = 0.25, 0.5 and 0.75 alike.  And a_k scales as c^(-alpha) uniformly, so the
shape of A_i is free of the budget.

| family | median max/min of a over c in [0.1, 1000] | shape drift over c |
|---|---|---|
| log(1+x) | 472.8 | 0.796 |
| x^0.25 | 10.0 | 0.000 |
| sqrt(x) | 100.0 | 0.000 |
| x^0.75 | 1000.0 | 0.000 |

Under a power law the identified set is exactly a ray: the scale is free and
nothing else is.  Under log(1+x) the shape drifts with the budget as well.

The firm-level conclusion is unchanged by the functional form.  Variance of the
normalised A at c = 10 comes from category in every family: 99.6% for sqrt(x),
98.2% for x^0.25, 100.0% for x^0.75, with firm variance 0.0% throughout.
Pairwise correlations between firms' normalised shapes under sqrt(x) run from
0.990 to a median of 0.997; the scale spread is 1.47x (Anthropic 714.9 to Meta
485.6).

## 5. Out-of-sample robustness
Covered by the model comparison in item 1.  The ridge selection is the
informative part: in two of three firm-specific families the penalty chosen by
cross-validation shrinks the firm parameters entirely to the pooled value.

## 6. Does more history buy a better production function?
For each training size n, draw n events at random, fit with the ridge chosen by
4-fold CV inside the training split, score on the held-out events, 120 draws per
size (`expC_learning_curve.py`).

| training events | shared | + firm scale a_i | + firm shape b_i | gain from a_i |
|---|---|---|---|---|
| 12 | -1.331 | -1.249 | -1.933 | +0.082 |
| 20 | -0.067 | -0.079 | -0.130 | -0.012 |
| 26 | +0.136 | +0.093 | -0.181 | -0.043 |
| 32 | +0.120 | +0.097 | +0.057 | -0.024 |
| 40 | +0.238 | +0.204 | +0.144 | -0.034 |
| 48 | +0.243 | +0.226 | +0.212 | -0.017 |
| 56 | +0.190 | +0.165 | +0.140 | -0.024 |
| 60 | +0.085 | +0.027 | -0.048 | -0.059 |

The shared response improves with history: unusable below about 20 events,
crossing zero there and reaching roughly 0.24 by 40 to 48, a slope of +0.68 per
e-fold of training events.  The gain from a_i is negative at every size from 16
events on, with a trend of -0.043 per e-fold.  The two-parameter version is worse
at every size.

The n = 56 and n = 60 rows fall because the test set is down to 9 and 5 events.
The negative gain is in tension with expR's Model B result of +0.056; expR fits
the scale under leave-one-out, where every firm has many training events, while
random splits leave some firms thinly covered.

## 7. Extrapolation
Three candidate forms fitted to each series -- linear in time, linear in log
time, and a saturating exponential a - b exp(-ct).  Form chosen by backtest on
the last five points, then refit on the full series and extrapolated 18 months
with a band of +/-1.96 residual standard deviations (`expD_extrapolation.py`).

| series | chosen form | backtest RMSE | resid sd | now | +18 months |
|---|---|---|---|---|---|
| Frontier rating | log-time | 13.6 | 14.7 | 1522 | 1621 |
| Models within 10% of top | linear | 27.8 | 11.6 | 107 | 180 |
| Leader - 10th best | saturating | 9.4 | 12.1 | 50.2 | 45.7 |
| Effective firms (OpenRouter) | linear | 1.78 | 0.87 | 9.6 | 17.0 |
| Months behind frontier | linear | 0.84 | 0.67 | 9.4 | 15.7 |
| Entry gap (points) | saturating | 8.5 | 10.3 | 70.6 | 81.8 |

Backtest RMSE by form, chosen form starred:

| series | linear | log-time | saturating |
|---|---|---|---|
| Frontier rating | 26.269 | 13.563* | 26.262 |
| Models within 10% of top | 27.774* | 37.324 | 27.776 |
| Leader - 10th best | 14.059 | 10.539 | 9.441* |
| Effective firms | 1.778* | 2.233 | 1.778 |
| Months behind frontier | 0.838* | 1.824 | 2.117 |
| Entry gap | 20.857 | 11.177 | 8.471* |

Three series separate cleanly: frontier rating is log-time (13.6 against 26.3),
months behind frontier is linear (0.84 against 1.82 and 2.12), and leader minus
10th best is saturating (9.4 against 14.1), flattening near 46 points rather than
going to zero.  Models within 10% of top and effective firms tie between linear
and saturating, so their +18-month values are the linear reading only.  The entry
gap's residual sd of 10.3 against a level of 70 makes its band wide.

These are curve fits with no behavioural model behind them, and each backtest
rests on five points.

Scripts: `expC_production_set.py`, `expC_production_family.py`,
`expC_learning_curve.py`, `expD_extrapolation.py`, `expR_structural_Fi.py`.

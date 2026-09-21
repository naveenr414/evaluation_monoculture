# Evaluation and Monoculture — Empirical Results

Working write-up of the empirical program, August 2026. All results run on data held
locally; scripts are in `experiments/`. Numbers here are the final ones (recomputed on
the complete datasets); where an earlier run differed materially, both are shown.

Confidence is marked throughout: **[solid]**, **[moderate]**, **[soft]**.

---

## 0. Headline findings

1. **The two standard estimators of model similarity are mechanically determined.** Raw
   cosine on capability profiles is 0.9999 in every period; the two-way fixed-effect
   correction is algebraically pinned near −1/(N−1) regardless of the data. **[solid]**
2. **Public evaluation resolves far fewer dimensions than it advertises** — 2–4 independent
   dimensions against 6–8 nominal — and finer measurement buys real structure only at a
   rate of roughly 2× per 6.5× resolution. **[solid]**
3. **Technology differences are suggestive, not established.** Broad six-dimension
   directions are highly aligned; the 39-task test is borderline (p = 0.050), while
   the more readable pairwise-alignment test is not significant (p = 0.202). **[moderate]**
4. **Homogenizing pressure is real.** Firms close ~55% of their per-category gap to
   rivals' frontier with each flagship release. (The panel previously showed this
   strengthening over time, 0.34→0.52; after fixing a duplicate-event bug — see §3.4 —
   that trend reverses to flat-to-mildly-declining, 0.62→0.52. The "intensifying" claim
   is retracted.) **[solid]**
5. **Part of that pull is imitation, not evaluation response.** The pull toward a specific
   leading model is identical whether a firm trails it or already beats it. **[moderate]**
6. **Monoculture has not occurred.** Capability profiles are no more similar than chance
   and show no trend over two years; deployment concentration fell sharply. **[solid]**
7. **Deployment is fragmenting fast** — effective number of firms 3.25 → 9.19 in 18
   months, top-three traffic share 76% → 47%. **[solid]**
8. **The chasing response is close to linear and close to diagonal across categories** —
   no detectable curvature in gap→improvement, and richer cross-category response models
   buy only marginal predictive power over the simple scalar law. **[solid null / moderate]**
9. **Innovation residuals are not persistent (rho ≈ 0), but that does not resolve the
   steady-state over-prediction** from §3.3 — ruling out "transitory innovation
   mislabeled as permanent" as the explanation, leaving a genuine missing homogenizing
   force as the live hypothesis. Over-prediction ratio is 2.56× (not the earlier 3.4×,
   see #11). **[moderate]**
10. **Firm heterogeneity in chasing strength is borderline** — a permutation test gives
   p = 0.036 (OpenAI vs Anthropic driving it), but the variance-component estimate itself
   is indistinguishable from zero. **[soft]**
11. **A duplicate-event-label bug was found and fixed** (`common.py`'s `build_events`, see
   §3.4) that silently pooled 8 of 65 release episodes. Fixed throughout this document as
   of 2026-08-10: beta moves 0.4265→0.5495, the over-prediction ratio 3.40×→2.56×, and the
   early/late trend in §3.1 reverses direction. No numbers in this document predate the
   fix.
12. **The missing force is concentrated at release events, not continuous or artifactual**
   — ruled out rating-scale drift (unchanged models barely move, rho=0.897/28d vs 0.390 at
   releases) and a ceiling effect (dispersion rises, not falls, with capability level).
   Points toward technique/recipe diffusion as the remaining candidate, untested for lack
   of data. **[solid]**
13. **A continuous version of the Theorem-9 test is inconclusive**, and more importantly
   Arena vote-share looks like the wrong instrument for evaluator-chosen w (it tracks
   organic traffic classification, not board decisions) — the discrete category-launch
   design in §5.3 remains the right operationalization. **[soft]**
14. **Historical (c,w) variation is empirically ~2-dimensional**, well below both the
   nominal 8 categories and the structural 7-dimensional cap — categories' attention
   shares move together, not independently (permutation p=0.001). This bounds how much of
   Theorem 9's identification claim the historical record can actually deliver. **[solid]**
15. **No 1- or 2-parameter structural F_i improves held-out prediction, in any of three
   tested bases.** An initial pass suggested a small gain from a firm-specific productivity
   scalar, but that was an artifact of a weak shared baseline (rank-2 truncation) — against
   the best pooled model (the full, untruncated shared response matrix), CV selects
   infinite shrinkage on every firm-specific term tried. Firm identity buys nothing
   recoverable at this sample size (65 events). **[solid null]**
16. **The open-vs-closed gap-chasing split does not survive a formal test.** Exact
   permutation over all 35 possible 4-vs-4 firm partitions gives p=0.771; the actual
   open/closed split ranks 27th of 35 by effect size — indistinguishable from an arbitrary
   partition. **Drop this framing.** **[solid null]**
17. **Semi-synthetic replication of board-splitting confirms the direction but shrinks the
   effect.** Starting from real capability profiles, the real estimated response law, and
   bootstrap-resampled real release residuals (not the stylized Gaussian toy of §6),
   splitting into 2–4 boards buys **1.09–1.35× steady-state diversity** vs one shared board
   — smaller than §6's stylized 1.4–2.1×, same direction. **[moderate]**
18. **A synthetic identification test with KNOWN ground-truth F_i shows scale is
   recoverable, direction is not — at any rank tested.** Detection that firms differ at
   all improves with w-shock rank (1→4), consistent with Theorem 9. But cosine similarity
   between the recovered and true heterogeneity *direction* stays ≈0 at every rank 1–7,
   because the estimator only spans a 2-dimensional subspace of the true 64-dimensional
   response matrix. More independent evaluator perturbations help detect *that* F_i
   differs, not *how*, unless the estimator itself has more capacity. **[moderate]**

---

## 1. Data assets

| asset | file | coverage |
|---|---|---|
| Arena per-category leaderboard | `data/leaderboard/text/full-*.parquet` | 215 snapshots, 2023-05-08 → 2026-05-27, 368 models, 60 orgs, 26 categories, with BT `variance` and `vote_count` (100% populated from 2024) |
| Style-controlled ratings | `data/style_control/text_style_control/full-*.parquet` | 155 snapshots, 2024-08-28 → 2026-05-27, same schema |
| Open LLM per-subtask results | `data/ollm_subtasks.parquet` | **4,499 models × 40 leaf subtasks** (24 BBH, 7 MATH, 3 GPQA, 3 MuSR, IFEval, MMLU-Pro, ARC) |
| Model metadata | `data/benchmarks.parquet` | 3,861 models, parameter counts, upload dates, 6 aggregate scores |
| Training compute | `data/epoch_notable_models.csv` | 1,043 models, 534 with training FLOP, 665 with dataset size |
| Deployment traffic | `data/openrouter_daily.parquet` | 521 days, 2025-01-01 → 2026-06-05, 347 models, 39 orgs |
| Raw battles | `data/140k/data/train-*.parquet` | 2025-04-17 → 2025-07-24 only; tags limited to creative_writing / if / math / criteria |

**Note.** `data/arena_leaderboard.parquet` is overall-rating only — do not use it for any
profile work. The per-category panel is the workhorse.

**Acquisition notes.** The Open LLM subtask data lives in ~4,500 gated
`open-llm-leaderboard/<org>__<model>-details` HF repos (`gated="auto"`, per-repo access).
HF rate-limits at **1,000 API requests per 5 minutes**; the metadata call is the binding
constraint, so throttle to ~3/s (`experiments/pull_ollm_subtasks_resume.py`). Epoch AI's
`notable_ai_models.csv` downloads freely with no auth.

---

## 2. Part I — Measurement

Everything downstream depends on this section. Both natural estimands are broken, in
different ways, and both break toward finding convergence.

### 2.1 Raw cosine is saturated

Computing the paper's H = cos(θ_i, θ_i') directly on per-category BT ratings gives
**0.9999 in every quarter, slope +0.0000** across 2024-06 → 2026-05.

Cause: rating vectors are all-positive with entries near 1300, so they all point close to
the diagonal. Two components dominate and neither is strategic — a global quality level,
and a common category-difficulty profile. Cosine on raw θ answers "are these both LLM
rating vectors."

**Implication for the model.** H cannot be a statement about levels. It has to be about
profile *direction* after common components are removed — which is consistent with
Theorem 9 identifying F only up to a pointwise scale λ(x).

*Script: `experiments/exp0_measurement.py`*

### 2.2 The two-way fixed-effect correction is algebraically pinned

The standard fix — remove a firm effect and a category effect, then take cosines on the
residuals — forces both row sums and column sums to zero. Since Σᵢrᵢ = 0,

```
0 = ‖Σᵢ rᵢ‖² = Σᵢ‖rᵢ‖² + Σ_{i≠j} rᵢ·rⱼ    ⟹    mean off-diagonal cosine ≈ −1/(N−1)
```

With N = 8 frontier firms that is −0.143. **Observed: −0.105 to −0.139.** Against a
permutation null (200 draws, reshuffling residuals within each category), excess similarity
is **+0.010, z = 1.27, not significant, no trend**.

So a small or negative residual H is arithmetic, not evidence. **Any such statistic must be
reported as excess over the permutation null.**

Substantively: once corrected, frontier firms' profiles are statistically indistinguishable
from a random reshuffling of who is good at what, and have been for two years.

*Script: `experiments/exp0b_estimand_null.py`*

### 2.3 The noise bias runs toward spurious divergence

BT ratings are estimates, and estimation error inflates apparent dispersion. Sampling noise
is **27–30% of cross-firm dispersion and rising** (mean BT variance 22 → 56), because votes
per model are *falling* (median 7,832 → 5,741) as the model set grows faster than battle
volume.

The usual concern is that accumulating votes shrink noise and manufacture spurious
convergence. Here it runs the other way: noise grows, so the artifact would be spurious
*divergence*. Correcting costs nothing since `variance` is published per observation.

### 2.4 Effective dimensionality of the instruments

Effective rank measured as a participation ratio, (Σλ)²/Σλ², each benchmarked against its
own single-latent-factor-plus-binomial-noise null. Task-demeaned convention.

**On the full 4,499-model population:**

| battery | nominal | effective rank | 1-factor null | **real excess** | top factor |
|---|---|---|---|---|---|
| 6 family aggregates | 6 | 1.80 | 1.07 | **+0.73** | 73.0% |
| 24 BBH leaves | 24 | 2.17 | 1.06 | **+1.11** | 67.0% |
| 39 leaves (all) | 39 | 2.52 | 1.09 | **+1.44** | 61.8% |

Arena's 8 categories, measured separately, give **3.81** effective dimensions against a
pure-noise null of 6.81; the top 3 of 8 components carry 80% of residual variance.

**Finer measurement does buy real structure, at a stable exchange rate:** excess rank
roughly doubles for 6.5× the nominal resolution (ratio 1.97 on the full population, 2.06 on
a 1,288-model subsample).

**Important caveat — effective rank is population-dependent.** On the smaller, higher-quality
subsample the same batteries gave 2.07 / 2.59 / 3.13 with a 54.9% top factor. The full
population includes many more hobbyist fine-tunes spanning a wider quality range, which
strengthens the general factor. *Quote the ratio, not the level, and always state which
model population a level refers to.* A frontier-only board and an all-comers board have
different resolving power over the same capability space.

Slicing the *same* battles more finely does not help: Arena's industry categories give a
signal-to-noise of 1.79 against 2.54 for the coarse core set. The coarseness is in what
pairwise human preference resolves, not in the labels.

The two-way (general-ability-removed) convention gives an effective rank of 4.47, below
the matched one-factor-plus-noise null of 6.03. Thus the extra structure left after
removing general ability is not clearly more than measurement noise. The tasks that add
the most distinct information are tracking shuffled objects, MuSR team allocation, web
of lies, and object placement; several algebra and number-theory tasks are highly
redundant.

*Scripts: `experiments/expC_effective_K.py`, `expJ3_matched.py`, `expJ2_robust.py`*

---

## 3. Part II — The force

### 3.1 Eval-chasing **[solid]**

An **event** is a firm's flagship changeover. For each event *e* and category *k*:

- `dtheta` = new flagship's rating − outgoing flagship's rating
- `gap` = (max over rivals' flagships in category k) − outgoing flagship's rating

Regress `dtheta` on `gap` with **event** and **category** fixed effects. β > 0 means firms
improve most where they lagged the shared board (eval-chasing); β < 0 means they deepen
existing strengths (comparative advantage).

**65 events, 8 firms, 520 observations** (corrected from 57/65 after the §3.4 event-label
fix — the current numbers below supersede the earlier 0.4265/3.4× figures):

| | β | |
|---|---|---|
| naive | **+0.5362** | cluster-robust SE 0.0846, t = +6.34 |
| noise-corrected | **+0.5523** | |
| early half | +0.624 | 32 events |
| late half | +0.516 | 33 events |

**Firms close ~55% of their per-category gap to rivals' frontier with each flagship
release.** The early/late split now runs the *opposite* direction from the pre-fix
result (which showed strengthening 0.34→0.52): chasing is flat-to-mildly-declining,
0.62 early vs 0.52 late. The previously reported "pull is strengthening over time" claim
does not survive the fix and should be dropped, not merely re-stated with new numbers.

*Confound handled:* `gap` contains −θ_old and `dtheta` also contains −θ_old, so measurement
error in θ_old inflates their covariance — pure mean reversion would give β > 0 with nothing
real behind it. Using the published BT sampling variances, that bias is subtracted
analytically from both the covariance and the variance.

*Script: `experiments/exp1_evalchasing.py`*

### 3.1b Heterogeneous gap-chasing **[exploratory]**

The pooled β hides substantial firm-level variation, but the sample is small for
several firms. Re-estimating the same event/category fixed-effects slope within each
firm gives (post-fix, 65 events):

| firm | events | corrected β |
|---|---:|---:|
| Alibaba | 11 | 1.39 |
| OpenAI | 14 | 1.36 |
| xAI | 5 | 1.28 |
| Mistral | 3 | 1.02 |
| Anthropic | 18 | 0.69 |
| Google | 5 | 0.50 |
| DeepSeek | 7 | 0.48 |
| Meta | 2 | −0.72 |

Meta's negative estimate is based on only two events and is not informative. See §3.7 for
a formal test of whether these firm differences are real: a permutation test gives
p = 0.036, borderline evidence, concentrated in the OpenAI/Anthropic contrast specifically
rather than a general ranking across all eight firms — treat the table above as
descriptive, not as validated per-firm estimates.

Grouping firms as open-weight (Meta, Mistral, DeepSeek, Alibaba) versus closed (OpenAI,
Anthropic, Google, xAI) gives a visible descriptive gap that the pre-fix run did not show:
0.61 closed versus 0.43 open after the sampling-noise correction (was reported as "nearly
identical, 0.42 vs 0.45" before the fix). **§7.5 formally tests this and it does not
survive**: exact permutation over all 35 possible 4-vs-4 firm partitions gives p=0.771,
and the actual open/closed split ranks 27th of 35 by effect size. Not distinguishable from
an arbitrary partition of the 8 firms — drop the open/closed framing.

Across categories, the interaction estimates are all positive: math is now the strongest
response (0.67), followed by English (0.58) and multi-turn (0.53); instruction-following
remains weakest (0.41). Firm-by-category slopes are too noisy to support strong
firm-specific claims: most cells contain only 2–18 events.

Model-size heterogeneity cannot be estimated cleanly for this panel: closed-lab parameter
counts are unavailable, and only 8 open events have name-derived size labels (median
235B). The small/large split (0.75 vs 0.08, corrected) remains far too small to treat as
evidence.

*Script: `experiments/exp1_heterogeneous.py`*

### 3.2 Imitation is a separate channel **[moderate]**

Eval-chasing predicts firms move toward the *per-category* leader. Distillation-style
imitation predicts they move toward *one model's whole profile*, including where they
already lead it.

The naive horse race is uninformative — gap-to-category-leader and gap-to-best-overall-rival
correlate at **0.981**, and the corrected coefficients flip sign. Not reported as a finding.

Orthogonalised (frontier excluding the teacher vs. the teacher), both channels are
significant at 1%: board-following **+0.282**, teacher-following **+0.184**.

The sharp test:

| | teacher pull (corrected) |
|---|---|
| firm already **ahead** of teacher | **+0.350** |
| firm **behind** teacher | **+0.349** |

Eval-chasing predicts the pull should vanish once a firm leads the teacher. It doesn't —
the two are statistically identical. That is imitation, not board response.

It does **not** account for the missing homogenizing force in §3.3: adding the teacher term
reduces residual variance only from 272.0 to 270.8.

*Script: `experiments/expB_distillation.py`*

### 3.3 The steady-state accounting over-predicts diversity **[moderate]**

Partial adjustment toward the frontier plus idiosyncratic innovation implies cross-firm
dispersion evolves as D′ = (1−β)²D + σ², with steady state D* = σ²/(1−(1−β)²).

With β = 0.5495 and σ²_innov = 208.3 (residual variance from the §3.1 regression, minus the
sampling noise it inherits — post-fix numbers, see §3.4):

- **predicted D\* = 261.3**
- **observed (noise-corrected) = 102.0**
- **ratio = 2.56** — the model predicts *more* diversity than exists

An independent check that doesn't go through the formula: measure directly how fast a firm's
profile deviation decays between releases. Gap-chasing alone predicts persistence of
1−β = 0.451. **Measured: 0.390** (noise-corrected; 0.204 naive). Deviations still die a bit
faster than eval-chasing explains, though the gap between measured and predicted persistence
is narrower post-fix than it first appeared (0.390 vs 0.451, compared to the earlier 0.339 vs
0.574).

Implied half-life of a profile deviation: 0.87 releases ≈ 0.8 months, at the observed median
release spacing of 28 days (down from the earlier 45-day estimate, now that the previously
merged episodes are counted separately).

**§3.8 resolves the open question this section originally posed** (unmodelled force vs.
transitory innovation): measured persistence of the innovation residual itself is ρ≈−0.16,
i.e. innovation already looks transitory, and feeding that back into the model barely moves
the prediction (D\*=278 vs D\*=261). **The 2.56× over-prediction is not a dynamic-model
misspecification about persistence — it points to an actual missing homogenizing force.**

*Script: `experiments/expA_steady_state.py`*

### 3.4 Data-quality note: duplicate event labels (fixed) **[housekeeping]**

While building the multivariate follow-ups below, found that `build_events`'s event id
(`firm:old_model->new_model`) was **not unique**: 8 of the panel's 65 true release episodes
shared a label with another release, because a firm's flagship sometimes toggles back and
forth between the same two model names at different dates (e.g. a thinking-mode variant
launched, reverted, and relaunched). `groupby('event')` / `pivot_table(index='event')` then
silently pooled two independent episodes into one "event" for fixed-effects purposes.

**Fixed**: `common.py`'s `build_events` now appends the switch date to the event id, and
`exp1_evalchasing.py` / `exp1_heterogeneous.py` (which build events inline rather than via
`common.py`) got the same fix. All numbers in §3.1, §3.1b, §3.3, and §3.5–3.8 in this
document are now on the de-duplicated 65-episode panel. The change moved the noise-corrected
beta from 0.4265 to 0.5495 and the steady-state over-prediction ratio from 3.40× to 2.56× —
real, not cosmetic, and it also flipped the early/late trend direction reported in §3.1 (was
strengthening, is now flat-to-declining). Anything citing the old 0.4265/3.40×/57-event
figures from before 2026-08-10 is stale.

### 3.5 Multivariate gap-response matrix (Exp K) **[moderate]**

Does a firm close each category's gap independently (B = beta·I), or respond to gaps in
one category with improvement in another? Fit dtheta_e = B·gap_e per release episode
(65 complete 8-category vectors), comparing specs by leave-one-episode-out CV R²:

| spec | params | CV R^2 |
|---|---:|---:|
| scalar, B=beta*I | 1 | 0.151 |
| common, B=beta*I + gamma*(11^T-I) | 2 | 0.156 |
| rank-1 (ridge+SVD) | ~15 | 0.144 |
| rank-2 | ~29 | 0.117 |
| rank-3 | ~43 | 0.131 |
| unrestricted ridge | 64 | 0.163 |

The 2-parameter common-spillover model is the best of the restricted specs and nearly
matches the fully unrestricted, heavily-shrunk 64-parameter matrix; low-rank truncations
do not beat the scalar model. On the full-sample ridge-regularized B, off-diagonal entries
are not negligible in magnitude (mean |off-diag| 0.22 vs mean diagonal 0.47, Frobenius
ratio 1.36 — e.g. gaps in `instruction_following` predict *negative* improvement of
−0.5 to −0.85 in most other categories), but they do not buy reliable out-of-sample
predictive power at n=65. **Read this as: cannot reject B ≈ beta·I; the off-diagonal
structure in the point estimate is suggestive but not established.**

*Script: `experiments/expK_gap_matrix.py`*

### 3.6 Nonlinear shape of gap-chasing (Exp N) **[solid null]**

Testing whether dtheta = f(gap) is linear (as assumed throughout) against three
alternatives, all nested against or compared to the linear fit (R^2 = 0.279) within the
same event+category FE structure:

| test | result |
|---|---|
| linear + gap-quartile dummies vs linear alone | F=0.18, **p=0.911** |
| kinked spline (separate slope ahead/behind the FE-adjusted mean) | F=0.01, **p=0.943** |
| quadratic term | F=0.71, **p=0.401** |

None detect curvature. The response looks linear across the full observed gap range
(−73 to +180 Arena rating points). Category-level kink slopes vary descriptively (0.22 for
instruction_following to 0.93 for multi_turn on the ahead-side) but that is the same
category heterogeneity already reported in §3.1b, not evidence against linearity itself.
**Supports treating the simple partial-adjustment law as an adequate approximation.**

*Script: `experiments/expN_nonlinear.py`*

### 3.7 Is firm heterogeneity in chasing strength real? (Exp M) **[borderline]**

§3.1b's per-firm betas were descriptive with no formal test. Here: two-way (event+category)
FE beta_i per firm, with event-resampling bootstrap SEs (needed because firms have 2–18
episodes and small-sample analytic SEs are unreliable):

| firm | episodes | beta_i | bootstrap SE | 95% CI |
|---|---:|---:|---:|---|
| OpenAI | 14 | 1.36 | 0.24 | [0.99, 1.95] |
| Anthropic | 18 | 0.69 | 0.22 | [0.25, 0.96] |
| Alibaba | 11 | 1.39 | 2.88 | [0.59, 4.04] |
| xAI | 5 | 1.18 | 2.22 | [−0.58, 3.29] |
| Google | 5 | 0.48 | 1.54 | [−0.02, 2.07] |
| DeepSeek | 7 | 0.48 | 4.68 | [−0.41, 1.32] |
| Mistral | 3 | 1.02 | 4.83 | [−1.93, 14.76] |
| Meta | 2 | — | too few episodes | — |

Inverse-variance-weighted pooled beta = **0.99** (dominated by OpenAI and Anthropic, the
two precisely-estimated firms — not comparable to §3.1's pooled 0.55, which runs on the
buggy event grouping; see §3.4). The DerSimonian-Laird tau² estimator clips to 0
(Cochran's Q = 4.30 on df = 6, below its own asymptotic expectation), which would normally
read as "no heterogeneity." But a permutation test that reshuffles which firm owns each
release episode (not relying on the chi²  asymptotics) gives **p = 0.036** — mild evidence
against a common beta. Leave-one-firm-out shows this isn't one outlier: Q stays at 4.2–4.3
dropping any firm except OpenAI or Anthropic themselves (whose removal collapses Q to
0.13–0.36, confirming they're the source of the signal, not an artifact of the low-precision
firms). Restricting to firms with ≥5 episodes gives the same Q=4.30.

**Read as weak, borderline evidence that OpenAI and Anthropic chase gaps at genuinely
different rates from each other — not strong enough to support a general claim of
firm-specific production functions, and not strong enough to rank the other five firms.**

*Script: `experiments/expM_heterogeneity.py`*

### 3.8 Does persistence resolve §3.3's steady-state discrepancy? (Exp L) **[answers the open question — negatively]**

§3.3 originally left open whether the over-prediction of steady-state dispersion reflects a
missing homogenizing force or merely treating transitory innovation as if it were
permanent. Testing directly: take the same event+category-FE residual u (sigma²_innov =
208.3, matches §3.3), and track each (firm, category)'s residual into that firm's *next*
release.

**Measured persistence rho = −0.164** (noise-corrected; naive −0.141) — essentially zero,
mildly negative. Not asymmetric (ahead −0.12 vs behind −0.21, overlapping) and not
obviously driven by release spacing. So innovation is not "labeled persistent when it's
actually transitory" — **it already looks transitory, and the model already assumed that**
(the iid-innovation steady-state formula implicitly assumes rho = 0).

Feeding the measured rho back into an AR(1)-innovation version of the steady-state model
barely moves the prediction: D* = 261 (iid, rho=0) vs D* = 278 (measured rho=−0.164)
against observed D = 102 (noise-corrected) — ratio 2.56x vs 2.73x.

**Conclusion: transitory innovation was already the working assumption and is not the
missing piece. The over-prediction requires an actual missing homogenizing force**, not a
dynamic-model misspecification about persistence.

*Script: `experiments/expL_persistence.py`*

---

## 4. Part III — The outcome

### 4.1 No convergence in capability profiles **[solid]**

See §2.2. Excess similarity over the permutation null is +0.010 (z = 1.27, ns) with no
trend across 2024-06 → 2026-05.

### 4.2 Deployment is fragmenting **[solid]**

OpenRouter token shares, Jan 2025 → Jun 2026:

| | start | end | trend |
|---|---|---|---|
| effective number of **firms** | 3.25 | **9.19** | +4.9/yr, r = +0.93 |
| effective number of **models** | 7.2 | 22.3 | +13.0/yr, r = +0.85 |
| top-3 firm token share | 76% | **47%** | −0.29/yr, r = −0.92 |

This is the most legible monoculture measure available and it moves decisively the wrong
way for a consolidation narrative.

*Caveat:* OpenRouter skews toward developer and long-tail traffic and misses direct
enterprise API usage. Suggestive of the broader market, not authoritative.

### 4.3 Welfare: evaluation attention vs. actual usage **[soft]**

KL(v ‖ w) with v = OpenRouter token share, w = Arena vote share, at model granularity:
**mean KL = 1.26 nats, total-variation distance 0.58**, mildly declining (−0.46/yr,
r = −0.43).

*Blocker:* the slug-to-Arena name matcher reaches only 113 of 347 models = **29.7% of
tokens**, because OpenRouter appends release dates to slugs
(`anthropic/claude-4.6-sonnet-20260217`). The number is real but computed on a biased,
older subset. Fixing the matcher is a few hours of string handling and is a prerequisite
for taking this seriously.

### 4.4 Style is converging too — prediction failed **[soft]**

The theory suggested firms would differentiate on style once measurable dimensions
converged. Using (raw − style-controlled) ratings as the style measure:

| | slope/yr | r |
|---|---|---|
| style level dispersion | −245.7 | −0.757 |
| style profile dispersion | −6.3 | −0.238 |
| objective dispersion (SC) | −125.5 | −0.766 |

Style dispersion is **falling**, not rising. §5.2 as written doesn't have support.

*Caveats:* the usable raw/SC overlap window is only 2025-06 onward (99 weekly snapshots,
heavily autocorrelated, so the t-statistics are optimistic), and the recent-window objective
decline contrasts with the flat trend over the longer 2024-06 window. Something changed in
the last year that is worth a separate look.

*Script: `experiments/expE_style.py`*

---

## 5. Part IV — Technology

### 5.1 Comparing production functions via scaling **[moderate]**

This is the one setting where investment is *observed*. For open-weight labs we see
parameter counts, giving the paper's log family (Corollary 6) a level term:

```
theta_{i,k} = a_{i,k} + A_{i,k} · log(params)      controls: instruct-tuning, release date
```

**A_i is firm i's technology direction** — how it converts scale into capability across
dimensions — and Theorem 9's conclusion is exactly that the A_i are parallel.

**Result at 39 dimensions, 11 labs (269 eligible models from the 4,499-model panel):**

| | 6 aggregate dims | **39 leaf dims** |
|---|---|---|
| direction test | p = 0.75 | **p = 0.0500** |
| mean pairwise cosine | +0.920 | +0.822 |
| share along common direction | 0.929 | 0.837 |

The broad six-dimension result is robust to adding release-date controls: mean cosine is
0.941 without the control and 0.920 with it, with the common direction explaining about
93–95% of variation. At 39 dimensions, the calibrated residual-direction test is exactly
at the conventional 5% cutoff, but the pairwise-cosine test gives p = 0.202. The cautious
conclusion is that labs mostly share a scaling direction, with suggestive evidence of
departures rather than an established general technology difference.

The apparent departures are concentrated in two labs:

| lab | z vs own estimation noise |
|---|---|
| allenai | **+3.95** |
| qwen | **+3.59** |
| ibm-granite | +1.51 |
| all others | < 1.0 |

This is evidence against treating the six aggregate dimensions as a complete test, but it
is not strong enough to claim that open labs generally have distinct technologies. The
result is also limited to open-weight labs and cannot separate architecture, data mixture,
and post-training choices.

**Methodological trap, worth a paragraph in the paper.** The null *must* be built from each
lab's own bootstrap deviations, which carry the correct error covariance. An
independent-per-coordinate null over 39 correlated tasks is badly mis-calibrated and
returned a degenerate p = 1.0 on the first attempt.

*Scripts: `experiments/expG2_robust.py`, `expG5_direction_test.py`*

### 5.2 Compute: allocation differs even where technology looks shared **[moderate]**

‖A‖ from §5.1 is capability per log-*parameter*, which ignores training tokens. With
C = 6ND and empirically log D = b_i·log N + const, we get log C = (1+b_i)·log N + const, so
per-FLOP efficiency is A/(1+b_i).

The token-scaling exponent varies sharply by lab (Epoch AI data):

| lab | b_i |
|---|---|
| google | 0.29 |
| nvidia | 0.38 |
| microsoft | 0.46 |
| qwen | 0.69 |
| meta | 0.81 |
| **deepseek** | **1.47** |

This **re-ranks** rather than merely rescales: DeepSeek falls from 3rd to last on per-FLOP
efficiency, because it looks efficient per parameter only by training on far more tokens
per parameter. Spearman between the two rankings among measured labs: 0.543.

**Interpretation that matters for the paper:** b_i is an *allocation* choice — how a lab
splits a scale increment between parameters and data — sitting underneath what a
parameter-only analysis reads as technology. That is the paper's own allocation-vs-technology
distinction appearing in the training recipe.

*Caveat:* only 6 labs have a genuinely measured b_i. The rest use the pooled 0.64, so their
per-FLOP values are a constant rescaling and their rank shifts carry no information —
**do not report them as findings.**

*Script: `experiments/expG6_compute.py`*

### 5.3 The (c, w) perturbation test on Arena **[soft]**

Theorem 9's own design: category launches are discrete changes in w, and firms with parallel
F should respond identically. Since no rating exists for a category before it launches, the
response is measured on control categories with full pre-periods.

Against 56 placebo windows (mean 99.6, sd 39.6):

| shock | date | z | p |
|---|---|---|---|
| multi_turn | 2024-06-29 | +2.53 | 0.018 |
| math + instruction_following | 2024-07-08 | +2.66 | 0.018 |
| creative_writing | 2024-10-23 | −0.07 | 0.60 |
| industry + expert (8 cats) | 2025-11-05 | −1.25 | 0.84 |

**The two significant rows are 9 days apart with overlapping windows — that is one episode,
not two.** So: one of four distinct shocks shows heterogeneous response. Treat as
suggestive at best. §5.1 is the far stronger test of the same proposition.

*Script: `experiments/expD_theorem9.py`*

---

## 6. Part V — Design levers, calibrated

Reduced-form simulation calibrated on §3.1 and §3.3 (β = 0.4265, σ² = 232.9), sweeping the
§5.3 recommendations. "Effective dims" is the diversity analogue of §2.4's rank measure.

**More scored categories.** Attention per category falls as 1/K, so per-category chasing
weakens and diversity rises — K = 32 gives 3.7× the per-category dispersion of K = 8. **But**
under a rank-4 cap matching the measured instrument, effective diversity dimensions saturate
at ~3.6 regardless of K:

| K | eff. dims, uncapped | eff. dims, rank-4 cap |
|---|---|---|
| 8 | 6.94 | 2.93 |
| 16 | 14.45 | 3.39 |
| 32 | 26.85 | 3.60 |

**Declaring more categories over the same battle pool cannot buy diversity the pool cannot
express.** §2.4 says finer measurement *does* help, at ~2× per 6.5× resolution — so the
lever works only if the prompt distribution genuinely widens with the categories, which is a
harder intervention than adding leaderboard tabs.

**Splitting leaderboards works.** L = 4 boards buys 1.4–2.1× diversity at realistic
anchoring (firms devoting 50–75% of attention to their own board). The fully unanchored
s = 1.0 case explodes to 137×, but that is a random walk with nothing pinning cross-board
levels — an artifact, not a result.

**Concentrating w is a trap for the metric.** Going from uniform to a steep weight profile
raises raw dispersion 7.6× while *cutting* effective diversity dimensions from 6.94 to 3.51.
Concentration buys unmonitored drift in the categories nobody scores, not diversity. **Raw
dispersion is the wrong statistic to report here** and this is worth saying explicitly.

*Script: `experiments/expF_design_levers.py`*

### 6.1 Semi-synthetic replication with real state and real residuals (Exp T) **[moderate]**

§6 is a fully stylized simulation: Gaussian innovations, no real firm identity, frontier
defined as a simple max. This upgrades it to semi-synthetic, closing the gap the design
section otherwise leaves open: initial capability profiles θ_i⁰ are the real 8-category
Arena flagship snapshot (2026-05-27, all 8 frontier firms present); the response law is
the real fitted β (0.354, through-origin OLS on the raw gap→Δθ events — a different
convention from §3.1's FE-demeaned 0.5495, noted for consistency) or the real pooled
ridge matrix B as a robustness check; and the innovation term at every simulated release is
a bootstrap draw from the actual 65 real per-event residual vectors, preserving their real
cross-category correlation structure rather than drawing iid Gaussian noise.

L ∈ {1, 2, 4} boards partition the 8 categories into L attention blocks; s ∈ [0,1] mixes a
uniform common weight vector with each board's own-block-weighted vector, and the relevant
frontier for firm i is computed only among rivals sharing its board (so s moves both "what
is measured" and "who you're compared to" together). After 15 simulated releases, 300
Monte Carlo reps per cell:

| L | s | D_T ratio vs L=1 (scalar B) | D_T ratio (ridge B) |
|---|---|---|---|
| 2 | 0.00 | 1.09× | 1.08× |
| 2 | 0.50 | 1.12× | 1.14× |
| 2 | 1.00 | 1.20× | 1.35× |
| 4 | 0.00 | 1.22× | 1.16× |
| 4 | 0.50 | 1.23× | 1.20× |
| 4 | 1.00 | 1.27× | 1.28× |

**Splitting boards still inflates steady-state diversity, and the ridge-B robustness check
agrees in direction and rough magnitude with the scalar law** — but the effect is
substantially smaller once grounded in the real initial dispersion and real noise scale
(1.09–1.35×) than §6's stylized 1.4–2.1×. Effective rank (participation ratio of the
cross-firm covariance) and the count of distinct per-category leaders both move only
slightly (effective rank 1.7→1.8, leaders 2.6→2.7) — raw pairwise cosine stays saturated
at 0.999 throughout, consistent with §2.1's finding that raw cosine is mechanically
uninformative here too, so D_T and effective rank are the metrics worth reading, not raw
cosine similarity.

**L = 8 (singleton boards) is not a valid comparison point.** With one firm per board there
are never any rivals, so the update degenerates to a pure unanchored random walk — no
restoring force pulls firms back toward anyone, and D_T grows without bound in T rather
than reaching a steady state. Its raw ratio (1.96× at T=15) is an artifact of the
simulation horizon, not a genuine "more boards is better" result, mirroring §6's own
caveat about the fully-unanchored s=1.0 case exploding to 137×.

*Script: `experiments/expT_semisynth_boards.py`*

---

## 7. Part VI — The missing force, continuous attention, and identification rank

Follow-up on the three priorities from §11 (2026-08-10): find the missing homogenizing
force behind §3.3/§3.8's 2.56× over-prediction, test a continuous version of Theorem 9's
w-perturbation design, and measure the empirical rank of that perturbation.

### 7.1 The missing force is concentrated at releases, not continuous or artifactual **[solid]**

Three candidates tested against the panel:

**(a) Rating-scale / voter-population drift.** If part of "homogenization" were a
measurement artifact (ratings drifting as the voter/prompt population shifts, independent
of any real model change), a literally unchanged flagship model should show the same
decay toward the pack as a released one. Using 14,805 pairs of snapshots within
unchanged-model spells (up to 363 days apart) and fitting an exponential decay in the
per-category two-way residual:

| lag | rho (naive) |
|---|---:|
| 0–7d | 0.968 |
| 7–14d | 0.911 |
| 14–30d | 0.871 |
| 30–60d | 0.817 |
| 60–120d | 0.774 |
| 120d+ | 0.504 |

Fitted decay implies **rho = 0.897 over a 28-day window** (matched to the median release
spacing) for a model that never changed — against **0.390 measured at actual release
events** (§3.3). Unchanged models barely move. **Rating-scale drift is not the missing
force; convergence is tied to real model updates, not a measurement artifact.**

**(b) Mechanical/ceiling compression.** Does dispersion fall simply because mean
capability rose (a bounded rating scale compressing differences near the top)? Partial
correlation of dispersion D_t against mean capability level, controlling for the calendar
time trend: **+0.360** — dispersion *rises* with capability net of trend, the opposite of
a ceiling story. **Not a ceiling effect.**

**(c) Continuous vs. at-event convergence.** Directly compares (a)'s ambient decay
(0.897 per 28 days, unchanged models) to the at-release decay (0.390). The gap is large:
convergence is not a slow background process happening at a similar rate whether or not a
firm releases — **it is concentrated specifically at release events.**

**Taken together, this narrows the missing force considerably.** It is not rating drift,
not a ceiling effect, and not a continuous process — it acts through whatever firms do
when they ship a new flagship. Combined with §3.2 (teacher-specific output imitation
barely reduces residual variance), the remaining candidate is **imitation at the level of
technique/recipe rather than output**: firms adopting widely-known training tricks
(reasoning-RL, specific post-training recipes, shared datasets) that become common
knowledge across the field between releases and get incorporated at the next release,
regardless of which specific rival is ahead. This is consistent with, but not proof of,
the diffusion story — it is not directly testable with the data on hand (no recipe/
technique metadata), and is the natural next step if such data becomes available (e.g.
coding technical reports for keyword co-occurrence of technique adoption timing).

*Script: `experiments/expO_missing_force.py`*

### 7.2 Continuous w-attention response **[inconclusive — likely a measure-validity problem]**

Extends §5.3's Theorem-9 test (category launches, one informative episode) using
continuous variation: `w_{k,m}` = new Arena votes recorded in category k during calendar
month m (vote_count increments, floored at 0 to handle occasional resets), as a share of
total votes that month. Outcome: change in the mean FRONTIER flagship rating in category k
from month m to m+1. Category + month fixed effects, 25 months, 185 category-month
observations.

| test | coefficient | |
|---|---:|---|
| forward: w_share(m) → Δθ̄(m→m+1) | **−3.74** | cluster SE 2.60, t = −1.44 |
| forward, lag 2 months | −12.26 | |
| forward, lag 3 months | −10.93 | |
| **placebo**, reverse timing: Δθ̄(m−1→m) → w_share(m) | **+3.91** | similar magnitude, opposite sign |

No robust forward effect, and the reverse-timing placebo is comparable in size and
opposite in sign — not the pattern a clean causal effect would produce. **More important
than the null result: this is likely the wrong instrument.** Attention share is heavily
concentrated in two catch-all categories (`longer_query` 34%, `english` 20%) against
specialist categories like `math` (3%) — this reflects Arena's automatic classification of
*organically occurring* user traffic into categories, not a deliberate evaluator choice to
emphasize category k. Theorem 9's w is a supply-side weighting decision by the evaluator;
vote-share growth is closer to a demand-side variable (`c`, user population composition)
than to `w`. **The category-launch design in §5.3 remains the cleaner operationalization
of an actual w-shock** — a genuine board decision to start scoring something — even though
it only yields one informative episode. A better continuous w-proxy would need data on
deliberate board decisions (prompt-sourcing changes, category-weighting announcements),
which is not available here.

*Script: `experiments/expP_continuous_w.py`*

### 7.3 Empirical identification rank of the w-perturbations **[solid]**

Independent of whether §7.2's forward test is trustworthy, the *rank* question stands on
its own: how many independent directions has evaluator attention actually moved in,
historically? Using the same category × month `w_share` panel, month-to-month changes
`Δw_share` necessarily sum to 0 (shares sum to 1), capping the structural rank at K−1 = 7.
The realized effective rank (participation ratio of the change covariance's eigenvalues)
is far below even that cap:

- **observed effective rank = 2.06** of nominal 8 (structural cap 7)
- top 1/2/3 components carry 66.7% / 82.7% / 94.0% of the variance
- permutation null (categories shuffled independently in time): mean 2.72, 90% interval
  [2.45, 2.85] — **observed rank is far below the null, z = −5.07, p = 0.001**

Categories do not move independently — attention shifts co-move along roughly two shared
directions, significantly fewer than chance. This lands in the same range as §2.4's
resolving-power numbers for the *output* instrument itself (Arena's effective rank 3.81 of
8 nominal categories). **The (c,w) perturbations available to identify F in Theorem 9 are
themselves only ~2-dimensional in practice** — nowhere near the 7 nominal degrees of
freedom the category count would suggest, and on the same order as the instrument's own
resolving power. This directly limits how much of F Theorem 9's identification argument
can actually recover from the historical record, independent of any measurement problem on
the output side.

*Script: `experiments/expQ_w_rank.py`*

### 7.4 Does a 1–2 parameter structural F_i predict held-out shocks? **[solid null]**

Direct test of the §7.3 concern: fit F_i(x) = a_i·F_0(x) [+ b_i·G(x)] and ask whether it
predicts a firm's response to a held-out release event, via leave-one-event-out CV, across
three different choices of shared basis F_0:

- **rank-2 SVD basis**: isotropic response beta0=0.472 (§3.1's scalar law) plus a rank-1
  cross-category correction (u1, s1=2.134, v1) extracted from §3.5's pooled ridge B (the
  correction alone carries 77% of B's off-diagonal structure). a_i loads on the isotropic
  term, b_i on the correction.
- **full matrix basis**: a_i scales the *entire* pooled ridge B (all cross-category
  structure, not truncated to rank 1) uniformly.
- **own+spillover basis**: firm-specific (beta_i, gamma_i) on own-category gap plus a
  uniform spillover from other categories' gaps (§3.5's "common" 2-parameter family, now
  fit per firm instead of pooled).

All firm-specific parameters are ridge-shrunk toward their pooled value, shrinkage
strength chosen by nested CV:

| model | basis | firm params | CV R² |
|---|---|---:|---:|
| Model 0 | rank-2 SVD, no firm terms | 0 | 0.307 |
| Model 1 | rank-2 SVD, a_i free | 1 | 0.331 |
| Model 2 | rank-2 SVD, a_i & b_i free | 2 | 0.307 |
| **Model B** | **full ridge matrix, a_i free** | 1 | **0.363** (best at *infinite* shrinkage) |
| Model C | own+spillover, beta_i free | 1 | 0.192 (best at infinite shrinkage) |
| Model C | own+spillover, beta_i & gamma_i free | 2 | 0.192 (best at infinite shrinkage) |

**This overturns the first-pass reading.** Model 1's apparent "+0.024 gain from a_i" was an
artifact of a weak shared baseline: the rank-2 truncation throws away real shared structure
that the untruncated matrix (Model B) captures. Once given the best available *pooled*
baseline (Model B), nested CV selects **infinite shrinkage on every firm-specific
parameter tried, in every basis** — the fully pooled model always wins. **No 1- or
2-parameter firm-specific structural term improves held-out prediction in any
specification tested.** This is a cleaner and more decisive answer to §7.3's question than
the original pass gave: it is not "1 parameter is weakly identified, 2 is not" — it is that
firm identity buys nothing recoverable at this sample size (65 events) once the shared
component of the response is modeled well.

**The per-firm a_i table from the rank-2 basis (Model 1) is still worth recording as a
descriptive pattern**, even though it carries no predictive value: `xai` 2.44, `mistral`
1.74, `alibaba` 1.48, `anthropic` 0.96, `openai` 0.72, `google` 0.42, `deepseek` 0.23,
`meta` −0.05 (n=2, unreliable). Notably, OpenAI and Anthropic **swap relative position**
from §3.7's pure-scalar beta_i (OpenAI 1.36 highest of the precisely-estimated firms,
Anthropic 0.69 lower): OpenAI's high scalar beta_i looks like it's substantially explained
by the shared cross-category correction pattern (a gap in `instruction_following` or
`longer_query` predicts broad-based decline elsewhere, loading negatively on 7 of 8
categories) rather than genuine own-category catch-up, whereas Anthropic's response is
closer to pure isotropic chasing. Interesting as a description of the fitted numbers; not
a finding, since §7.4's main result is that none of this buys out-of-sample accuracy.

*Caveat, applies to all rows*: the shared basis (whichever family) is fit once on the full
sample and reused across folds — only the firm-specific scalar(s) are held out. This
affects absolute R² levels equally across all rows, so it doesn't change the ranking, but
none of these numbers are directly comparable to §3.5's fully-out-of-sample figures.

*Script: `experiments/expR_structural_Fi.py`*

### 7.5 Formal open-vs-closed chasing test **[solid null — retract the framing]**

§3.1b's post-fix descriptive split (0.61 closed vs 0.43 open) had never been tested. Fit
`dtheta = beta·gap + gamma·(gap×Closed) + event FE + category FE`, equivalently
beta_closed and beta_open on their own two-way-demeaned subsamples:

- beta_closed = 0.610, beta_open = 0.430, **gamma = +0.181**
- event-resampling bootstrap: SE(gamma) = 0.295, 95% CI [−0.415, +0.730], t ≈ 0.61
- **exact permutation test over all 35 distinct ways to split the 8 frontier firms into
  two groups of 4**: null mean |gamma| = 0.341, observed |gamma| = 0.181, **p = 0.771**

The actual open/closed partition ranks **27th of 35** possible 4-vs-4 splits by |gamma| —
one of the *least* extreme partitions available, not a special one. **Drop the open/closed
framing entirely; it is not distinguishable from an arbitrary partition of the 8 firms.**
(The most extreme split by this statistic groups {alibaba, mistral, openai, xai} vs the
rest, gamma=0.879 — a partition with no substantive interpretation, offered only to show
what the null distribution's tail looks like.)

*Script: `experiments/expS_open_closed.py`*

### 7.6 Semi-synthetic identification test: does w-rank recover F_i? (Exp U) **[moderate]**

§7.3 measured the empirical rank of historical w-variation (2.06); §7.4 found real data
can't identify a structural F_i from it. Real data can't distinguish "the historical
record has too little variation" from "the estimator can't recover F_i even given
enough" — because we never know the true F_i. This experiment builds a synthetic ground
truth to separate the two: real θ_i⁰ and the real bootstrap-resampled residual pool (same
construction as §6.1), but a KNOWN, synthetic B_i* and a w_t process whose shocks are
confined to an r-dimensional subspace of the simplex's tangent space, r ∈ {1,2,4,7}, with
TOTAL SHOCK VARIANCE HELD FIXED across r so "more rank" isn't confounded with "bigger
shocks" — exactly the confound flagged as a design requirement going in. One further
condition resamples the real historical Δw_share panel directly, rescaled to the same
total shock variance (its raw variance is ~25× the synthetic budget and, unscaled,
destabilizes the simulation — an early run before this fix produced nonsensical exploding
statistics, since fixed by matching shock energy across all conditions).

Three ground-truth families of increasing heterogeneity, recovered by the SAME
family-agnostic estimator throughout (§7.4's rank-2 SVD-basis a_i, b_i model, refit fresh
on each simulated panel — the "researcher" doesn't know which family generated the data):
homogeneous B_i*=B_0 (negative control), a_i-scale heterogeneity calibrated to the real
per-firm a_i values estimated today (range 0.29–1.40, an independent re-run of §7.4's
Model 1 under today's code), and fully heterogeneous B_i* = B_0 + random per-firm KxK
perturbation. 30 simulated releases/firm, 20 reps/cell:

| r | gain1 (detects heterogeneity exists) | Frobenius error | cosine(recovered dir, true dir) |
|---|---|---|---|
| 1 | 0.118 | 3.1 | 0.007 |
| 2 | 0.149 | 3.1 | 0.001 |
| 4 | 0.254 | 3.0 | 0.006 |
| 7 | 0.224 | 2.9 | 0.004 |
| empirical (real Δw shape, rescaled) | 0.110 | 3.1 | 0.005 |

(fully-heterogeneous ground truth; the homogeneous negative control stays flat at
gain1 ≈ 0.001 at every r — the detector is well-calibrated, not just noisy.)

**Two results, cutting in opposite directions.** (1) *Detecting that firms differ at all*
gets easier as w-rank rises from 1 to 4 (gain1 roughly doubles), matching Theorem 9's
qualitative prediction, then plateaus into r=7 within this sample's noise. The real
historical w-shock *shape* performs like a rank ~1–2 synthetic condition, independently
matching §7.3's measured 2.06. (2) *Recovering the direction* of that heterogeneity does
not improve with rank at all — cosine similarity sits at 0.00–0.12 for every r, in every
family, indistinguishable from chance alignment of a fixed direction with a 2-dimensional
subspace (≈√(2/64) = 0.18). The estimator only has two free parameters (a_i, b_i) to
represent a firm's deviation from the shared response matrix, so it structurally cannot
represent a general 64-dimensional deviation regardless of how much independent w-variation
generated it — this is an empirical instance of Theorem 9's own qualifier that (c,w)
variation identifies F **up to pointwise scale**, not full direction, unless the estimator
is given enough basis richness to hold the answer.

*Caveat*: the ground-truth heterogeneity scale (30% of ‖B_0‖/K per matrix entry) and the
total-shock-variance budget (chosen so w stays in the simplex over 30 steps without
frequent clipping) are both experimenter choices, not independently calibrated to data —
these numbers describe how recovery scales with rank *given* those choices, not an
absolute recoverability threshold.

*Script: `experiments/expU_w_rank_identification.py`*

### 7.7 Scoping note: recipe/technique-diffusion data **[not pursued — no local data]**

§7.1 narrowed the missing-force candidates to technique/recipe diffusion at release time,
but there is no local data to test it: no model cards, technical-report corpus, or
training-recipe metadata in `data/`. Testing this would require external acquisition
(scraping technical reports or blog posts for technique mentions/timing, e.g. "RL on
verifiable rewards," specific post-training methods) — a real data-collection project, not
a follow-up script. Flagged as a scoping decision, not attempted here.

---

## 8. Synthesis

Four findings that look contradictory but measure four different objects:

| finding | object |
|---|---|
| homogenizing pressure (§3.1) | the **direction firms move** — a flow |
| profiles scattered (§4.1) | **where firms stand** on relative strengths — a stock |
| style converging (§4.4) | how much score comes from **tone and formatting** |
| deployment fragmenting (§4.2) | **market share** — who gets the traffic |

Resolution:

- **Flow vs. stock.** Firms close 43% of the gap per release, but each release also throws
  off new idiosyncratic capability. Chasing drains the spread; innovation refills it. The
  result is a stable nonzero scatter, not collapse. Additionally, "the frontier in category
  k" is a different firm for each k, so nobody is heading toward a single destination.
- **Why style behaves differently.** Style converged because it has no innovation resupply —
  nobody invents a genuinely novel formatting trick that buys an edge. Same pull, opposite
  outcome, because one dimension gets replenished and the other doesn't.
- **Homogenization plausibly *causes* fragmentation.** Gap-chasing is precisely the mechanism
  by which trailing labs became good enough to use. Convergence on quality level → divergence
  in market share. This inverts the usual monoculture worry and deserves its own section
  rather than being left for a referee to find.

**The one-sentence version:** firms converged on being good, stayed scattered on what they're
good at, converged on how they sound — and the convergence on being good is what fragmented
the market.

**Relationship to the theory.** Theorem 5 says outputs do not identify technology *even with
perfect measurement*. The empirics add a second, independent obstacle: the instruments are
too low-rank to resolve what we claim to measure. These are different problems that rhyme,
and the paper is stronger for saying so — there are two separate reasons you cannot read
technology off a leaderboard, and one route (perturbing the evaluation) that gets around
both. §5.1 is the one setting where a scale perturbation is observable, but its technology
result is only suggestive. The stronger conclusion is that open labs mostly share a broad
scaling direction while still differing in training-data allocation and possibly in a few
specialized directions.

---

## 9. What is soft, stated plainly

- **The technology-difference result is borderline.** The calibrated direction test is
  p = 0.0500, the pairwise-cosine test is p = 0.202, and the visible departures are
  concentrated in two labs. Do not present this as a general technology difference.
- **The steady-state over-prediction is narrowed but still unexplained.** §7.1 ruled out
  rating drift and a ceiling effect and showed it's concentrated at release events, leaving
  technique/recipe diffusion as the live candidate — but that is not directly testable
  without recipe-level data, so it remains an inference, not a finding.
- **The continuous-w test (§7.2) most likely used the wrong instrument**, not just a null
  result — Arena vote-share conflates evaluator attention with organic traffic
  classification. Don't read the null forward coefficient as evidence against a w→capability
  effect; it's evidence against this specific measure of w.
- **The firm-heterogeneity result (§3.7) is a permutation-test artifact of two firms.**
  p = 0.036 but the variance-component point estimate is exactly 0; don't state it more
  strongly than "OpenAI and Anthropic look different from each other."
- **The multivariate response matrix (§3.5) cannot distinguish B ≈ beta·I from a richer
  model at n = 65 episodes.** The off-diagonal point estimates are large but unstable
  under cross-validation.
- **The Arena Theorem-9 test is one shock episode**, not four.
- **The style result** runs on a short, heavily autocorrelated window.
- **KL(v‖w)** is computed on 29.7% of tokens due to name matching.
- **Scaling work is open-weight only** — OpenAI, Anthropic, and frontier Google are absent,
  and they are the firms the paper is about.
- **β is a fixed-effects panel regression, not an experiment.** Reverse causality (boards
  add categories *because* those capabilities became salient, which is also when firms
  invested) is partially but not fully addressed by the staggered-launch design.
- Two labs in the efficiency table (`arcee-ai`, `nousresearch` in earlier runs) are
  fine-tune shops, not pretraining labs; their A_i is derivative and they should be dropped
  rather than described as independent technologies.
- **§6.1's board-splitting ratios use a different β estimation convention than §3.1**
  (through-origin scalar OLS on raw events, 0.354, vs. the FE-demeaned 0.5495 used
  elsewhere) — chosen for simplicity in the forward simulation, not because it's the
  preferred estimate. Don't quote 0.354 as *the* headline β.
- **§7.6's ground-truth heterogeneity scale and shock-variance budget are experimenter
  choices**, not independently calibrated to data (unlike §6.1, which starts from the real
  dispersion level). The rank→recovery relationship is real given those choices; the
  absolute recoverability threshold is not a claim about the actual historical record.
- **§7.6's cosine-direction null result rests on one estimator family** (the rank-2
  SVD-basis a_i,b_i model). It shows that estimator's capacity is the bottleneck for
  direction recovery, not that direction is unrecoverable in principle — a full per-firm
  ridge matrix (§7.4's "Model B") was not re-tested in the synthetic setting.

---

## 10. Known defects in the theory draft to fix alongside

- **H is defined as cosine similarity but used as if H = 0 means homogeneity**, while
  Lemma 8 gives H ≥ 1 − O(ε). The appendix proof of Theorem 5 uses the *normal component* of
  θ_i w.r.t. θ_1, which vanishes iff parallel — so define H as **heterogeneity**
  (normal-component norm, or 1 − cos) and flip Lemma 8. One line, but two theorem statements
  are currently incoherent.
- **Theorem 5's transversality argument has a gap**: the claim that the equilibrium-response
  term cancelling the direct term is "an exceptional coincidence on a measure-zero subset"
  is asserted, not proved, and it is the load-bearing step.
- **(D2) in Theorem 9** — that no discrepancy survives *every* perturbation — is close to
  unfalsifiable as stated.
- Theorem 5 cites "Lemma 1" where it means Lemma 4; Lemma 4 has a duplicated empty proof
  block.

---

## 11. Priorities from 2026-08-10 — resolved

All three answered in §7 (Part VI):

1. **Missing homogenizing force (§7.1) [solid]**: not rating drift, not a ceiling effect,
   not continuous — concentrated specifically at release events. Narrows the remaining
   candidate to technique/recipe diffusion (not output imitation, which §3.2 already
   showed is insufficient), untestable further without recipe-level data.
2. **Continuous w_t response (§7.2) [inconclusive]**: no robust forward effect, and a
   reverse-timing placebo of similar size undermines a causal read. More importantly,
   Arena vote-share looks like a demand-side variable (organic traffic classification),
   not the supply-side evaluator-weight decision Theorem 9 means by w — the category-launch
   design in §5.3 remains the right operationalization.
3. **Identification rank of the w-perturbations (§7.3) [solid]**: effective rank 2.06 of
   nominal 8 (structural cap 7), significantly below a permutation null (z=−5.07,
   p=0.001) — historical attention shifts move in ~2 shared directions, not independently
   per category.

**New open item, replacing the above**: §7.1 points at technique/recipe diffusion as the
likeliest remaining candidate for the missing force, but there is no data on hand to test
it directly (would need technical-report text, patch notes, or similar) — scoped in §7.7
and set aside; would need external data acquisition, not a follow-up script.

## 12. Follow-up round (2026-08-10, same day) — resolved

1. **Structural F_i test (§7.4) [solid null]**: tried three shared bases (rank-2 SVD, full
   pooled matrix, own+spillover). An initial pass on the rank-2 basis suggested a small
   gain from a firm-specific productivity scalar, but that reversed once compared against
   the best pooled model (the untruncated full matrix) — CV then selects infinite
   shrinkage on every firm-specific term, in every basis. No recoverable firm
   heterogeneity in held-out prediction at n=65 events.
2. **Formal open-vs-closed test (§7.5) [solid null]**: exact permutation over all 35
   possible 4-vs-4 firm partitions gives p=0.771; the actual split ranks 27th of 35 by
   effect size. Framing dropped.
3. **Recipe/technique-diffusion data (§7.7) [scoped, not pursued]**: no usable data in
   `data/` (no model cards, technical-report corpus, or recipe metadata). Would require
   external scraping — a real data-collection project, not a follow-up script.

---

## 13. Semi-synthetic pair (2026-08-10, evening) — resolved

Two design-section experiments, upgraded from fully stylized (§6, §7.3) to
semi-synthetic: real θ_i⁰, real response law, real bootstrap-resampled residuals, with
only the treatment variable (board structure; w-shock rank) synthetic and controlled.

1. **Board-splitting replication (§6.1) [moderate]**: confirms §6's direction at smaller
   magnitude — 1.09–1.35× diversity from splitting into 2–4 boards, vs. §6's stylized
   1.4–2.1×. Robust to using the pooled ridge B instead of the scalar law. L=8 (singleton
   boards) degenerates into an unanchored random walk, as §6 already warned the fully
   unanchored case would.
2. **Rank-controlled identification test (§7.6) [moderate]**: with a known synthetic
   ground truth (impossible to construct from real data alone), shows that increasing
   w-shock rank from 1 to 4 does help detect *that* firms differ (gain1 roughly doubles),
   consistent with Theorem 9 and with §7.3's real-rank measurement — but does not help
   recover *which direction* they differ in; cosine similarity to the true heterogeneity
   pattern stays at chance level (≈0) regardless of rank, because the recovery estimator's
   own capacity (2 free parameters) is the binding constraint, not the richness of the
   perturbation. This reframes §7.4's "F_i is not recoverable from real data" null: the
   historical record's low rank (§7.3) is a bottleneck, but even a generously rank-7
   synthetic record would not have recovered direction with this estimator family — the
   estimator needs upgrading alongside the evaluation design.

---

## 14. Reproduction

```
experiments/common.py                     shared panel construction + FE estimators
experiments/exp0_measurement.py           raw vs residual cosine, noise accounting
experiments/exp0b_estimand_null.py        permutation null for residual cosine
experiments/exp1_evalchasing.py           beta, with mean-reversion correction
experiments/exp1_heterogeneous.py         firm/group/category gap-chasing splits
experiments/expA_steady_state.py          steady-state accounting + persistence test
experiments/expK_gap_matrix.py            multivariate gap-response matrix B, scalar/common/low-rank/ridge
experiments/expL_persistence.py           persistence of innovation residuals, rho and steady-state re-check
experiments/expM_heterogeneity.py         firm-level beta_i, bootstrap SEs, DL tau^2, permutation test
experiments/expN_nonlinear.py             linearity of gap->improvement: bins, kink, quadratic
experiments/expO_missing_force.py         static-model drift, ceiling effect, continuous-vs-at-event decay
experiments/expP_continuous_w.py          continuous evaluator-attention share vs subsequent improvement
experiments/expQ_w_rank.py                effective rank of the w_share perturbation matrix (uses expP's output)
experiments/expR_structural_Fi.py         1-2 parameter structural F_i, leave-one-event-out prediction
experiments/expS_open_closed.py           gap x Closed interaction, bootstrap + exact 35-way permutation test
experiments/expT_semisynth_boards.py      semi-synthetic board-splitting: real state/law/residuals, synthetic L,s
experiments/expU_w_rank_identification.py semi-synthetic identification: real state/residuals, synthetic B_i*, rank-r w-shocks
experiments/expB_distillation.py          board-following vs teacher-following
experiments/expC_effective_K.py           Arena effective dimensionality
experiments/expD_theorem9.py              (c,w) perturbation test on category launches
experiments/expE_style.py                 raw minus style-controlled dispersion
experiments/expF_design_levers.py         calibrated dose-response for design levers
experiments/expG4_highdim.py              superseded independent-error direction test
experiments/expG5_direction_test.py       calibrated direction test (USE THIS ONE)
experiments/expG6_compute.py              per-FLOP efficiency via Epoch token scaling
experiments/expH_deployment.py            deployment concentration, KL(v||w)
experiments/expJ3_matched.py              matched-resolution effective rank (USE THIS ONE)
experiments/expG2_robust.py               release-date robustness for broad directions
experiments/expJ2_robust.py                effective-rank and task-uniqueness robustness
experiments/pull_ollm_subtasks_resume.py  throttled, resumable HF subtask pull
```

Superseded, kept for provenance: `expG_production_functions.py`,
`expG3_residual_direction.py` (6-dimension version); `expI_subtask_rank.py` (n = 44,
superseded by `expJ3_matched.py`); `expJ_highrank.py` (superseded by the matched
resolution analysis).

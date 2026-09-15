"""Priority 2: continuous w_t / evaluation-attention response.

Extends the Theorem-9 test (§5.3, category launches as discrete w-shocks, effectively one
informative episode) to the FULL panel using continuous variation in evaluator attention.

w_{k,m} = total new Arena votes recorded in category k during calendar month m, summed
across all models with data that month (vote_count is cumulative per model but resets
occasionally on refits, so increments are floored at 0 -- a conservative choice that can
only understate w). w_share_{k,m} = w_{k,m} / sum_k w_{k,m}, the evaluator's revealed
attention allocation across categories that month.

Outcome: theta_bar_{k,m} = mean flagship rating across FRONTIER firms in category k at the
end of month m. Test whether w_share_{k,m} predicts theta_bar_{k,m+1} - theta_bar_{k,m},
i.e. does more evaluator attention to k this month predict more aggregate improvement in k
next month, with category and month fixed effects (so this nets out category base rates and
industry-wide trends -- it's asking about within-category TIMING of attention vs improvement).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, flagships, CORE, FRONTIER

d = load_panel(cats=CORE, firms_only=False)
d['month'] = d.dt.dt.to_period('M')

# ---- w_{k,m}: new votes per category per month, summed across ALL models (not just frontier) ----
cat_d = d[d.category.isin(CORE)].sort_values(['model_name', 'category', 'dt']).copy()
cat_d['dvotes'] = cat_d.groupby(['model_name', 'category']).vote_count.diff().clip(lower=0)
w = cat_d.groupby(['category', 'month']).dvotes.sum().unstack('category').fillna(0.0)
w = w.reindex(columns=CORE)
w_share = w.div(w.sum(axis=1), axis=0)
print(f'months with data: {len(w)}  ({w.index.min()} to {w.index.max()})')
print('\nmean attention share by category:')
print(w_share.mean().sort_values(ascending=False).round(3).to_string())

# ---- theta_bar_{k,m}: mean FRONTIER flagship rating in category k, last snapshot of month m ----
df = load_panel(cats=CORE)
flag = flagships(df)
rat = df.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')

theta_bar = {}
for m, grp in df.groupby(df.dt.dt.to_period('M')):
    last_date = grp.dt.max()
    names_at = flag.xs(last_date, level='dt', drop_level=True) if last_date in flag.index.get_level_values('dt') else None
    if names_at is None:
        continue
    vals = []
    for firm in FRONTIER:
        mdl = names_at.get(firm)
        if mdl is None:
            continue
        try:
            vals.append(rat.loc[(last_date, mdl)][CORE])
        except KeyError:
            continue
    if len(vals) < 4:
        continue
    theta_bar[m] = pd.concat(vals, axis=1).mean(axis=1)
tb = pd.DataFrame(theta_bar).T.sort_index()
tb = tb.reindex(columns=CORE)
print(f'\nmonths with theta_bar: {len(tb)}')

# ---- align w_share (month m) with change in theta_bar (m -> m+1) ----
months = sorted(set(w_share.index) & set(tb.index))
rows = []
for i in range(len(months) - 1):
    m0, m1 = months[i], months[i + 1]
    if (m1 - m0).n != 1:
        continue  # only use genuinely consecutive months
    for k in CORE:
        if pd.notna(tb.loc[m0, k]) and pd.notna(tb.loc[m1, k]) and pd.notna(w_share.loc[m0, k]):
            rows.append(dict(month=m0, cat=k, w=w_share.loc[m0, k],
                              dtheta=tb.loc[m1, k] - tb.loc[m0, k]))
p = pd.DataFrame(rows)
print(f'\ncategory-month observations for the regression: {len(p)} '
      f'({p.month.nunique()} months x up to {len(CORE)} categories)')


def demean(x, by):
    return x - x.groupby(by).transform('mean')


# within category + month FE: isolates whether attention TIMING within a category's own
# history (net of category base rate and industry-wide monthly trend) predicts improvement
W, Y = p.w.copy(), p.dtheta.copy()
for _ in range(60):
    W = demean(demean(W, p.cat), p.month)
    Y = demean(demean(Y, p.cat), p.month)
vw, cov = np.mean(W * W), np.mean(W * Y)
delta = cov / vw
n, n_cat, n_month = len(p), p.cat.nunique(), p.month.nunique()
resid = Y - delta * W
# cluster-robust SE by category (8 clusters -- treat as indicative, not definitive)
sc = pd.Series(W.values * resid.values, index=p.index).groupby(p.cat.values).sum()
se = np.sqrt((sc ** 2).sum()) / (vw * n)
print(f'\ndelta (w_share -> next-month dtheta), category+month FE: {delta:+.3f}  '
      f'(cluster SE {se:.3f}, t={delta/se:+.2f})')

# raw (no FE) correlation for context
raw_corr = np.corrcoef(p.w, p.dtheta)[0, 1]
print(f'raw correlation (no FE): {raw_corr:+.3f}')

# lag-2 and lag-3 robustness (does attention predict improvement 2-3 months out too?)
print('\n--- robustness: longer lags ---')
for lag in [2, 3]:
    rows_l = []
    for i in range(len(months) - lag):
        m0, m1 = months[i], months[i + lag]
        if (m1 - m0).n != lag:
            continue
        for k in CORE:
            if pd.notna(tb.loc[m0, k]) and pd.notna(tb.loc[m1, k]) and pd.notna(w_share.loc[m0, k]):
                rows_l.append(dict(month=m0, cat=k, w=w_share.loc[m0, k],
                                    dtheta=tb.loc[m1, k] - tb.loc[m0, k]))
    pl = pd.DataFrame(rows_l)
    if len(pl) < 20:
        continue
    Wl, Yl = pl.w.copy(), pl.dtheta.copy()
    for _ in range(60):
        Wl = demean(demean(Wl, pl.cat), pl.month)
        Yl = demean(demean(Yl, pl.cat), pl.month)
    d_l = np.mean(Wl * Yl) / np.mean(Wl * Wl)
    print(f'  lag={lag} months: delta={d_l:+.3f}  n={len(pl)}')

# placebo: does dtheta predict PRIOR w (reverse timing -- should be weaker if causal direction is w->dtheta)
print('\n--- placebo: does improvement predict PRECEDING attention (reverse timing)? ---')
rows_rev = []
for i in range(1, len(months)):
    m0, m1 = months[i - 1], months[i]
    if (m1 - m0).n != 1:
        continue
    for k in CORE:
        if pd.notna(tb.loc[m0, k]) and pd.notna(tb.loc[m1, k]) and pd.notna(w_share.loc[m1, k]):
            rows_rev.append(dict(month=m1, cat=k, w=w_share.loc[m1, k],
                                  dtheta_prior=tb.loc[m1, k] - tb.loc[m0, k]))
pr = pd.DataFrame(rows_rev)
Wr, Yr = pr.w.copy(), pr.dtheta_prior.copy()
for _ in range(60):
    Wr = demean(demean(Wr, pr.cat), pr.month)
    Yr = demean(demean(Yr, pr.cat), pr.month)
d_rev = np.mean(Wr * Yr) / np.mean(Wr * Wr)
print(f'  reverse (this-month w on prior-month dtheta): delta={d_rev:+.3f}  n={len(pr)}')

print('\n=== summary ===')
print(f'forward (w_t -> dtheta_t+1): {delta:+.3f} (t={delta/se:+.2f})')
print(f'reverse (dtheta_t-1 -> w_t): {d_rev:+.3f}  (placebo/reverse-causality check)')

# save the w_share panel for expQ (identification rank)
w_share.to_parquet('experiments/_w_share_panel.parquet')
print('\nsaved w_share panel to experiments/_w_share_panel.parquet for expQ')

"""Experiment E: is differentiation being displaced onto dimensions the board
does not score?

Arena publishes raw and style-controlled (SC) ratings. Style control statistically
strips length/formatting effects, so
        style advantage = raw - SC
is a per-model, per-category, per-period measure of the "free differentiation on
brand/style" of section 5.2.

Prediction under the reframing: objective diversity sits at a stable level, while
dispersion in style advantage RISES as firms push differentiation onto the
unscored margin.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, two_way_resid, flagships, PANEL, STYLE, CORE, FRONTIER

raw = load_panel(PANEL, cats=CORE, firms_only=False, since='2024-08-01')
sc = load_panel(STYLE, cats=CORE, firms_only=False, since='2024-08-01')

key = ['dt', 'model_name', 'category']
m = raw[key + ['rating', 'variance', 'org']].merge(
    sc[key + ['rating', 'variance']], on=key, suffixes=('_raw', '_sc'))
m['style'] = m.rating_raw - m.rating_sc
print(f'merged raw+SC: {len(m):,} rows, {m.dt.nunique()} snapshots, '
      f'{m.dt.min().date()} to {m.dt.max().date()}')
print(f'style advantage: mean={m['style'].mean():+.1f}  sd={m['style'].std():.1f}')

# methodology break noted in project notes: SC refit on 2025-05-19
BREAK = pd.Timestamp('2025-05-19')

mf = m[m.org.isin(FRONTIER)]
flag = flagships(load_panel(PANEL, cats=CORE))

rows = []
for date, snap in mf.groupby('dt'):
    keep = {f: flag.get((date, f)) for f in FRONTIER}
    keep = {f: mm for f, mm in keep.items() if mm is not None}
    s = snap[snap.model_name.isin(keep.values())]
    piv_st = s.pivot_table(index='model_name', columns='category', values='style')
    piv_sc = s.pivot_table(index='model_name', columns='category', values='rating_sc')
    piv_v = s.pivot_table(index='model_name', columns='category', values='variance_sc')
    piv_st = piv_st.reindex(columns=CORE).dropna()
    piv_sc = piv_sc.reindex(columns=CORE).dropna()
    common = piv_st.index.intersection(piv_sc.index)
    if len(common) < 5:
        continue
    St, Sc_, Vv = piv_st.loc[common].values, piv_sc.loc[common].values, \
        piv_v.reindex(index=common, columns=CORE).values
    N = len(common); nfac = (N - 1) / N
    rows.append(dict(
        date=date, N=N,
        # style differentiation: cross-firm spread in average style advantage
        style_level_disp=St.mean(axis=1).var(ddof=1),
        # style profile differentiation (two-way residual)
        style_prof_disp=two_way_resid(St).var(axis=0, ddof=1).mean(),
        # objective differentiation on style-controlled ratings
        obj_disp=two_way_resid(Sc_).var(axis=0, ddof=1).mean() - np.nanmean(Vv) * nfac,
    ))

e = pd.DataFrame(rows).set_index('date')
print(f'\nusable snapshots: {len(e)}')
print(e.resample('QE').mean().round(2).to_string())


def trend(y, idx, label):
    t = (idx - idx[0]).days.values / 365.25
    ok = np.isfinite(y)
    b = np.polyfit(t[ok], y[ok], 1)[0]
    r = np.corrcoef(t[ok], y[ok])[0, 1]
    # Newey-West-ish: cluster by quarter for a crude SE
    n = ok.sum()
    se = np.std(y[ok] - np.polyval(np.polyfit(t[ok], y[ok], 1), t[ok])) / \
        (np.std(t[ok]) * np.sqrt(n / 8))
    print(f'  {label:26s} slope/yr={b:+8.2f}  r={r:+.3f}  t~{b/se:+.2f}')


print('\n--- trends over the full SC window ---')
for c, lbl in [('style_level_disp', 'style level dispersion'),
               ('style_prof_disp', 'style profile dispersion'),
               ('obj_disp', 'objective dispersion (SC)')]:
    trend(e[c].values, e.index, lbl)

print('\n--- post-methodology-break only (2025-05-19 onward) ---')
e2 = e[e.index >= BREAK]
print(f'  n={len(e2)}')
for c, lbl in [('style_level_disp', 'style level dispersion'),
               ('style_prof_disp', 'style profile dispersion'),
               ('obj_disp', 'objective dispersion (SC)')]:
    trend(e2[c].values, e2.index, lbl)

print('\nPrediction: style dispersion RISING while objective dispersion FLAT')
print('would show differentiation displaced onto the unscored margin.')

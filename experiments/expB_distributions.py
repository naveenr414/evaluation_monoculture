"""Experiment B support: distributions of evaluated quality and of realized
usage, early versus late, and the entry gap expressed as a fraction.

(4) rating distribution among FRONTIER-firm models, 2025Q1 vs 2026Q2
(5) OpenRouter token-share distribution, 2025 H1 vs 2026 H1
(6) entry gap of new frontier-firm releases as a fraction of the leader's
    rating and of the leader's rating above the floor

Writes expB_distributions.pdf.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp

sys.path.insert(0, 'experiments')
from common import load_panel, CORE, FRONTIER

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 10.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})


def strip(ax):
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[d.category.isin(CORE)].dropna(subset=['rating'])

# ---- (4) quality distribution, frontier firms only ------------------------
print('=' * 72)
print('(4) RATING DISTRIBUTION, FRONTIER-FIRM MODELS')
print('=' * 72)
fr = d[d.org.isin(FRONTIER)].copy()
fr['q'] = pd.PeriodIndex(fr.dt, freq='Q')
EARLY, LATE = pd.Period('2025Q1'), pd.Period('2026Q2')


def gaps_frontier(per):
    """gap to the category leader, over the whole (not just frontier) field."""
    out = []
    sub = d[pd.PeriodIndex(d.dt, freq='Q') == per]
    for (t, cat), g in sub.groupby(['dt', 'category']):
        top = g.rating.max()
        f = g[g.org.isin(FRONTIER)]
        out.append(pd.DataFrame({'gap': top - f.rating.values,
                                 'rating': f.rating.values, 'cat': cat}))
    return pd.concat(out, ignore_index=True)


A, B = gaps_frontier(EARLY), gaps_frontier(LATE)
for lab, x in (('2025Q1', A), ('2026Q2', B)):
    g = x.gap.values
    print(f'  {lab}: n={len(g):5d} models x cats   mean gap {g.mean():6.1f}  '
          f'median {np.median(g):6.1f}  sd {g.std():6.1f}  '
          f'p90 {np.percentile(g, 90):6.1f}  share within 50 pts '
          f'{100*(g <= 50).mean():5.1f}%')
ks = ks_2samp(A.gap.values, B.gap.values)
print(f'  Kolmogorov-Smirnov D={ks.statistic:.3f}  p={ks.pvalue:.3g}')
print(f'  distinct frontier-firm models: {fr[fr.q==EARLY].model_name.nunique()} '
      f'-> {fr[fr.q==LATE].model_name.nunique()}')

# ---- (5) usage distribution ----------------------------------------------
print('\n' + '=' * 72)
print('(5) OPENROUTER TOKEN-SHARE DISTRIBUTION')
print('=' * 72)
o = pd.read_parquet('data/openrouter_daily.parquet')
o['date'] = pd.to_datetime(o.date)
o['org'] = o.model_permaslug.str.split('/').str[0].str.lower()
P1 = (o.date >= '2025-01-01') & (o.date < '2025-07-01')
P2 = (o.date >= '2026-01-01') & (o.date < '2026-07-01')


def ushare(mask):
    g = o[mask]
    sm = g.groupby('model_permaslug').total_tokens.sum().sort_values(ascending=False)
    sm = sm / sm.sum()
    sf = g.groupby('org').total_tokens.sum().sort_values(ascending=False)
    sf = sf / sf.sum()
    return sm, sf


def gini(s):
    v = np.sort(np.asarray(s, float))
    n = len(v)
    return (2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum())


rows = []
for lab, mask in (('2025 H1', P1), ('2026 H1', P2)):
    sm, sf = ushare(mask)
    rows.append(dict(period=lab, models=len(sm), firms=len(sf),
                     eff_models=1 / (sm ** 2).sum(), eff_firms=1 / (sf ** 2).sum(),
                     top1=sm.iloc[0], top3=sm.iloc[:3].sum(),
                     top10=sm.iloc[:10].sum(),
                     top1_firm=sf.iloc[0], top3_firm=sf.iloc[:3].sum(),
                     gini_model=gini(sm.values)))
U = pd.DataFrame(rows)
print(U.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))
sm1, _ = ushare(P1)
sm2, _ = ushare(P2)

# ---- (6) entry gap as a fraction -----------------------------------------
print('\n' + '=' * 72)
print('(6) ENTRY GAP OF NEW FRONTIER-FIRM RELEASES, AS A FRACTION')
print('=' * 72)
rows = []
for cat, g in d.groupby('category'):
    piv = g.pivot_table(index='dt', columns='model_name', values='rating').sort_index()
    seen = piv.notna().cumsum() > 0
    fst = piv.notna().idxmax().where(piv.notna().any())
    start = piv.index[piv.notna().any(axis=1)].min()
    orgs = g.groupby('model_name').org.first()
    for m in piv.columns:
        t0 = fst[m]
        if pd.isna(t0) or t0 <= start + np.timedelta64(14, 'D'):
            continue
        if orgs.get(m) not in FRONTIER:
            continue
        row = piv.loc[t0]
        inc = row[[c for c in piv.columns
                   if c != m and seen.loc[t0, c] and pd.notna(row[c])]]
        if len(inc) < 20 or pd.isna(row[m]):
            continue
        lead, floor = inc.max(), inc.min()
        rows.append(dict(cat=cat, t0=t0, gap=lead - row[m],
                         frac_lead=(lead - row[m]) / lead,
                         frac_span=(lead - row[m]) / (lead - floor)))
G = pd.DataFrame(rows)
G['q'] = pd.PeriodIndex(G.t0, freq='Q')
print(G.groupby('q')[['gap', 'frac_lead', 'frac_span']].mean()
      .join(G.groupby('q').size().rename('n'))
      .to_string(float_format=lambda x: f'{x:8.4f}'))
x = (G.t0 - G.t0.min()).dt.days.values / 365.25
for c in ('gap', 'frac_lead', 'frac_span'):
    b = np.polyfit(x, G[c].values, 1)[0]
    r = np.corrcoef(x, G[c].values)[0, 1]
    print(f'  {c:<10} trend {b:+9.4f}/yr   corr {r:+.3f}')

# -------------------------------------------------------------- figure ----
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4))
ax = axes[0]
xs = np.linspace(0, max(A.gap.max(), B.gap.max()), 400)
for v, col, lab in ((A.gap.values, MUTED, '2025Q1'), (B.gap.values, BLUE, '2026Q2')):
    k = gaussian_kde(v, bw_method=0.25)(xs)
    ax.fill_between(xs, k, color=col, alpha=0.15, lw=0)
    ax.plot(xs, k, color=col, lw=1.7, label=lab)
ax.set_xlabel('gap to category leader')
ax.set_ylabel('density')
ax.set_title('Frontier-firm models', fontsize=10.5, color=INK_2, pad=6)
ax.legend(frameon=False, fontsize=9)
strip(ax)

ax = axes[1]
for s, col, lab in ((sm1, MUTED, '2025 H1'), (sm2, BLUE, '2026 H1')):
    v = np.sort(s.values)[::-1]
    ax.plot(np.arange(1, len(v) + 1), 100 * np.cumsum(v), color=col, lw=1.7,
            label=lab)
ax.set_xscale('log')
ax.set_xlabel('models, ranked by tokens')
ax.set_ylabel('cumulative % of tokens')
ax.set_title('Usage concentration', fontsize=10.5, color=INK_2, pad=6)
ax.legend(frameon=False, fontsize=9, loc='lower right')
strip(ax)

ax = axes[2]
qm = G.groupby('q').frac_lead.agg(['mean', 'sem'])
t = qm.index.to_timestamp() + pd.Timedelta(days=45)
ax.errorbar(t, 100 * qm['mean'], yerr=196 * qm['sem'].fillna(0), color=BLUE,
            lw=1.7, marker='o', ms=3.6, capsize=2)
ax.set_ylabel('entry gap as % of leader rating')
ax.set_title('Entry gap, relative', fontsize=10.5, color=INK_2, pad=6)
ax.set_ylim(0, None)
strip(ax)
fig.tight_layout(pad=0.7, w_pad=1.8)
fig.savefig('expB_distributions.pdf')
print('\nwrote expB_distributions.pdf')

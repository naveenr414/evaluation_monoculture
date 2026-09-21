"""Is deployment/evaluation effective rank rising because there are simply MORE
near-frontier models, or because of something else (e.g. usage fragmenting
evenly across an unchanged number of good models)?

  1. openrouter_effective_rank.pdf   effective # of models actually served
                                      (1/HHI of OpenRouter token share) over time
  2. near_top_count.pdf              small multiples: for x in {1,2,5,10}%, how
                                      many models sit within x% of that
                                      category's top rating, averaged across the
                                      8 CORE categories, over time -- the direct
                                      test of the "more near-top models" hypothesis
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, 'experiments')
from common import load_panel, CORE

INK, INK_2, MUTED, GRID, BASELINE, SURFACE = \
    '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 9.5,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})


def strip_axes(ax, keep=('left', 'bottom')):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=3, width=0.6)


# ============================================================= FIGURE 1 ====
o = pd.read_parquet('data/openrouter_daily.parquet')
o['date'] = pd.to_datetime(o.date)

rows = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    tm = g.groupby('model_permaslug').total_tokens.sum()
    sm = tm / tm.sum()
    rows.append(dict(month=per.to_timestamp(), eff_models=1 / (sm ** 2).sum(), n_models=len(sm)))
h = pd.DataFrame(rows).set_index('month').sort_index()

t = (h.index - h.index[0]).days.values / 365.25
b, r = np.polyfit(t, h.eff_models.values, 1)[0], np.corrcoef(t, h.eff_models.values)[0, 1]

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(h.index, h.eff_models, color=BLUE, lw=1.8, marker='o', markersize=3.2,
        markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=3)
ax.fill_between(h.index, h.eff_models, color=BLUE, alpha=0.08, zorder=1)
ax.set_ylabel(r'effective # models served  ($1/\mathrm{HHI}$ of token share)')
ax.set_xlabel('month')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
ax.annotate(f'slope +{b:.1f}/yr  (r={r:+.2f})', xy=(0.03, 0.94), xycoords='axes fraction',
            fontsize=8, color=INK_2, va='top')
fig.tight_layout(pad=0.7)
fig.savefig('openrouter_effective_rank.pdf')
plt.close(fig)
print(f'eff_models: {h.eff_models.iloc[0]:.2f} -> {h.eff_models.iloc[-1]:.2f}  '
      f'(slope {b:+.2f}/yr, r={r:+.3f})')

# ============================================================= FIGURE 2 ====
d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
dates = sorted(d.dt.unique())
X_VALS = [1, 2, 5, 10]

rows = []
for t_ in dates:
    snap = d[d.dt == t_]
    per_cat_counts = {x: [] for x in X_VALS}
    for cat in CORE:
        sub = snap[(snap.category == cat)].dropna(subset=['rating'])
        if len(sub) < 5:
            continue
        top = sub.rating.max()
        for x in X_VALS:
            thresh = top * (1 - x / 100)
            per_cat_counts[x].append(int((sub.rating >= thresh).sum()))
    if not per_cat_counts[X_VALS[0]]:
        continue
    row = {'dt': t_}
    for x in X_VALS:
        row[x] = np.mean(per_cat_counts[x])
    rows.append(row)
nt = pd.DataFrame(rows).set_index('dt').sort_index()
nt_monthly = nt.resample('MS').mean()

fig, axes = plt.subplots(1, len(X_VALS), figsize=(11.5, 3.0), sharex=True)
for ax, x in zip(axes, X_VALS):
    s = nt_monthly[x].dropna()
    tt = (s.index - s.index[0]).days.values / 365.25
    if len(s) > 3:
        b, r = np.polyfit(tt, s.values, 1)[0], np.corrcoef(tt, s.values)[0, 1]
    else:
        b, r = np.nan, np.nan
    ax.plot(s.index, s.values, color=ORANGE, lw=1.8, marker='o', markersize=2.8,
            markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=0.4, zorder=3)
    ax.fill_between(s.index, s.values, color=ORANGE, alpha=0.08, zorder=1)
    ax.set_title(f'within {x}% of top', fontsize=9, loc='left', color=INK)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.annotate(f'{b:+.1f}/yr' if not np.isnan(b) else '', xy=(0.05, 0.94),
                xycoords='axes fraction', fontsize=7.5, color=INK_2, va='top')

axes[0].set_ylabel('avg. # models\n(across CORE categories)')
fig.tight_layout(pad=0.6, w_pad=1.2)
fig.savefig('near_top_count.pdf')
plt.close(fig)
print('\navg # models within x% of category top, first -> last month:')
print(pd.DataFrame({'first': nt_monthly.iloc[0], 'last': nt_monthly.iloc[-1]}).round(1).to_string())

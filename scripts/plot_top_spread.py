"""How tightly packed is the top of the leaderboard, and how does that depend on
where you draw the line?

For every snapshot and category, the spread is the top rating minus the rating
at the (100 - p) percentile of all rated models, for p = 1, 5, 10 percent.  The
number of models the band covers is plotted alongside, because a percentile band
grows with the pool: 1 percent of 56 models is 1 model and 1 percent of 349 is
3.5.  A fixed-rank version (top 10, 20, 50) is shown for comparison.

Writes top_spread_by_pct.pdf.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import load_panel, CORE

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

PCTS = [1, 5, 10]
RANKS = [10, 20, 50]

d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[d.category.isin(CORE)].dropna(subset=['rating'])
d = d[d.dt >= pd.Timestamp('2024-01-01')]

rows = []
for (t, cat), g in d.groupby(['dt', 'category']):
    v = np.sort(g.rating.values)[::-1]
    n = len(v)
    if n < 20:
        continue
    r = dict(dt=t, cat=cat, n=n, top=v[0])
    for p in PCTS:
        k = max(int(np.ceil(p / 100 * n)), 2)
        r[f'spread_{p}'] = v[0] - v[k - 1]
        r[f'k_{p}'] = k
    for q in RANKS:
        r[f'rank_{q}'] = v[0] - v[min(q, n) - 1]
    rows.append(r)

S = pd.DataFrame(rows)
S['q'] = pd.PeriodIndex(S.dt, freq='Q')
qs = S.groupby('q').mean(numeric_only=True)
print('quarterly means, pooled over the 8 categories')
cols = ['n'] + [f'spread_{p}' for p in PCTS] + [f'k_{p}' for p in PCTS] \
       + [f'rank_{q}' for q in RANKS]
print(qs[cols].to_string(float_format=lambda x: f'{x:7.1f}'))

print('\ntrend per year (OLS on the snapshot-category panel)')
x = (S.dt - S.dt.min()).dt.days.values / 365.25
for c in [f'spread_{p}' for p in PCTS] + [f'k_{p}' for p in PCTS] \
         + [f'rank_{q}' for q in RANKS]:
    b1 = np.polyfit(x, S[c].values, 1)[0]
    r = np.corrcoef(x, S[c].values)[0, 1]
    print(f'  {c:<12} {b1:+8.2f} per year   corr {r:+.3f}')

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), sharex=True)
t = qs.index.to_timestamp() + pd.Timedelta(days=45)
for ax, p in zip(axes, PCTS):
    ax.plot(t, qs[f'spread_{p}'], color=BLUE, lw=1.8, marker='o', ms=3.8)
    ax.set_ylabel(f'top $-$ {100-p}th pctile (points)', color=BLUE)
    ax.tick_params(colors=BLUE, length=3, width=0.6)
    ax.set_ylim(0, max(qs[f'spread_{pp}'].max() for pp in PCTS) * 1.12)
    ax.set_title(f'Top {p}%', fontsize=11, color=INK_2, pad=6)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax2 = ax.twinx()
    ax2.plot(t, qs[f'k_{p}'], color=ORANGE, lw=1.3, ls=(0, (3, 2)),
             marker='s', ms=3)
    ax2.set_ylabel('models in the band', color=ORANGE)
    ax2.tick_params(colors=ORANGE, length=3, width=0.6)
    ax2.set_ylim(0, qs[f'k_{PCTS[-1]}'].max() * 1.15)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
fig.tight_layout(pad=0.7, w_pad=2.4)
fig.savefig('top_spread_by_pct.pdf')
plt.close(fig)
print('\nwrote top_spread_by_pct.pdf')

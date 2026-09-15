"""At what distance from the category leader do new models arrive, and is that
distance changing?

For every model we find its first snapshot with a rating in a category.  The
entry gap is the best rating among models ALREADY present at that snapshot minus
the entrant's own rating, so a negative gap means the entrant took the lead.
Panels are categories, x is the entry date, y is the mean entry gap.

Writes entrant_gap_by_cat.pdf.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, 'experiments')
from common import load_panel, CORE, MATURE_DAYS

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'

DC = {'coding': 'Coding', 'math': 'Math', 'hard_prompts': 'Hard Prompts',
      'instruction_following': 'Instruction Following',
      'creative_writing': 'Creative Writing', 'longer_query': 'Longer Query',
      'multi_turn': 'Multi-Turn', 'english': 'English'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 10.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

# full history so that "first appearance" is genuine, not an artifact of the
# start of the window
d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[d.category.isin(CORE)].dropna(subset=['rating'])
snaps = np.sort(d.dt.unique())
print(f'{len(snaps)} snapshots {pd.Timestamp(snaps[0]).date()} .. '
      f'{pd.Timestamp(snaps[-1]).date()}')

rows = []
for cat, g in d.groupby('category'):
    piv = g.pivot_table(index='dt', columns='model_name', values='rating')
    piv = piv.sort_index()
    seen = piv.notna().cumsum() > 0
    first = piv.notna().idxmax().where(piv.notna().any())
    # models already rated when the CATEGORY launched are not entrants: they all
    # share the category's first snapshot as their apparent entry date
    cat_start = piv.index[piv.notna().any(axis=1)].min()
    for m in piv.columns:
        t0 = first[m]
        if pd.isna(t0) or t0 <= cat_start + np.timedelta64(14, 'D'):
            continue
        row = piv.loc[t0]
        incumbents = row[[c for c in piv.columns
                          if c != m and seen.loc[t0, c] and pd.notna(row[c])]]
        if len(incumbents) < 10:
            continue
        lead = incumbents.max()
        # matured rating of the entrant, 56 days after entry
        later = [s for s in piv.index if s >= t0 + np.timedelta64(MATURE_DAYS, 'D')]
        mat = piv.loc[later[0], m] if later and pd.notna(piv.loc[later[0], m]) else np.nan
        rows.append(dict(cat=cat, model=m, t0=t0,
                         entry_gap=lead - row[m],
                         mature_gap=lead - mat,
                         lead=lead, rating=row[m]))

E = pd.DataFrame(rows)
E['q'] = pd.PeriodIndex(E.t0, freq='Q')
E = E[E.t0 >= pd.Timestamp('2024-01-01')]
print(f'entrants x categories: {len(E)}  distinct models: {E.model.nunique()}')
print('category launch dates excluded from the entrant set:')
for cat, g in d.groupby('category'):
    print(f'  {cat:<24} {pd.Timestamp(g.dt.min()).date()}')
print('\nentrants per quarter (Coding):')
print(E[E.cat == 'coding'].groupby('q').size().to_string())

fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.4), sharex=True, sharey=True)
summ = []
for ax, cat in zip(axes.ravel(), CORE):
    s = E[E.cat == cat]
    ax.axhline(0, color=BASELINE, lw=0.9, ls=(0, (3, 2)))
    ax.scatter(s.t0, s.entry_gap, s=9, color=BLUE, alpha=0.22, lw=0, zorder=2)
    qm = s.groupby('q').entry_gap.agg(['mean', 'sem', 'size'])
    x = qm.index.to_timestamp() + pd.Timedelta(days=45)
    ax.errorbar(x, qm['mean'], yerr=1.96 * qm['sem'].fillna(0), color=INK,
                lw=1.5, marker='o', ms=3.5, capsize=2, zorder=3)
    ax.set_title(DC[cat], fontsize=10, color=INK_2, pad=5)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # linear trend in points per year
    xx = (s.t0 - s.t0.min()).dt.days.values / 365.25
    b1, b0 = np.polyfit(xx, s.entry_gap.values, 1)
    summ.append((DC[cat], s.entry_gap.mean(), s.entry_gap.median(),
                 100 * (s.entry_gap <= 0).mean(), b1, len(s)))
for ax in axes[:, 0]:
    ax.set_ylabel('entry gap to leader')
fig.tight_layout(pad=0.7, w_pad=1.2, h_pad=1.2)
fig.savefig('entrant_gap_by_cat.pdf')
plt.close(fig)

print('\nentry gap by category, 2024-01 onward')
print(pd.DataFrame(summ, columns=['category', 'mean', 'median', 'pct_taking_lead',
                                  'trend_per_year', 'n'])
      .to_string(index=False, float_format=lambda x: f'{x:8.1f}'))

print('\nmean entry gap by quarter, pooled over categories:')
print(E.groupby('q').entry_gap.agg(['mean', 'median', 'size'])
      .to_string(float_format=lambda x: f'{x:8.1f}'))
print('\nwrote entrant_gap_by_cat.pdf')

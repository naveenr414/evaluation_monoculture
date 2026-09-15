"""How far back in time is a new release, in months?

For a model entering category k at date t with rating theta, the frontier age is
t minus the most recent date at which the running-maximum frontier F(t') was
still at or below theta.  A model that takes the lead scores zero.  A model that
would have been best six months ago scores six.  This converts a rating gap into
a time gap, which is comparable across categories and across years.

Restricted to the frontier firms, so it tracks serious releases rather than the
long tail.  Panels are categories.  Writes months_behind_frontier.pdf.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, 'experiments')
from common import load_panel, CORE, FRONTIER

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

# releases from 2024-06 on, so the frontier has at least a year of history
# behind it and the measure is not censored by the start of the panel
FIRST_RELEASE = pd.Timestamp('2024-06-01')

d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[d.category.isin(CORE)].dropna(subset=['rating'])

rows = []
for cat, g in d.groupby('category'):
    piv = g.pivot_table(index='dt', columns='model_name', values='rating').sort_index()
    frontier = piv.max(axis=1).cummax()          # monotone historical frontier
    ft, fv = frontier.index.values, frontier.values
    seen = piv.notna().cumsum() > 0
    fst = piv.notna().idxmax().where(piv.notna().any())
    orgs = g.groupby('model_name').org.first()
    for m in piv.columns:
        t0 = fst[m]
        if pd.isna(t0) or t0 < FIRST_RELEASE:
            continue
        if orgs.get(m) not in FRONTIER:
            continue
        theta = piv.loc[t0, m]
        if pd.isna(theta):
            continue
        # last snapshot BEFORE the release whose frontier was still at or
        # below theta; searching the whole series would look into the future
        # and return negative ages for models near the current frontier
        upto = np.searchsorted(ft, np.datetime64(t0), side='right')
        j = np.searchsorted(fv[:upto], theta, side='right') - 1
        if j < 0:
            continue
        months = (pd.Timestamp(t0) - pd.Timestamp(ft[j])).days / 30.44
        rows.append(dict(cat=cat, model=m, org=orgs.get(m), t0=t0,
                         theta=theta, months=months,
                         gap=frontier.loc[t0] - theta))

E = pd.DataFrame(rows)
E['q'] = pd.PeriodIndex(E.t0, freq='Q')
print(f'{E.model.nunique()} frontier-firm releases, {len(E)} model-category rows')
print('releases per firm:', E.drop_duplicates('model').org.value_counts().to_dict())

print('\nmonths behind the frontier, quarterly mean pooled over categories')
print(E.groupby('q').months.agg(['mean', 'median', 'size'])
      .to_string(float_format=lambda x: f'{x:7.2f}'))

print('\nby category')
rows = []
for cat in CORE:
    s = E[E.cat == cat].sort_values('t0')
    x = (s.t0 - s.t0.min()).dt.days.values / 365.25
    b1 = np.polyfit(x, s.months.values, 1)[0]
    r = np.corrcoef(x, s.months.values)[0, 1]
    rows.append((DC[cat], s.months.mean(), s.months.median(),
                 100 * (s.months <= 0.5).mean(), b1, r, len(s)))
print(pd.DataFrame(rows, columns=['category', 'mean', 'median', 'pct_at_frontier',
                                  'trend_per_yr', 'corr', 'n'])
      .to_string(index=False, float_format=lambda x: f'{x:8.2f}'))

fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.4), sharex=True, sharey=True)
for ax, cat in zip(axes.ravel(), CORE):
    s = E[E.cat == cat]
    ax.scatter(s.t0, s.months, s=13, color=BLUE, alpha=0.28, lw=0, zorder=2)
    qm = s.groupby('q').months.agg(['mean', 'sem'])
    t = qm.index.to_timestamp() + pd.Timedelta(days=45)
    ax.errorbar(t, qm['mean'], yerr=1.96 * qm['sem'].fillna(0), color=INK,
                lw=1.5, marker='o', ms=3.5, capsize=2, zorder=3)
    ax.axhline(0, color=BASELINE, lw=0.9, ls=(0, (3, 2)))
    ax.set_title(DC[cat], fontsize=10, color=INK_2, pad=5)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
for ax in axes[:, 0]:
    ax.set_ylabel('months behind the frontier')
fig.tight_layout(pad=0.7, w_pad=1.2, h_pad=1.2)
fig.savefig('months_behind_frontier.pdf')
plt.close(fig)
print('\nwrote months_behind_frontier.pdf')

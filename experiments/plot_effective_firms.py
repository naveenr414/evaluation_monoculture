"""Effective number of firms, on the deployment side and on the evaluation side.

Deployment (OpenRouter): the inverse Herfindahl of monthly token shares, by firm
and by model.  This is the usual concentration measure -- if three firms split
the market evenly the effective number is three.

Evaluation (Arena leaderboard): the same statistic needs a weight, and the
leaderboard has no usage.  We use two, both computed per category and averaged:
  top-k presence   a firm's share of the k best-rated models
  softmax          a firm's share of exp(theta / tau), tau = 50 rating points,
                   i.e. attention concentrated on the best models

Leaderboard series stop at 2026-03-19, because the organization field goes blank
for 50-61 tail models from 2026-04-20 and every firm-keyed statistic breaks.

Writes effective_firms.pdf.
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

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 10.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

ORG_BREAK = pd.Timestamp('2026-03-19')
TAU = 50.0
TOPK = [10, 20, 50]


def eff(shares):
    s = np.asarray(shares, float)
    s = s[s > 0]
    s = s / s.sum()
    return 1.0 / (s ** 2).sum()


# ------------------------------------------------ deployment ---------------
o = pd.read_parquet('data/openrouter_daily.parquet')
o['date'] = pd.to_datetime(o.date)
o['org'] = o.model_permaslug.str.split('/').str[0].str.lower()
rows = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    tm = g.groupby('model_permaslug').total_tokens.sum()
    tf = g.groupby('org').total_tokens.sum()
    rows.append(dict(month=per.to_timestamp(), eff_firms=eff(tf.values),
                     eff_models=eff(tm.values), n_firms=len(tf),
                     n_models=len(tm), top3=tf.nlargest(3).sum() / tf.sum()))
OR = pd.DataFrame(rows).set_index('month')
print('OpenRouter deployment')
print(OR.round(2).to_string())

# ------------------------------------------------ evaluation ---------------
d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[d.category.isin(CORE)].dropna(subset=['rating'])
d = d[(d.dt >= pd.Timestamp('2024-01-01')) & (d.dt <= ORG_BREAK)]
d = d[d.org.astype(str).str.strip() != '']

rows = []
for (t, cat), g in d.groupby(['dt', 'category']):
    g = g.sort_values('rating', ascending=False)
    r = dict(dt=t, cat=cat, n_models=len(g), n_firms=g.org.nunique())
    for k in TOPK:
        top = g.head(k)
        r[f'eff_top{k}'] = eff(top.groupby('org').size().values)
    w = np.exp((g.rating.values - g.rating.max()) / TAU)
    r['eff_softmax'] = eff(pd.Series(w).groupby(g.org.values).sum().values)
    r['eff_models_softmax'] = eff(w)
    rows.append(r)
LB = pd.DataFrame(rows)
LB['month'] = LB.dt.dt.to_period('M').dt.to_timestamp()
LBm = LB.groupby('month').mean(numeric_only=True)
print('\nArena leaderboard, averaged over the 8 categories')
print(LBm[['n_models', 'n_firms', 'eff_top10', 'eff_top20', 'eff_top50',
           'eff_softmax']].round(2).to_string())

print('\ntrend per year')
for lab, df, cols in (('OpenRouter', OR, ['eff_firms', 'eff_models', 'top3']),
                      ('Leaderboard', LBm, ['eff_top10', 'eff_top20',
                                            'eff_top50', 'eff_softmax',
                                            'n_firms'])):
    t = (df.index - df.index[0]).days.values / 365.25
    for c in cols:
        b = np.polyfit(t, df[c].values, 1)[0]
        r = np.corrcoef(t, df[c].values)[0, 1]
        print(f'  {lab:<12} {c:<18} {b:+7.3f}/yr   r {r:+.3f}   '
              f'{df[c].iloc[0]:6.2f} -> {df[c].iloc[-1]:6.2f}')

fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4))
ax = axes[0]
ax.plot(OR.index, OR.eff_firms, color=BLUE, lw=1.8, marker='o', ms=3.4,
        label='firms')
ax.plot(OR.index, OR.eff_models, color=ORANGE, lw=1.4, ls=(0, (3, 2)),
        marker='s', ms=3, label='models')
ax.set_title('Deployment (OpenRouter tokens)', fontsize=10.5, color=INK_2, pad=6)
ax.set_ylabel('effective number')
ax.legend(frameon=False, fontsize=9, loc='upper left')

ax = axes[1]
for k, col in zip(TOPK, (BLUE, AQUA, VIOLET)):
    ax.plot(LBm.index, LBm[f'eff_top{k}'], color=col, lw=1.6, label=f'top {k}')
ax.set_title('Evaluation, top-$k$ presence', fontsize=10.5, color=INK_2, pad=6)
ax.set_ylabel('effective number of firms')
ax.legend(frameon=False, fontsize=9, loc='upper left')

ax = axes[2]
ax.plot(LBm.index, LBm.eff_softmax, color=BLUE, lw=1.8, label='firms')
ax.plot(LBm.index, LBm.eff_models_softmax, color=ORANGE, lw=1.4, ls=(0, (3, 2)),
        label='models')
ax.set_title(f'Evaluation, softmax weight ($\\tau={TAU:.0f}$)',
             fontsize=10.5, color=INK_2, pad=6)
ax.set_ylabel('effective number')
ax.legend(frameon=False, fontsize=9, loc='upper left')

for ax in axes:
    ax.set_ylim(0, None)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
fig.tight_layout(pad=0.7, w_pad=1.8)
fig.savefig('effective_firms.pdf')
plt.close(fig)
print('\nwrote effective_firms.pdf')

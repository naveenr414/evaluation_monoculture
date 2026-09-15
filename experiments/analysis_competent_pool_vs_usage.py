"""Does usage spread across more models because more models are near-frontier
competent (not because homogenization/specialization is changing)?

Two independent monthly series, both from real deployment/rating data, no
shared machinery so neither can be mechanically forcing agreement with the
other:

  n_competent   = effective count of models within reach of the Arena
                  'overall' frontier that month (soft count via
                  exp(-(top_rating - rating)/SCALE), SCALE=100 Elo ~ a
                  competitiveness band -- NOT a raw count of all submitted
                  models, which mechanically grows as the pool grows)
  eff_used      = effective number of models actually served on OpenRouter
                  that month (1/HHI of token share, same definition as expH)

If the user's hypothesis is right, these should move together over time and
be positively correlated -- usage breadth tracks the size of the competent
pool, not a change in how homogeneous/specialized models are.

Output: experiments/plot_competent_pool_vs_usage.png
"""
import sys
sys.path.insert(0, 'experiments')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCALE = 100.0  # Elo competitiveness band

# ---- n_competent: Arena 'overall' leaderboard ----
d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
ov = d[d.category == 'overall'].dropna(subset=['rating'])
ov = ov[ov.dt >= '2025-01-01']

comp_rows = []
for dt, g in ov.groupby('dt'):
    top = g.rating.max()
    w = np.exp(-(top - g.rating.values) / SCALE)
    comp_rows.append(dict(dt=dt, n_total=len(g), n_competent=w.sum()))
comp = pd.DataFrame(comp_rows).set_index('dt')

# monthly: last Arena snapshot of the month
comp_m = comp.groupby(comp.index.to_period('M')).last()
comp_m.index = comp_m.index.to_timestamp()

# ---- eff_used: OpenRouter deployment concentration (same defn as expH H1) ----
o = pd.read_parquet('data/openrouter_daily.parquet')
o['date'] = pd.to_datetime(o.date)
use_rows = []
for per, g in o.groupby(o.date.dt.to_period('M')):
    tm = g.groupby('model_permaslug').total_tokens.sum()
    sm = tm / tm.sum()
    use_rows.append(dict(month=per.to_timestamp(), n_models=len(sm), eff_used=1 / (sm ** 2).sum()))
use = pd.DataFrame(use_rows).set_index('month')

m = comp_m.join(use, how='inner')
print(m.round(2).to_string())

t = (m.index - m.index[0]).days.values / 365.25
for c in ['n_total', 'n_competent', 'eff_used']:
    b = np.polyfit(t, m[c].values, 1)[0]
    r = np.corrcoef(t, m[c].values)[0, 1]
    print(f'{c:12s} slope/yr={b:+8.2f}  r_vs_time={r:+.3f}')
r_xy = np.corrcoef(m.n_competent.values, m.eff_used.values)[0, 1]
r_xy_total = np.corrcoef(m.n_total.values, m.eff_used.values)[0, 1]
print(f'\ncorr(n_competent, eff_used) = {r_xy:+.3f}')
print(f'corr(n_total,     eff_used) = {r_xy_total:+.3f}  (raw pool size, for comparison)')

# ---- plot: two stacked time series (no dual axis) + a scatter ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={'width_ratios': [1.3, 1.3, 1]})

for ax in axes[:2]:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')

ax = axes[0]
ax.plot(m.index, m.n_competent, color='#0072B2', lw=2.2, marker='o', ms=3.5)
ax.set_title('Effective # of near-frontier models\n(Arena "overall", competitive with #1)')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('effective count')

ax = axes[1]
ax.plot(m.index, m.eff_used, color='#009E73', lw=2.2, marker='o', ms=3.5)
ax.set_title('Effective # of models actually served\n(OpenRouter, 1/HHI of token share)')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('effective count')

ax = axes[2]
sc = ax.scatter(m.n_competent, m.eff_used, c=t, cmap='viridis', s=45, zorder=3)
ax.set_xlabel('effective # near-frontier models')
ax.set_ylabel('effective # models used')
ax.set_title(f'r = {r_xy:+.2f}')
ax.spines[['top', 'right']].set_visible(False)
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
cb.set_label('year', fontsize=8)
cb.set_ticks([t.min(), t.max()])
cb.set_ticklabels([m.index[0].strftime('%Y-%m'), m.index[-1].strftime('%Y-%m')])

fig.suptitle('Is wider usage explained by a bigger competent pool, not by (de)homogenization?', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('experiments/plot_competent_pool_vs_usage.png', dpi=150, bbox_inches='tight')
print('\nwrote experiments/plot_competent_pool_vs_usage.png')

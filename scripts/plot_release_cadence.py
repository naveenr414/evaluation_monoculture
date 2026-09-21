"""Release cadence, and whether the entry gap is shrinking for real or only
because the leaderboard is getting denser.

Three things over time:
  1. entrants per quarter, distinct firms per quarter, entrants per firm
  2. days between consecutive releases by the same firm
  3. the entry gap measured three ways
       points            leader - entrant
       normalized        the same divided by the sd of the incumbent ratings
       percentile        the entrant's rank among incumbents

If the point gap falls only because more models are packed into the same rating
range, the percentile and the normalized gap will be flat.

Writes release_cadence.pdf.
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
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

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
ovr = d[d.category == 'overall'].dropna(subset=['rating']).copy()

# entry date of every model, on the overall leaderboard
first = ovr.groupby('model_name').dt.min().rename('t0')
org = ovr.groupby('model_name').org.first()
ent = pd.DataFrame({'t0': first, 'org': org}).reset_index()
ent = ent[ent.t0 >= pd.Timestamp('2024-01-01')]
ent['q'] = pd.PeriodIndex(ent.t0, freq='Q')

per_q = ent.groupby('q').agg(models=('model_name', 'nunique'),
                             firms=('org', 'nunique'))
per_q['per_firm'] = per_q.models / per_q.firms
pool = ovr.groupby('dt').model_name.nunique()
pool.index = pd.PeriodIndex(pool.index, freq='Q')
per_q['pool'] = pool.groupby(level=0).mean()

# days between consecutive releases by the same firm
ent = ent.sort_values(['org', 't0'])
ent['gap_days'] = ent.groupby('org').t0.diff().dt.days
cad = ent.dropna(subset=['gap_days']).groupby('q').gap_days.agg(['median', 'mean', 'size'])

print('releases and cadence by quarter')
print(per_q.join(cad[['median', 'mean']].rename(
    columns={'median': 'median_days', 'mean': 'mean_days'}))
      .to_string(float_format=lambda x: f'{x:8.1f}'))

# ---- entry gap three ways, per category ----------------------------------
rows = []
for cat, g in d[d.category.isin(CORE)].dropna(subset=['rating']).groupby('category'):
    piv = g.pivot_table(index='dt', columns='model_name', values='rating').sort_index()
    seen = piv.notna().cumsum() > 0
    fst = piv.notna().idxmax().where(piv.notna().any())
    start = piv.index[piv.notna().any(axis=1)].min()
    for m in piv.columns:
        t0 = fst[m]
        if pd.isna(t0) or t0 <= start + np.timedelta64(14, 'D'):
            continue
        row = piv.loc[t0]
        inc = row[[c for c in piv.columns
                   if c != m and seen.loc[t0, c] and pd.notna(row[c])]]
        if len(inc) < 20 or pd.isna(row[m]):
            continue
        v = inc.values
        top = np.sort(v)[::-1]
        spread10 = top[0] - top[9]      # leader minus 10th best incumbent
        # top 1% of the incumbent pool: a MOVING number of models, 1 early and
        # about 3 late, so report how many it covers alongside the spread
        k1 = max(int(np.ceil(0.01 * len(v))), 2)
        spread1 = top[0] - top[k1 - 1]
        rows.append(dict(cat=cat, t0=t0, n_inc=len(v),
                         gap=v.max() - row[m],
                         norm=(v.max() - row[m]) / v.std(ddof=1),
                         pct=100 * (v < row[m]).mean(),
                         rank=int((v > row[m]).sum()) + 1,
                         top10norm=(v.max() - row[m]) / spread10 if spread10 > 0 else np.nan,
                         top1norm=(v.max() - row[m]) / spread1 if spread1 > 0 else np.nan,
                         spread10=spread10, spread1=spread1, k1=k1,
                         sd_inc=v.std(ddof=1),
                         range_inc=v.max() - v.min()))
E = pd.DataFrame(rows)
E['q'] = pd.PeriodIndex(E.t0, freq='Q')
qs = E.groupby('q').agg(gap=('gap', 'mean'), norm=('norm', 'mean'),
                        pct=('pct', 'mean'), rank=('rank', 'median'),
                        top10norm=('top10norm', 'mean'),
                        top1norm=('top1norm', 'mean'),
                        spread10=('spread10', 'mean'),
                        spread1=('spread1', 'mean'), k1=('k1', 'mean'), sd_inc=('sd_inc', 'mean'),
                        n_inc=('n_inc', 'mean'), n=('gap', 'size'))
print('\nentry gap three ways, pooled over the 8 categories')
print(qs.to_string(float_format=lambda x: f'{x:8.1f}'))

for col, lab in (('gap', 'points'), ('norm', 'normalized'), ('pct', 'percentile'),
                 ('rank', 'debut rank'), ('top10norm', 'gap/top10 spread'),
                 ('spread1', 'top-1% spread'), ('top1norm', 'gap/top1% spread'),
                 ('spread10', 'top-10 spread')):
    sub = E.dropna(subset=[col])
    x = (sub.t0 - E.t0.min()).dt.days.values / 365.25
    b1, b0 = np.polyfit(x, sub[col].values, 1)
    r = np.corrcoef(x, sub[col].values)[0, 1]
    print(f'  trend in {lab:<11} {b1:+8.2f} per year   corr with time {r:+.3f}')

# ------------------------------------------------------------------ plot ---
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.2))
t = per_q.index.to_timestamp() + pd.Timedelta(days=45)

ax = axes[0]
ax.bar(t, per_q.models, width=70, color=BLUE, alpha=0.75, lw=0, label='models')
ax.plot(t, per_q.firms, color=ORANGE, lw=1.6, marker='o', ms=3.5, label='firms')
ax.set_ylabel('count per quarter')
ax.set_title('New models and firms', fontsize=10.5, color=INK_2, pad=6)
ax.legend(frameon=False, fontsize=9, loc='upper left')
strip(ax)

ax = axes[1]
ct = cad.index.to_timestamp() + pd.Timedelta(days=45)
ax.plot(ct, cad['median'], color=BLUE, lw=1.7, marker='o', ms=3.5, label='median')
ax.plot(ct, cad['mean'], color=MUTED, lw=1.3, ls=(0, (3, 2)), label='mean')
ax.set_ylabel('days since the firm\'s previous release')
ax.set_title('Release cadence', fontsize=10.5, color=INK_2, pad=6)
ax.legend(frameon=False, fontsize=9, loc='upper right')
strip(ax)

ax = axes[2]
qt = qs.index.to_timestamp() + pd.Timedelta(days=45)
ax.plot(qt, qs.gap, color=BLUE, lw=1.7, marker='o', ms=3.5)
ax.set_ylabel('mean entry gap (rating points)')
ax.set_title('Entry gap, points', fontsize=10.5, color=INK_2, pad=6)
ax.set_ylim(0, None)
strip(ax)

ax = axes[3]
ax.plot(qt, qs['rank'], color=AQUA, lw=1.7, marker='o', ms=3.5)
ax.set_ylabel('median debut rank', color=AQUA)
ax.tick_params(colors=AQUA, length=3, width=0.6)
ax.invert_yaxis()
ax2 = ax.twinx()
ax2.plot(qt, qs.spread1, color=ORANGE, lw=1.4, ls=(0, (3, 2)), marker='s', ms=3,
         label='top 1%')
ax2.plot(qt, qs.spread10, color=ORANGE, lw=1.1, ls=':', marker='^', ms=3,
         alpha=0.6, label='top 10')
ax2.set_ylabel('spread below the leader (points)', color=ORANGE)
ax2.legend(frameon=False, fontsize=8.5, loc='upper right', labelcolor=ORANGE)
ax2.tick_params(colors=ORANGE, length=3, width=0.6)
for s in ax2.spines.values():
    s.set_visible(False)
ax.set_title('Debut rank and top spread', fontsize=10.5, color=INK_2, pad=6)
strip(ax)
for a in axes:
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
fig.tight_layout(pad=0.7, w_pad=1.9)
fig.savefig('release_cadence.pdf')
plt.close(fig)
print('\nwrote release_cadence.pdf')

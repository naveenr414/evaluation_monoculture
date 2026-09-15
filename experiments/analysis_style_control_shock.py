"""Does a firm's persistent category-specific advantage survive a real w-shock?

LMArena publishes two scorings of the SAME battles at the SAME time: raw
preference votes (w_raw) and style-controlled votes (w_sc, which nets out
response length/formatting/markdown -- a real, sharp, well-documented
reweighting of what the judge rewards, in production since 2024-08-28).
That is genuine, repeated (c,w) variation on a FIXED x_i, t -- a much
higher-powered version of Theorem 9's identification test than a single
dated category-launch placebo (expD).

If a firm's category-specific edge (two-way residual, as in the persistent-
advantage plot) survives style control, it's real capability heterogeneity.
If it shrinks toward zero or flips sign under w_sc, the "advantage" was
partly/wholly a style-gaming artifact of the raw judge.

Output: experiments/plot_style_control_shock.png
"""
import sys
sys.path.insert(0, 'experiments')
from common import load_panel, firm_profiles, two_way_resid, CORE, FRONTIER, STYLE, PANEL
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SINCE = '2024-08-28'  # style-control launch date
d_raw = load_panel(path=PANEL, cats=CORE, since=SINCE)
d_sc = load_panel(path=STYLE, cats=CORE, since=SINCE)

common_dates = sorted(set(d_raw.dt.unique()) & set(d_sc.dt.unique()))
print(f'{len(common_dates)} snapshots common to raw and style-controlled panels since {SINCE}')

rows = []
for dt in common_dates:
    Pr, _, nr = firm_profiles(d_raw, CORE, dt)
    Ps, _, ns = firm_profiles(d_sc, CORE, dt)
    if len(Pr) < 5 or len(Ps) < 5 or nr != ns:
        continue
    Rr = two_way_resid(Pr)
    Rs = two_way_resid(Ps)
    for i, f in enumerate(nr):
        for j, c in enumerate(CORE):
            rows.append(dict(dt=dt, firm=f, cat=c, resid_raw=Rr[i, j], resid_sc=Rs[i, j]))
df = pd.DataFrame(rows)
print(f'{len(df)} (date, firm, cat) observations with both raw and SC residuals\n')

# pooled relationship: does the SC residual track the raw residual, or is a lot
# of the raw "advantage" explained away once style is controlled for?
r_pooled = np.corrcoef(df.resid_raw, df.resid_sc)[0, 1]
slope = np.polyfit(df.resid_raw, df.resid_sc, 1)[0]
print(f'pooled corr(resid_raw, resid_sc) = {r_pooled:+.3f}')
print(f'pooled slope (resid_sc ~ resid_raw) = {slope:+.3f}  '
      f'(1.0 = fully survives style control, 0 = fully explained by style)\n')

# per-(firm,cat) summary: mean residual under each regime, over the whole window
summ = df.groupby(['firm', 'cat'])[['resid_raw', 'resid_sc']].mean()
summ['shrinkage'] = 1 - summ.resid_sc / summ.resid_raw.where(summ.resid_raw.abs() > 1e-6)
summ['sign_flip'] = np.sign(summ.resid_raw) != np.sign(summ.resid_sc)
summ = summ.sort_values('resid_raw', key=lambda s: s.abs(), ascending=False)
print('largest raw advantages/disadvantages and what happens under style control:')
print(summ.head(15).round(2).to_string())
print(f'\n{summ.sign_flip.mean():.0%} of (firm,cat) pairs flip sign under style control')
print(f"of the top 15 |raw| pairs, {summ.head(15).sign_flip.mean():.0%} flip sign")

# ---- plot: raw vs SC mean residual, one point per (firm, cat) ----
FIRM_COLOR = {
    'openai': '#E69F00', 'anthropic': '#CC79A7', 'google': '#0072B2',
    'meta': '#009E73', 'mistral': '#56B4E9', 'xai': '#D55E00',
    'deepseek': '#F0E442', 'alibaba': '#000000',
}
fig, ax = plt.subplots(figsize=(7.5, 7))
lim = max(summ.resid_raw.abs().max(), summ.resid_sc.abs().max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], color='#AAAAAA', lw=1, ls='--', zorder=0,
        label='survives unchanged')
ax.axhline(0, color='#DDDDDD', lw=0.8, zorder=0)
ax.axvline(0, color='#DDDDDD', lw=0.8, zorder=0)
for (firm, cat), row in summ.iterrows():
    ax.scatter(row.resid_raw, row.resid_sc, color=FIRM_COLOR[firm], s=55, zorder=3,
               edgecolor='white', linewidth=0.5)
    if abs(row.resid_raw) > summ.resid_raw.abs().quantile(0.75):
        ax.annotate(f'{firm}\n{cat}', (row.resid_raw, row.resid_sc), fontsize=6.5,
                    xytext=(4, 4), textcoords='offset points', color=FIRM_COLOR[firm])
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel('mean two-way residual, RAW preference votes (w_raw)')
ax.set_ylabel('mean two-way residual, STYLE-CONTROLLED votes (w_sc)')
ax.set_title('Does a firm-category advantage survive a real w-shock (style control)?')
ax.spines[['top', 'right']].set_visible(False)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=f)
           for f, c in FIRM_COLOR.items()]
ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig('experiments/plot_style_control_shock.png', dpi=150, bbox_inches='tight')
print('\nwrote experiments/plot_style_control_shock.png')

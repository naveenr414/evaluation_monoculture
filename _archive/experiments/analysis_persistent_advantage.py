"""Persistent firm x category advantage over time.

For each snapshot, compute the two-way (firm, category) residual of flagship
BT ratings -- i.e. each firm's rating in a category after netting out (a) that
firm's own average level across categories and (b) that category's average
level across firms that quarter. A firm with a persistently positive residual
in one category (and not others) is showing a genuine, sustained category-
specific edge rather than an overall-quality effect.

Output: experiments/plot_persistent_advantage.png
"""
import sys
sys.path.insert(0, 'experiments')
from common import load_panel, firm_profiles, two_way_resid, CORE, FRONTIER
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

d = load_panel(cats=CORE)
dates = sorted(d.dt.unique())

rows = []
for dt in dates:
    P, V, names = firm_profiles(d, CORE, dt)
    if len(P) < 5:
        continue
    R = two_way_resid(P)
    for i, f in enumerate(names):
        for j, c in enumerate(CORE):
            rows.append(dict(dt=dt, firm=f, cat=c, resid=R[i, j]))
df = pd.DataFrame(rows)

# Okabe-Ito categorical palette (CVD-safe), fixed hue order by firm
FIRM_COLOR = {
    'openai':    '#E69F00',
    'anthropic': '#CC79A7',
    'google':    '#0072B2',
    'meta':      '#009E73',
    'mistral':   '#56B4E9',
    'xai':       '#D55E00',
    'deepseek':  '#F0E442',
    'alibaba':   '#000000',
}
CAT_LABEL = {
    'coding': 'Coding', 'math': 'Math', 'hard_prompts': 'Hard Prompts',
    'instruction_following': 'Instruction Following', 'creative_writing': 'Creative Writing',
    'longer_query': 'Longer Query', 'multi_turn': 'Multi-Turn', 'english': 'English',
}

# pick the 2 most extreme, longest-standing (firm, cat) pairs per category to
# highlight in color; every other firm is drawn as a thin gray backdrop line so
# the "persistent in SOME categories, not others" contrast is visible per-panel
g = df.groupby(['firm', 'cat'])['resid']
summ = g.agg(['mean', 'std', 'count'])
summ['t'] = summ['mean'] / (summ['std'] / np.sqrt(summ['count']))
highlight = {}
for cat in CORE:
    top = summ.xs(cat, level='cat')['t'].abs().sort_values(ascending=False).head(2).index.tolist()
    highlight[cat] = top

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
axes = axes.ravel()

for ax, cat in zip(axes, CORE):
    sub = df[df.cat == cat]
    hi = highlight[cat]
    for firm in FRONTIER:
        if firm in hi:
            continue
        s = sub[sub.firm == firm].sort_values('dt')
        if len(s) < 10:
            continue
        ax.plot(s.dt, s.resid, color='#B0B0B0', lw=0.9, alpha=0.6, zorder=1)
    for firm in hi:
        s = sub[sub.firm == firm].sort_values('dt')
        if len(s) < 10:
            continue
        ax.plot(s.dt, s.resid, color=FIRM_COLOR[firm], lw=2.0, zorder=3)
        end_y = s.resid.values[-6:].mean()
        ax.annotate(firm, xy=(s.dt.values[-1], end_y), xytext=(4, 0),
                    textcoords='offset points', color=FIRM_COLOR[firm],
                    fontsize=8.5, fontweight='bold', va='center', clip_on=False)
    ax.axhline(0, color='#888888', lw=0.8, ls='--', zorder=0)
    ax.set_title(CAT_LABEL[cat], fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)

fig.supylabel('Firm x category residual (BT points, own-level & category-level netted out)', fontsize=10)
fig.text(0.5, 0.005,
         'Color = the 2 firms with the strongest, most consistent edge in that category (labeled at line end); gray = all other frontier firms',
         ha='center', fontsize=8.5, color='#666666')
fig.suptitle('Some firms hold a persistent edge in some categories -- but not all', fontsize=13)
fig.tight_layout(rect=[0, 0.02, 0.94, 0.96])
fig.savefig('experiments/plot_persistent_advantage.png', dpi=150, bbox_inches='tight')
print('wrote experiments/plot_persistent_advantage.png')

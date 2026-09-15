"""How crowded is the top of each category, and has it changed?

For every category and every threshold x, the share of rated models whose
rating is at least (1 - x/100) times that snapshot's top rating in that
category -- the same threshold definition the draft uses in RQ2.  Averaged over
the snapshots in each period.

  near_top_by_cat.pdf        8 category panels, one curve per period
  near_top_by_cat_rows.pdf   the same as two rows, early on top, latest below

Because Arena ratings have an arbitrary origin, the console output repeats the
exercise in rating points.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'experiments')
from common import load_panel, CORE

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

XS = np.arange(0, 51, 1)
PTS = np.arange(0, 301, 5)

d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
d = d[d.category.isin(CORE)].dropna(subset=['rating'])
d['q'] = pd.PeriodIndex(d.dt, freq='Q')
quarters = sorted(d.q.unique())
EARLY, LATE = quarters[0], quarters[-1]
print(f'periods: {EARLY} and {LATE}')


def curves(sub):
    """Mean over snapshots of the share (and count) of models within x% of top."""
    sh, ct, pt = [], [], []
    for _, g in sub.groupby('dt'):
        r = g.rating.values
        if len(r) < 10:
            continue
        top = r.max()
        sh.append([(r >= top * (1 - x / 100)).mean() for x in XS])
        ct.append([(r >= top * (1 - x / 100)).sum() for x in XS])
        pt.append([(r >= top - p).mean() for p in PTS])
    if not sh:
        return None
    return (np.array(sh).mean(0), np.array(ct).mean(0), np.array(pt).mean(0),
            len(sh), sub.groupby('dt').size().mean())


res = {}
for cat in CORE:
    for per in (EARLY, LATE):
        sub = d[(d.category == cat) & (d.q == per)]
        res[(cat, per)] = curves(sub)

print('\nmodels rated per snapshot, and share within 10% of top:')
rows = []
for cat in CORE:
    a, b = res[(cat, EARLY)], res[(cat, LATE)]
    i10 = list(XS).index(10)
    rows.append((DC[cat], a[4], b[4], 100 * a[0][i10], 100 * b[0][i10],
                 a[1][i10], b[1][i10]))
print(pd.DataFrame(rows, columns=['category', f'n_models_{EARLY}',
                                  f'n_models_{LATE}', f'share10_{EARLY}',
                                  f'share10_{LATE}', f'count10_{EARLY}',
                                  f'count10_{LATE}'])
      .to_string(index=False, float_format=lambda x: f'{x:8.1f}'))

print('\nsame, in RATING POINTS (share within 50 points of the category top):')
rows = []
for cat in CORE:
    a, b = res[(cat, EARLY)], res[(cat, LATE)]
    i50 = list(PTS).index(50)
    i100 = list(PTS).index(100)
    rows.append((DC[cat], 100 * a[2][i50], 100 * b[2][i50],
                 100 * a[2][i100], 100 * b[2][i100]))
print(pd.DataFrame(rows, columns=['category', f'within50_{EARLY}',
                                  f'within50_{LATE}', f'within100_{EARLY}',
                                  f'within100_{LATE}'])
      .to_string(index=False, float_format=lambda x: f'{x:8.1f}'))


def draw(ax, cat, periods, colors, show_legend=False):
    for per, col in zip(periods, colors):
        r = res[(cat, per)]
        if r is None:
            continue
        ax.plot(XS, 100 * r[0], color=col, lw=1.6, label=str(per))
        ax.fill_between(XS, 100 * r[0], color=col, alpha=0.12, lw=0)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.set_title(DC[cat], fontsize=10, color=INK_2, pad=5)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc='lower right',
                  handlelength=1.5)


# ---- layout 1: one panel per category, both periods overlaid -------------
fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.2), sharex=True, sharey=True)
for i, (ax, cat) in enumerate(zip(axes.ravel(), CORE)):
    draw(ax, cat, (EARLY, LATE), (MUTED, BLUE), show_legend=(i == 0))
for ax in axes[1]:
    ax.set_xlabel('threshold $x$ (within $x\\%$ of top)')
for ax in axes[:, 0]:
    ax.set_ylabel('% of models')
fig.tight_layout(pad=0.7, w_pad=1.2, h_pad=1.2)
fig.savefig('near_top_by_cat.pdf')
plt.close(fig)

# ---- layout 2: rows are periods, columns are categories ------------------
fig, axes = plt.subplots(2, 8, figsize=(16.5, 4.6), sharex=True, sharey=True)
for j, cat in enumerate(CORE):
    for i, (per, col) in enumerate(((EARLY, MUTED), (LATE, BLUE))):
        ax = axes[i, j]
        draw(ax, cat, (per,), (col,))
        if i == 1:
            ax.set_title('')
            ax.set_xlabel('$x\\%$')
for i, per in enumerate((EARLY, LATE)):
    axes[i, 0].set_ylabel(f'{per}\n% of models')
fig.tight_layout(pad=0.6, w_pad=0.7, h_pad=0.9)
fig.savefig('near_top_by_cat_rows.pdf')
plt.close(fig)
print('\nwrote near_top_by_cat.pdf, near_top_by_cat_rows.pdf')

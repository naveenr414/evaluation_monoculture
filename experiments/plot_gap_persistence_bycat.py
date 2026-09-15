"""Quarter-to-quarter persistence of a firm's position, split by category.

  resid_gap_bycat.pdf       g_i,k(q)      vs  g_i,k(q+1)      -- absolute gap
  resid_relgap_bycat.pdf    gtilde_i,k(q) vs  gtilde_i,k(q+1) -- relative gap

where g is the distance to the category leader and gtilde is g with the firm's
own mean gap and the category mean gap removed, so it measures how lopsided a
firm is rather than how far behind it is.  Slope 1 is a random walk; slope
below 1 is catch-up.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'experiments')
from common import load_panel, flagships, CORE

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
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 11,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
flag = flagships(d)
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
dates = sorted(d.dt.unique())
last = d.dt.max()
snap = d[(d.dt == last) & (d.category == 'overall')].dropna(subset=['rating'])
ranked = (snap.sort_values('rating', ascending=False).groupby('org').head(1)
              .sort_values('rating', ascending=False))
ORGS = [o for o in ranked.org if o and o not in ('inception-ai',)][:23]

rows = []
for t in dates:
    prof = {}
    for o in ORGS:
        m = flag.get((t, o))
        if m is None:
            continue
        try:
            r = rat.loc[(t, m)][CORE]
        except KeyError:
            continue
        if r.isna().any():
            continue
        prof[o] = r.values.astype(float)
    if len(prof) < 6:
        continue
    nm = sorted(prof)
    P = np.array([prof[n] for n in nm])
    G = P.max(0, keepdims=True) - P
    Gt = G - G.mean(1, keepdims=True)      # remove the firm's own mean gap
    Gt = Gt - Gt.mean(0, keepdims=True)     # and the category mean
    for i, o in enumerate(nm):
        for j, c in enumerate(CORE):
            rows.append((t, o, c, G[i, j], Gt[i, j]))

p = pd.DataFrame(rows, columns=['dt', 'org', 'cat', 'gap', 'relgap'])
p['q'] = pd.PeriodIndex(p.dt, freq='Q')
q = p.groupby(['q', 'org', 'cat'])[['gap', 'relgap']].mean().reset_index()
quarters = sorted(q.q.unique())


def pairs(col, cat, lag=1):
    w = q[q.cat == cat].pivot_table(index='org', columns='q', values=col)
    xs, ys = [], []
    for i in range(len(quarters) - lag):
        a, b = w.get(quarters[i]), w.get(quarters[i + lag])
        if a is None or b is None:
            continue
        m = a.notna() & b.notna()
        xs.append(a[m].values); ys.append(b[m].values)
    return np.concatenate(xs), np.concatenate(ys)


def fit(x, y):
    b1, b0 = np.polyfit(x, y, 1)
    e = y - (b0 + b1 * x)
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    se = np.sqrt((e ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    return b0, b1, se, r2


def facet(col, outpath, color, sym, xlab, ylab, lag=1):
    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.4))
    res = []
    for ax, cat in zip(axes.ravel(), CORE):
        x, y = pairs(col, cat, lag)
        b0, b1, se, r2 = fit(x, y)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.07 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=BASELINE,
                lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.scatter(x, y, s=14, color=color, alpha=0.45, lw=0, zorder=2)
        gx = np.linspace(lo - pad, hi + pad, 10)
        ax.plot(gx, b0 + b1 * gx, color=INK, lw=1.4, zorder=3)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(DC[cat], fontsize=10.5, color=INK_2, pad=5)
        ax.text(0.04, 0.96, f'$\\beta={b1:.2f}$\n$R^2={r2:.2f}$',
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                color=INK_2)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ('left', 'bottom'))
        ax.tick_params(length=3, width=0.6)
        res.append((DC[cat], b1, se, r2, len(x)))
    for ax in axes[1]:
        ax.set_xlabel(xlab)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylab)
    fig.tight_layout(pad=0.7, w_pad=1.4, h_pad=1.4)
    fig.savefig(outpath)
    plt.close(fig)
    return pd.DataFrame(res, columns=['category', 'slope', 'se', 'R2', 'n'])


out = {}
for lag in (1, 2, 3):
    sfx = '' if lag == 1 else f'_q{lag}'
    out[('gap', lag)] = facet(
        'gap', f'resid_gap_bycat{sfx}.pdf', BLUE, 'g',
        r'$g_{i,k}$ at quarter $q$', rf'$g_{{i,k}}$ at quarter $q+{lag}$', lag)
    out[('relgap', lag)] = facet(
        'relgap', f'resid_relgap_bycat{sfx}.pdf', ORANGE, 'gt',
        r'$\tilde{g}_{i,k}$ at quarter $q$',
        rf'$\tilde{{g}}_{{i,k}}$ at quarter $q+{lag}$', lag)

for col, lab in (('gap', 'ABSOLUTE GAP'),
                 ('relgap', 'RELATIVE GAP (firm and category means removed)')):
    w = out[(col, 1)][['category']].copy()
    for lag in (1, 2, 3):
        t = out[(col, lag)].set_index('category')
        w[f'b_q+{lag}'] = w.category.map(t.slope)
        w[f'se_q+{lag}'] = w.category.map(t.se)
    w['n_q+3'] = w.category.map(out[(col, 3)].set_index('category').n)
    print(f'\n{lab}: slope by horizon')
    print(w.sort_values('b_q+1', ascending=False).to_string(
        index=False, float_format=lambda x: f'{x:6.3f}'))
print('\nwrote resid_{gap,relgap}_bycat{,_q2,_q3}.pdf')

"""Decile transition matrix: if a firm-category sits in decile s of the gap
distribution this quarter, where is it next quarter?

Decile cut points are taken from the pooled distribution of the INITIAL value
and applied unchanged to the next quarter, so the matrix is not forced to be
doubly stochastic and net drift toward smaller gaps is visible.  Rows sum to
one.  All eight categories pooled.

  gap        g_i,k = max_j theta_j,k - theta_i,k
  relgap     g with the firm's own mean gap and the category mean removed

Writes resid_gap_transition.pdf.
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
    Gt = G - G.mean(1, keepdims=True)
    Gt = Gt - Gt.mean(0, keepdims=True)
    for i, o in enumerate(nm):
        for j, c in enumerate(CORE):
            rows.append((t, o, c, G[i, j], Gt[i, j]))

p = pd.DataFrame(rows, columns=['dt', 'org', 'cat', 'gap', 'relgap'])
p['q'] = pd.PeriodIndex(p.dt, freq='Q')
q = p.groupby(['q', 'org', 'cat'])[['gap', 'relgap']].mean().reset_index()
quarters = sorted(q.q.unique())


def pairs(col, lag=1):
    w = q.pivot_table(index=['org', 'cat'], columns='q', values=col)
    xs, ys = [], []
    for i in range(len(quarters) - lag):
        a, b = w.get(quarters[i]), w.get(quarters[i + lag])
        if a is None or b is None:
            continue
        m = a.notna() & b.notna()
        xs.append(a[m].values); ys.append(b[m].values)
    return np.concatenate(xs), np.concatenate(ys)


def transition(x, y, nbin=10):
    """Cut points from the INITIAL distribution, applied to both periods."""
    edges = np.quantile(x, np.linspace(0, 1, nbin + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    sx = np.clip(np.digitize(x, edges[1:-1]), 0, nbin - 1)
    sy = np.clip(np.digitize(y, edges[1:-1]), 0, nbin - 1)
    M = np.zeros((nbin, nbin))
    for a, b in zip(sx, sy):
        M[a, b] += 1
    cnt = M.sum(1)
    T = M / np.where(cnt[:, None] == 0, 1, cnt[:, None])
    return T, cnt, edges, sx, sy


def panel(ax, T, cnt, title):
    im = ax.pcolormesh(T * 100, cmap='Blues', vmin=0, vmax=45,
                       edgecolors=SURFACE, linewidth=0.5)
    n = T.shape[0]
    ax.plot([0, n], [0, n], color='#c0392b', lw=1.0, ls=(0, (3, 2)))
    for i in range(n):
        for j in range(n):
            v = T[i, j] * 100
            if v >= 1:
                ax.text(j + 0.5, i + 0.5, f'{v:.0f}', ha='center', va='center',
                        fontsize=7.4,
                        color=SURFACE if v > 26 else INK_2)
    ax.set_xlim(0, n); ax.set_ylim(n, 0)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(n) + 0.5); ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(range(1, n + 1)); ax.set_yticklabels(range(1, n + 1))
    ax.set_xlabel('decile at quarter $q+1$')
    ax.set_ylabel('decile at quarter $q$')
    ax.set_title(title, fontsize=11, color=INK_2, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return im


fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
res = {}
for ax, (col, title) in zip(axes, [('gap', 'Gap to category leader'),
                                   ('relgap', 'Relative gap')]):
    x, y = pairs(col)
    T, cnt, edges, sx, sy = transition(x, y)
    im = panel(ax, T, cnt, title)
    res[col] = (T, cnt, edges, sx, sy, len(x))
cb = fig.colorbar(im, ax=axes, shrink=0.72, pad=0.02, aspect=24)
cb.outline.set_visible(False)
cb.set_label('probability (%)', fontsize=10.5, color=INK_2)
cb.ax.tick_params(length=2.5, labelsize=9, color=MUTED)
fig.savefig('resid_gap_transition.pdf', bbox_inches='tight')
plt.close(fig)

for col, lab in (('gap', 'GAP TO LEADER'), ('relgap', 'RELATIVE GAP')):
    T, cnt, edges, sx, sy, n = res[col]
    print('\n' + '=' * 70)
    print(f'{lab}   n = {n} firm-category-quarter transitions')
    print('=' * 70)
    print('decile cut points (from the initial distribution):',
          ', '.join(f'{e:.0f}' for e in edges[1:-1]))
    stay = np.diag(T)
    near = np.array([T[i, max(0, i - 1):i + 2].sum() for i in range(10)])
    drift = np.array([(np.arange(10) - i) @ T[i] for i in range(10)])
    out = pd.DataFrame({
        'decile': range(1, 11), 'n': cnt.astype(int),
        'stay_%': 100 * stay, 'within_1_%': 100 * near,
        'mean_move': drift})
    print(out.to_string(index=False, float_format=lambda x: f'{x:6.1f}'))
    print(f'  overall: stay {100*np.average(stay, weights=cnt):.1f}%, '
          f'within one decile {100*np.average(near, weights=cnt):.1f}%')
print('\nwrote resid_gap_transition.pdf')

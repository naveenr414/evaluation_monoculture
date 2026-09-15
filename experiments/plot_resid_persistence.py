"""RQ1 persistence of firm-level per-category position, quarter to quarter.

Firm level: at every snapshot each firm is represented by its flagship (highest
'overall'-rated) model, so the panel tracks the firm's current frontier rather
than a fixed model that decays as rivals ship.

Two measures of a firm's position in category k, both computed within a snapshot
and then averaged within a quarter:

  gap      g_i,k = max_j theta_j,k - theta_i,k     (distance to the category leader)
  residual r_i,k = theta_i,k - mean_k - mean_i     (two-way FE comparative advantage)

Figures (each: gap panel | residual panel)
  resid_dist_early_late.pdf    distribution at 2025Q1 vs the final quarter
  resid_persistence_q1.pdf     quarter q  vs quarter q+1, pooled over q
  resid_persistence_q4.pdf     quarter q  vs quarter q+4 (one year), pooled over q
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, 'experiments')
from common import load_panel, two_way_resid, flagships, CORE

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 11, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 11.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})


def strip_axes(ax, keep=('left', 'bottom')):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=3, width=0.6)


# ---------------------------------------------------------------- data -----
d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
flag = flagships(d)
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
dates = sorted(d.dt.unique())

last = d.dt.max()
snap_last = d[(d.dt == last) & (d.category == 'overall')].dropna(subset=['rating'])
ranked = (snap_last.sort_values('rating', ascending=False)
                    .groupby('org').head(1).sort_values('rating', ascending=False))
ORGS = [o for o in ranked.org if o and o not in ('inception-ai',)][:23]


def profile(t):
    out = {}
    for org in ORGS:
        m = flag.get((t, org))
        if m is None:
            continue
        try:
            r = rat.loc[(t, m)][CORE]
        except KeyError:
            continue
        if r.isna().any():
            continue
        out[org] = r.values.astype(float)
    return out


# long panel: one row per (snapshot, firm, category) with both measures
rows = []
fe_r2 = {}
for t in dates:
    prof = profile(t)
    if len(prof) < 6:
        continue
    names = sorted(prof)
    P = np.array([prof[n] for n in names])
    G = P.max(0, keepdims=True) - P            # gap to the category leader
    R = two_way_resid(P)
    # share of raw cross-(firm,category) variance absorbed by firm+category means
    Pc = P - P.mean()
    fe_r2[t] = 1 - (R ** 2).sum() / (Pc ** 2).sum()
    for i, org in enumerate(names):
        for j, cat in enumerate(CORE):
            rows.append((t, org, cat, G[i, j], R[i, j], P[i, j]))

panel = pd.DataFrame(rows, columns=['dt', 'org', 'cat', 'gap', 'resid', 'theta'])
panel['q'] = pd.PeriodIndex(panel.dt, freq='Q')

q = (panel.groupby(['q', 'org', 'cat'])[['gap', 'resid', 'theta']]
          .mean().reset_index())
quarters = sorted(q.q.unique())
print('quarters:', [str(x) for x in quarters])
print('snapshots per quarter:',
      panel.groupby('q').dt.nunique().to_dict())
print('firms per quarter:', q.groupby('q').org.nunique().to_dict())

EARLY, LATE = quarters[0], quarters[-1]

MEAS = [('gap', r'gap to category leader  $g_{i,k}$', 'Gap to leader'),
        ('resid', r'two-way residual  $r_{i,k}$', 'Two-way residual')]

# ==================== FIGURE 1: distribution, early vs late ================
print('\n' + '=' * 74)
print(f'(1) DISTRIBUTION  {EARLY} vs {LATE}')
print('=' * 74)
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.05))
for ax, (col, xlab, title) in zip(axes, MEAS):
    a = q[(q.q == EARLY)][col].dropna().values
    b = q[(q.q == LATE)][col].dropna().values
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pad = 0.08 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 500)
    for vals, color, lab in ((a, MUTED, str(EARLY)), (b, BLUE, str(LATE))):
        kde = gaussian_kde(vals, bw_method=0.28)(xs)
        ax.fill_between(xs, kde, color=color, alpha=0.16, lw=0)
        ax.plot(xs, kde, color=color, lw=1.6, label=lab)
        ax.axvline(vals.mean(), color=color, lw=0.9, ls=':')
    ax.set_xlabel(xlab)
    ax.set_ylabel('density')
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.legend(frameon=False, fontsize=10, loc='upper right', handlelength=1.5)
    print(f'\n  {title}:')
    print(f'    {EARLY}: n={len(a):3d}  mean={a.mean():+7.2f}  sd={a.std(ddof=1):6.2f}  '
          f'IQR={np.percentile(a,75)-np.percentile(a,25):6.2f}')
    print(f'    {LATE}: n={len(b):3d}  mean={b.mean():+7.2f}  sd={b.std(ddof=1):6.2f}  '
          f'IQR={np.percentile(b,75)-np.percentile(b,25):6.2f}')
    # R^2 of "which quarter" as a predictor = eta^2 of the early/late split
    y = np.concatenate([a, b])
    grp = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
    print(f'    R^2 of the early/late split (eta^2) = {1 - ss_res/ss_tot:.4f}  '
          f'(variance ratio late/early = {b.var(ddof=1)/a.var(ddof=1):.3f})')
fig.tight_layout(pad=0.7, w_pad=2.0)
fig.savefig('resid_dist_early_late.pdf')
plt.close(fig)

r2e = np.mean([v for t, v in fe_r2.items() if pd.Timestamp(t).to_period('Q') == EARLY])
r2l = np.mean([v for t, v in fe_r2.items() if pd.Timestamp(t).to_period('Q') == LATE])
print(f'\n  share of raw theta variance absorbed by firm+category means (FE R^2): '
      f'{EARLY} {r2e:.4f} -> {LATE} {r2l:.4f}')
print(f'  (the complement, {1-r2e:.4f} -> {1-r2l:.4f}, is the comparative-advantage '
      f'residual share)')


# ==================== FIGURES 2 & 3: persistence scatters ==================
def lag_pairs(col, lag):
    """(x at quarter qi, y at quarter qi+lag) over all firms/cats/quarters."""
    w = q.pivot_table(index=['org', 'cat'], columns='q', values=col)
    xs, ys = [], []
    for i in range(len(quarters) - lag):
        a, b = w.get(quarters[i]), w.get(quarters[i + lag])
        if a is None or b is None:
            continue
        m = a.notna() & b.notna()
        xs.append(a[m].values); ys.append(b[m].values)
    return np.concatenate(xs), np.concatenate(ys)


def scatter_panel(ax, x, y, xlab, ylab, color):
    b1, b0 = np.polyfit(x, y, 1)
    pred = b0 + b1 * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    pad = 0.06 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=BASELINE, lw=0.9,
            ls=(0, (3, 2)), zorder=1)
    ax.scatter(x, y, s=13, color=color, alpha=0.38, lw=0, zorder=2)
    gx = np.linspace(lo - pad, hi + pad, 10)
    ax.plot(gx, b0 + b1 * gx, color=INK, lw=1.4, zorder=3)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_axes(ax)
    ax.text(0.035, 0.955, f'$R^2={r2:.3f}$\nslope $={b1:.3f}$\n$n={len(x)}$',
            transform=ax.transAxes, va='top', ha='left', fontsize=10, color=INK_2)
    return r2, b1


for lag, fname, label in ((1, 'resid_persistence_q1.pdf', 'q -> q+1  (one quarter)'),
                          (4, 'resid_persistence_q4.pdf', 'q -> q+4  (one year)')):
    print('\n' + '=' * 74)
    print(f'({2 if lag == 1 else 3}) PERSISTENCE  {label}')
    print('=' * 74)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax, (col, _, title) in zip(axes, MEAS):
        x, y = lag_pairs(col, lag)
        sym = 'g' if col == 'gap' else 'r'
        r2, b1 = scatter_panel(
            ax, x, y, f'${sym}_{{i,k}}$ at quarter $q$',
            f'${sym}_{{i,k}}$ at quarter $q+{lag}$',
            BLUE if col == 'gap' else ORANGE)
        print(f'  {title:<18} R^2={r2:.4f}  slope={b1:+.4f}  '
              f'corr={np.corrcoef(x,y)[0,1]:+.4f}  n={len(x)}')
    fig.tight_layout(pad=0.7, w_pad=2.0)
    fig.savefig(fname)
    plt.close(fig)

# ------------------------------- decay curve ------------------------------
print('\n' + '=' * 74)
print('R^2 BY LAG (quarters)')
print('=' * 74)
print(f'  {"lag":>4} {"gap R^2":>9} {"gap slope":>10} {"resid R^2":>10} '
      f'{"resid slope":>12} {"n":>6}')
for lag in range(1, len(quarters)):
    xg, yg = lag_pairs('gap', lag)
    xr, yr = lag_pairs('resid', lag)
    if len(xg) < 20:
        continue
    def fit(x, y):
        b1, b0 = np.polyfit(x, y, 1)
        return 1 - ((y - (b0 + b1 * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b1
    r2g, sg = fit(xg, yg); r2r, sr = fit(xr, yr)
    print(f'  {lag:>4} {r2g:>9.4f} {sg:>10.3f} {r2r:>10.4f} {sr:>12.3f} {len(xg):>6}')

print('\nwrote resid_dist_early_late.pdf, resid_persistence_q1.pdf, '
      'resid_persistence_q4.pdf')

"""How fast does a firm's per-category position revert?  Slope of

    y_{i,k}(t+h)  ~  y_{i,k}(t)

as a function of the horizon h, in months.  Slope 1 = random walk (no catch-up);
slope < 1 = the position closes toward zero.  Compared against the pure-AR(1)
benchmark rho_1^h, which a fully mean-reverting series must follow: sitting
ABOVE it means part of the position is permanent.

  y = gap      g_i,k = max_j theta_j,k - theta_i,k
  y = resid    r_i,k = two-way (firm x category) FE residual

Firm level throughout: each firm is its flagship model at each snapshot.
Outputs resid_slope_vs_horizon.pdf.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'experiments')
from common import load_panel, two_way_resid, flagships, CORE

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

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
    R = two_way_resid(P)
    for i, o in enumerate(nm):
        for j, c in enumerate(CORE):
            rows.append((t, o, c, G[i, j], R[i, j]))

p = pd.DataFrame(rows, columns=['dt', 'org', 'cat', 'gap', 'resid'])
p['m'] = pd.PeriodIndex(p.dt, freq='M')
mo = p.groupby(['m', 'org', 'cat'])[['gap', 'resid']].mean().reset_index()
months = sorted(mo.m.unique())
print(f'{len(months)} months: {months[0]} .. {months[-1]}')
print('firms per month:', mo.groupby("m").org.nunique().to_dict())

W = {c: mo.pivot_table(index=['org', 'cat'], columns='m', values=c)
     for c in ('gap', 'resid')}


def lag_frame(col, h):
    out = []
    for i in range(len(months) - h):
        a, b = W[col].get(months[i]), W[col].get(months[i + h])
        if a is None or b is None:
            continue
        m = a.notna() & b.notna()
        if m.sum() == 0:
            continue
        out.append(pd.DataFrame({
            'x': a[m].values, 'y': b[m].values,
            'org': [k[0] for k in a[m].index], 'cat': [k[1] for k in a[m].index]}))
    return pd.concat(out, ignore_index=True) if out else None


def ols_cluster(x, y, cl):
    """OLS slope with SE clustered on `cl` (firms) -- pairs overlap heavily."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for g in np.unique(cl):
        idx = cl == g
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    G = len(np.unique(cl))
    scale = G / max(G - 1, 1)
    V = XtX_inv @ meat @ XtX_inv * scale
    return beta[1], np.sqrt(V[1, 1])


def demean(D, cols=('org', 'cat')):
    D = D.copy()
    for g in cols:
        D['x'] = D.x - D.groupby(g).x.transform('mean')
        D['y'] = D.y - D.groupby(g).y.transform('mean')
    return D


HMAX = 16
res = {k: {'h': [], 'b': [], 'se': [], 'n': []}
       for k in ('gap', 'gap_fe', 'resid')}
for h in range(1, HMAX + 1):
    D = lag_frame('gap', h)
    if D is None or len(D) < 40:
        continue
    b, se = ols_cluster(D.x.values, D.y.values, D.org.values)
    res['gap']['h'].append(h); res['gap']['b'].append(b)
    res['gap']['se'].append(se); res['gap']['n'].append(len(D))
    Df = demean(D)
    b, se = ols_cluster(Df.x.values, Df.y.values, Df.org.values)
    res['gap_fe']['h'].append(h); res['gap_fe']['b'].append(b)
    res['gap_fe']['se'].append(se); res['gap_fe']['n'].append(len(Df))

    D = lag_frame('resid', h)
    b, se = ols_cluster(D.x.values, D.y.values, D.org.values)
    res['resid']['h'].append(h); res['resid']['b'].append(b)
    res['resid']['se'].append(se); res['resid']['n'].append(len(D))

print('\nslope by horizon (SE clustered on firm)')
print(f'{"h(mo)":>6} {"gap":>16} {"gap|firm,cat FE":>18} {"two-way resid":>16} {"n":>6}')
for i, h in enumerate(res['gap']['h']):
    print(f'{h:>6} {res["gap"]["b"][i]:>9.3f} ({res["gap"]["se"][i]:.3f}) '
          f'{res["gap_fe"]["b"][i]:>11.3f} ({res["gap_fe"]["se"][i]:.3f}) '
          f'{res["resid"]["b"][i]:>9.3f} ({res["resid"]["se"][i]:.3f}) '
          f'{res["gap"]["n"][i]:>6}')

# ------------------------------------------------------------------ plot ---
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
hs = np.array(res['gap']['h'], dtype=float)

specs = [
    (axes[0], [('gap', BLUE, 'gap to leader, pooled'),
               ('gap_fe', AQUA, r'gap, within firm$\times$category FE')],
     'Gap to category leader'),
    (axes[1], [('resid', ORANGE, 'two-way residual')],
     'Two-way FE residual'),
]
for ax, series, title in specs:
    ax.axhline(1, color=BASELINE, lw=0.9, ls=(0, (3, 2)))
    ax.axhline(0, color=BASELINE, lw=0.7)
    for key, color, lab in series:
        b = np.array(res[key]['b']); se = np.array(res[key]['se'])
        ax.fill_between(hs, b - 1.96 * se, b + 1.96 * se, color=color,
                        alpha=0.15, lw=0)
        ax.plot(hs, b, color=color, lw=1.7, marker='o', ms=3.4, label=lab)
        # pure-AR(1) benchmark anchored on the 1-month slope
        ax.plot(hs, b[0] ** hs, color=color, lw=1.1, ls=':',
                label=r'AR(1) benchmark $\rho_1^{h}$')
    ax.set_xlabel('horizon $h$ (months)')
    ax.set_title(title, fontsize=11.5, color=INK_2, pad=7)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0.4, HMAX + 0.6)
    ax.set_ylim(-0.15, 1.12)
    strip_axes(ax)
    ax.legend(frameon=False, fontsize=9.2, loc='lower left', handlelength=1.8)
axes[0].set_ylabel(r'slope of $y(t+h)$ on $y(t)$')
fig.tight_layout(pad=0.7, w_pad=1.6)
fig.savefig('resid_slope_vs_horizon.pdf')
plt.close(fig)
print('\nwrote resid_slope_vs_horizon.pdf')

# half-lives from the 1-month slopes
for key, lab in (('gap', 'gap pooled'), ('gap_fe', 'gap within FE'),
                 ('resid', 'two-way residual')):
    r1 = res[key]['b'][0]
    hl = np.log(0.5) / np.log(r1) if 0 < r1 < 1 else float('nan')
    print(f'  {lab:<20} rho_1mo={r1:.3f}  {100*(1-r1):4.1f}%/month  '
          f'half-life {hl:5.2f} months  implied 12-mo {r1**12:.3f} '
          f'vs observed {res[key]["b"][11]:.3f}')

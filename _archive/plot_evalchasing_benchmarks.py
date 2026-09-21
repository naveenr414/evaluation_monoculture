"""Does the gap-chasing coefficient depend on which rival defines the gap?

The draft measures the gap to the per-category leader.  We re-estimate beta with
four benchmarks, holding the specification fixed (firm x category and category x
quarter fixed effects, standard errors clustered on the release event):

  (a) best rival          max_j theta_j,k          -- the draft's measure
  (b) second-best rival   the runner-up in k
  (c) top-3 average       mean of the three best in k
  (d) fixed rival         one firm, the same for every event and category

Panels plot the within-transformed dtheta against the within-transformed gap, so
the plotted slope is the reported beta.  Writes evalchasing_benchmarks.pdf.
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (load_panel, flagships, fe_regress, within, CORE, FRONTIER,
                    MATURE_DAYS)

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 11,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

d = load_panel(cats=CORE)
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
var = d.pivot_table(index=['dt', 'model_name'], columns='category', values='variance')
flag = flagships(d)
snaps = np.sort(d.dt.unique())

# which firm leads a category most often?  that firm is the fixed benchmark
wins = {f: 0 for f in FRONTIER}
for t in snaps:
    present = {f: flag.get((t, f)) for f in FRONTIER}
    present = {f: m for f, m in present.items() if m is not None}
    if len(present) < 4:
        continue
    for k in CORE:
        best, bv = None, -np.inf
        for f, m in present.items():
            try:
                v = rat.loc[(t, m)][k]
            except KeyError:
                continue
            if pd.notna(v) and v > bv:
                best, bv = f, v
        if best:
            wins[best] += 1
FIXED = max(wins, key=wins.get)
print('category-leadership counts across snapshots:',
      {k: v for k, v in sorted(wins.items(), key=lambda kv: -kv[1])})
print(f'fixed benchmark firm = {FIXED}\n')

# ------------------------------------------------------------- events ------
events = []
for firm in FRONTIER:
    if firm not in flag.index.get_level_values('org'):
        continue
    ser = flag.xs(firm, level='org').sort_index()
    prev_m, prev_t = None, None
    for t, m in ser.items():
        if prev_m is not None and m != prev_m:
            events.append((firm, prev_m, m, t, prev_t))
        prev_m, prev_t = m, t

rows = []
for firm, old, new, switch, t_old in events:
    cand = [s for s in snaps if s >= switch + np.timedelta64(MATURE_DAYS, 'D')]
    if not cand:
        continue
    t_new = cand[0]
    try:
        th_old, th_new = rat.loc[(t_old, old)][CORE], rat.loc[(t_new, new)][CORE]
        v_old = var.loc[(t_old, old)][CORE]
    except KeyError:
        continue
    if th_old.isna().any() or th_new.isna().any():
        continue

    snap = rat.xs(t_old, level='dt')
    rivals = [(f, flag.get((t_old, f))) for f in FRONTIER if f != firm]
    rivals = [(f, m) for f, m in rivals if m is not None and m in snap.index]
    riv = snap.loc[[m for _, m in rivals]][CORE].dropna()
    if len(riv) < 4:
        continue
    riv_firm = {m: f for f, m in rivals}

    fixed_m = flag.get((t_old, FIXED)) if firm != FIXED else None
    for k in CORE:
        s = riv[k].sort_values(ascending=False)
        if len(s) < 3:
            continue
        b1, b2 = s.iloc[0], s.iloc[1]
        b3 = s.iloc[:3].mean()
        v1 = var.loc[(t_old, s.index[0])][k]
        v2 = var.loc[(t_old, s.index[1])][k]
        v3 = np.mean([var.loc[(t_old, m)][k] for m in s.index[:3]]) / 3.0
        if fixed_m is not None and fixed_m in snap.index and pd.notna(snap.loc[fixed_m, k]):
            bf = snap.loc[fixed_m, k]
            vf = var.loc[(t_old, fixed_m)][k]
        else:
            bf = vf = np.nan
        rows.append(dict(
            event=f'{firm}:{old}->{new}@{pd.Timestamp(switch).strftime("%Y%m%d")}',
            firm=firm, cat=k, switch=switch,
            dtheta=th_new[k] - th_old[k],
            gap_best=b1 - th_old[k], v_best=v1,
            gap_2nd=b2 - th_old[k], v_2nd=v2,
            gap_top3=b3 - th_old[k], v_top3=v3,
            gap_fixed=bf - th_old[k], v_fixed=vf,
            v_old=v_old[k]))

ev = pd.DataFrame(rows)
ev['q'] = pd.PeriodIndex(ev.switch, freq='Q').astype(str)
ev['ik'] = ev.firm + '|' + ev.cat
ev['kt'] = ev.cat + '|' + ev.q
print(f'events={ev.event.nunique()}  obs={len(ev)}  firms={ev.firm.nunique()}')

SPECS = [('gap_best', 'v_best', '(a) best rival', BLUE),
         ('gap_2nd', 'v_2nd', '(b) second-best rival', ORANGE),
         ('gap_top3', 'v_top3', '(c) top-3 average', AQUA),
         ('gap_fixed', 'v_fixed', f'(d) fixed rival ({FIXED})', VIOLET)]

fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.5))
out = []
for ax, (gcol, vcol, title, color) in zip(axes, SPECS):
    df = ev.dropna(subset=[gcol, vcol]).copy()
    y, X = df.dtheta.values, df[[gcol]].values.astype(float)
    groups = [df.ik.values, df.kt.values]
    n = len(df)
    ndof = df.ik.nunique() + df.kt.nunique() - 1
    dfac = (n - ndof) / n
    nc = np.array([df.v_old.mean() * dfac])
    nv = np.array([[(df.v_old.mean() + df[vcol].mean()) * dfac]])
    beta, se, beta_c, resid, _, _ = fe_regress(y, X, groups, noise_cov=nc,
                                            noise_var=nv, cluster=df.event.values)
    xw = within(X[:, 0], *groups)
    yw = within(y, *groups)
    r2 = 1 - (resid @ resid) / ((yw - yw.mean()) ** 2).sum()

    ax.axhline(0, color=BASELINE, lw=0.7)
    ax.axvline(0, color=BASELINE, lw=0.7)
    ax.scatter(xw, yw, s=15, color=color, alpha=0.42, lw=0, zorder=2)
    gx = np.linspace(xw.min(), xw.max(), 10)
    ax.plot(gx, beta[0] * gx, color=INK, lw=1.5, zorder=3)
    ax.set_title(title, fontsize=10.5, color=INK_2, pad=6)
    ax.set_xlabel('gap (within FE)')
    ax.text(0.04, 0.96,
            f'$\\beta={beta[0]:.3f}$ ({se[0]:.3f})\n'
            f'corr. $={beta_c[0]:.3f}$\n$R^2={r2:.2f}$, $n={n}$',
            transform=ax.transAxes, va='top', ha='left', fontsize=9, color=INK_2)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
    out.append((title, beta[0], se[0], beta[0] / se[0], beta_c[0], r2, n,
                df.event.nunique(), df[gcol].mean()))
axes[0].set_ylabel(r'$\Delta\theta_{i,k}$ (within FE)')
fig.tight_layout(pad=0.7, w_pad=1.5)
fig.savefig('evalchasing_benchmarks.pdf')
plt.close(fig)

t = pd.DataFrame(out, columns=['benchmark', 'beta', 'se', 't', 'beta_corr',
                               'R2', 'n', 'events', 'mean_gap'])
print('\nfirm x category + category x quarter FE, clustered on event')
print(t.to_string(index=False, float_format=lambda x: f'{x:7.3f}'))
print('\nwrote evalchasing_benchmarks.pdf')

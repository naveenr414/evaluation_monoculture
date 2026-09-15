"""RQ2 (eval-chasing): per-firm slopes of dtheta on gap.

Event = a firm's flagship changeover; for event e, category k:
    gap_{e,k}    = rivals' per-category frontier - outgoing flagship, at t_old
    dtheta_{e,k} = new flagship rating - outgoing flagship rating
Pooled across firms (exp1_evalchasing.py) this gives one beta. Here we split by
firm: does each top lab show the same "close the gap" slope, or do some chase
harder than others? Both gap and dtheta are residualized on event FE + category
FE first (common.within) -- otherwise the scatter is dominated by which
category happens to move more on average and by how big an event's overall
jump was, not by the within-event "which category lagged more" comparison that
identifies eval-chasing.

  1. evalchasing_scatter.pdf   FE-residualized gap vs dtheta, one color per firm,
                                each firm's own OLS line overlaid
  2. evalchasing_forest.pdf    beta-hat +/- 95% CI per firm (cluster-robust by
                                event), sorted high to low; noise-corrected
                                beta shown as a secondary marker
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'experiments')
from common import load_panel, build_events, fe_regress, within, CORE, FRONTIER

INK, INK_2, MUTED, GRID, BASELINE, SURFACE = \
    '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'

# same Okabe-Ito (CVD-safe) hues used in plot_persistent_advantage.py, so a firm
# keeps its color across figures in the paper
FIRM_COLOR = {
    'openai': '#E69F00', 'anthropic': '#CC79A7', 'google': '#0072B2',
    'meta': '#009E73', 'mistral': '#56B4E9', 'xai': '#D55E00',
    'deepseek': '#F0E442', 'alibaba': '#000000',
}
DISPLAY = {'openai': 'OpenAI', 'anthropic': 'Anthropic', 'google': 'Google',
           'meta': 'Meta', 'mistral': 'Mistral', 'xai': 'xAI',
           'deepseek': 'DeepSeek', 'alibaba': 'Alibaba'}
MIN_EVENTS = 3   # below this, cluster-robust SE by event is degenerate (meta: n=2)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 23, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 25,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 21, 'ytick.labelsize': 21,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.linewidth': 1.2, 'pdf.fonttype': 42,
})


def strip_axes(ax, keep=('left', 'bottom')):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=3, width=0.6)


d = load_panel(cats=CORE)
ev = build_events(d, cats=CORE)
print(f'events={ev.event.nunique()}  firms={ev.firm.nunique()}  obs={len(ev)}')
print(ev.groupby('firm').event.nunique().to_string())

firms = [f for f in FRONTIER if f in ev.firm.unique()
         and ev[ev.firm == f].event.nunique() >= MIN_EVENTS]
dropped = [f for f in FRONTIER if f in ev.firm.unique() and f not in firms]
if dropped:
    print(f'dropped for too few events (<{MIN_EVENTS}): {dropped}')

results = {}
for firm in firms:
    sub = ev[ev.firm == firm].copy()
    n = len(sub)
    Gd = within(sub.gap.values, sub.event.values, sub.cat.values)
    Yd = within(sub.dtheta.values, sub.event.values, sub.cat.values)
    X = Gd.reshape(-1, 1)
    dfac = (n - sub.event.nunique() - sub.cat.nunique() + 1) / n
    s2_old = sub.v_old.mean() * dfac
    s2_tot = (sub.v_old.mean() + sub.v_front.mean()) * dfac
    beta, se, beta_c, resid, _ = fe_regress(
        sub.dtheta.values, sub.gap.values.reshape(-1, 1),
        [sub.event.values, sub.cat.values],
        noise_cov=np.array([s2_old]), noise_var=np.array([[s2_tot]]),
        cluster=sub.event.values)
    results[firm] = dict(n=n, n_ev=sub.event.nunique(), gap=Gd, dtheta=Yd,
                          beta=beta[0], se=se[0], beta_c=beta_c[0])
    print(f'{firm:10s} n_ev={sub.event.nunique():3d} n={n:3d}  '
          f'beta={beta[0]:+.3f} (se={se[0]:.3f})  corrected={beta_c[0]:+.3f}')

# ============================================================= FIGURE 1 ====
fig, ax = plt.subplots(figsize=(13.0, 5.2))
# cap at roughly the 97th percentile of |gap| across firms -- a couple of
# single-event outliers otherwise stretch the fitted lines into a dramatic
# fan that's mostly extrapolation past where any data actually sits
all_gap = np.concatenate([r['gap'] for r in results.values()])
xmax = np.percentile(np.abs(all_gap), 97)

for firm in firms:
    r = results[firm]
    color = FIRM_COLOR[firm]
    ax.scatter(r['gap'], r['dtheta'], s=20, color=color, alpha=0.55,
               edgecolor='none', zorder=2)
    xs = np.linspace(-xmax, xmax, 50)
    ax.plot(xs, r['beta'] * xs, color=color, lw=3.8, zorder=3,
            label=f'{DISPLAY[firm]}  ($\\hat\\beta$={r["beta"]:+.2f})')

all_dtheta = np.concatenate([r['dtheta'] for r in results.values()])
ymax = np.percentile(np.abs(all_dtheta), 97)

ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-ymax, ymax)
ax.set_xlabel(r'$g_{i,k}$')
ax.set_ylabel(r'$\Delta\theta_{i,k}$')
ax.grid(color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
ax.legend(frameon=False, fontsize=21, loc='upper left', bbox_to_anchor=(1.01, 1.03),
          handlelength=1.4, labelspacing=0.7, borderaxespad=0.3)
fig.tight_layout(pad=0.7, rect=[0, 0, 0.72, 1])
fig.savefig('evalchasing_scatter.pdf', bbox_inches='tight')
plt.close(fig)

# ============================================================= FIGURE 2 ====
order = sorted(firms, key=lambda f: results[f]['beta_c'])  # low to high -> plotted bottom to top
fig, ax = plt.subplots(figsize=(9.0, 4.4))

for i, firm in enumerate(order):
    r = results[firm]
    color = FIRM_COLOR[firm]
    lo, hi = r['beta_c'] - 1.96 * r['se'], r['beta_c'] + 1.96 * r['se']
    ax.plot([lo, hi], [i, i], color=color, lw=5.0, zorder=2, solid_capstyle='round')
    ax.scatter([r['beta_c']], [i], color=color, s=190, zorder=3, edgecolor=SURFACE, lw=1.4)

ax.axvline(0, color=BASELINE, lw=0.9, zorder=1)
ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([DISPLAY[f] for f in order])
ax.set_xlabel(r'$\hat\beta$ 95% Confidence Interval')
ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax, keep=('bottom',))
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
fig.tight_layout(pad=0.7)
fig.savefig('evalchasing_forest.pdf')
plt.close(fig)
print(f'\ndropped from plots (n_ev < {MIN_EVENTS}): {dropped}')

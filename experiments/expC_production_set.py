"""RQ3. The identified set of the production matrix A_i.

Model: theta_i,k = a_i,k log(1 + x_i,k), investment x unobserved, budget
sum_k x_i,k = c_i, firm maximises sum_k w_k theta_i,k.

The first-order condition is w_k a_i,k = lambda exp(u_k) with u_k = log(1+x_i,k).
Substituting a_i,k = theta_i,k / u_k gives

        u_k exp(u_k) = w_k theta_i,k / lambda      ->   u_k = W(w_k theta_i,k / lambda)

with W the Lambert function, and lambda fixed by sum_k (exp(u_k) - 1) = c_i.
So for every (c_i, w) the matrix is point identified, and the identified set is
swept out by varying c_i and w.  That is Theorem 9 made arithmetic: without
variation in (c, w) the level of A_i is free.

Reports, for each firm and category, the range of a_i,k over a budget sweep, and
tests whether the RELATIVE pattern across categories is invariant to the budget.
Writes production_identified_set.pdf.
"""
import sys
import numpy as np
import pandas as pd
from scipy.special import lambertw
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'experiments')
from common import load_panel, flagships, CORE, FRONTIER

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'

DO = {'openai': 'OpenAI', 'anthropic': 'Anthropic', 'google': 'Google',
      'alibaba': 'Alibaba', 'meta': 'Meta', 'mistral': 'Mistral', 'xai': 'xAI',
      'deepseek': 'DeepSeek'}
DC = {'coding': 'Coding', 'math': 'Math', 'hard_prompts': 'Hard Pr.',
      'instruction_following': 'Instr. Fol.', 'creative_writing': 'Creat. Wr.',
      'longer_query': 'Longer Q.', 'multi_turn': 'Multi-Turn',
      'english': 'English'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 10.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

SNAP = pd.Timestamp('2026-03-19')     # before the organization-field break
C_GRID = np.logspace(-1, 3, 41)        # budget sweep, 0.1 to 1000


def solve_u(theta, w, c):
    """u_k = W(w_k theta_k / lambda) with lambda set by sum(exp(u)-1) = c."""
    def spend(loglam):
        u = np.real(lambertw(w * theta / np.exp(loglam)))
        return np.sum(np.exp(u) - 1) - c
    lo, hi = -40.0, 40.0
    if spend(lo) < 0 or spend(hi) > 0:
        return None
    loglam = brentq(spend, lo, hi, xtol=1e-12, rtol=1e-14)
    return np.real(lambertw(w * theta / np.exp(loglam)))


d = load_panel(cats=CORE, firms_only=False, since=None)
d = d[d.dt <= SNAP]
flag = flagships(d)
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
t = d.dt.max()

# theta measured above a floor: the weakest rated model in that category, so
# theta > 0.  The floor is itself a free normalisation and is swept below.
field = d[(d.dt == t) & (d.category.isin(CORE))].dropna(subset=['rating'])
floor = field.groupby('category').rating.min()[CORE].values

# evaluation weights: Arena vote counts per category, and uniform
vc = field.groupby('category').vote_count.sum().reindex(CORE)
W_VOTE = (vc / vc.sum()).values
W_UNIF = np.ones(len(CORE)) / len(CORE)
print(f'snapshot {pd.Timestamp(t).date()}')
print('vote-share weights:',
      {DC[c]: round(v, 3) for c, v in zip(CORE, W_VOTE)})

firms, TH = [], []
for f in FRONTIER:
    m = flag.get((t, f))
    if m is None:
        continue
    try:
        r = rat.loc[(t, m)][CORE]
    except KeyError:
        continue
    if r.isna().any():
        continue
    firms.append(f)
    TH.append(r.values.astype(float) - floor)
TH = np.array(TH)
print(f'{len(firms)} firms: {[DO.get(f, f) for f in firms]}')
print('theta above the category floor:')
print(pd.DataFrame(TH, index=[DO.get(f, f) for f in firms],
                   columns=[DC[c] for c in CORE]).round(0).to_string())

# ------------------------------------------------- sweep the budget -------
recs = []
for wi, (wlab, w) in enumerate((('votes', W_VOTE), ('uniform', W_UNIF))):
    for i, f in enumerate(firms):
        for c in C_GRID:
            u = solve_u(TH[i], w, c)
            if u is None or np.any(u <= 0):
                continue
            a = TH[i] / u
            for j, cat in enumerate(CORE):
                recs.append(dict(w=wlab, firm=f, cat=cat, c=c, a=a[j],
                                 u=u[j], x=np.exp(u[j]) - 1))
A = pd.DataFrame(recs)
print(f'\nsolved {A.groupby(["w","firm"]).c.nunique().min()}-'
      f'{A.groupby(["w","firm"]).c.nunique().max()} budget points per firm')

print('\n(1) IDENTIFIED SET OF a_i,k OVER THE BUDGET SWEEP (vote weights)')
sub = A[A.w == 'votes']
tab = sub.groupby(['firm', 'cat']).a.agg(['min', 'max'])
tab['ratio'] = tab['max'] / tab['min']
piv_min = tab['min'].unstack()[CORE]
piv_max = tab['max'].unstack()[CORE]
piv_min.index = [DO.get(f, f) for f in piv_min.index]
piv_max.index = [DO.get(f, f) for f in piv_max.index]
piv_min.columns = [DC[c] for c in CORE]
piv_max.columns = [DC[c] for c in CORE]
print('  min over the sweep:')
print(piv_min.round(1).to_string())
print('  max over the sweep:')
print(piv_max.round(1).to_string())
print(f'  max/min ratio: median {tab.ratio.median():.1f}, '
      f'range {tab.ratio.min():.1f} to {tab.ratio.max():.1f}')

print('\n(2) IS THE RELATIVE PATTERN INVARIANT TO THE BUDGET?')
print('  row-normalised a (a_i,k / mean_k a_i,k) at three budgets, vote weights:')
for c in (C_GRID[5], C_GRID[20], C_GRID[35]):
    s = sub[np.isclose(sub.c, c)]
    p = s.pivot_table(index='firm', columns='cat', values='a')[CORE]
    p = p.div(p.mean(axis=1), axis=0)
    print(f'   c={c:8.2f}  cross-firm sd of the normalised entries = '
          f'{p.values.std():.4f}')
rho = []
for f in firms:
    s = sub[sub.firm == f]
    p = s.pivot_table(index='c', columns='cat', values='a')[CORE]
    pn = p.div(p.mean(axis=1), axis=0)
    lo, hi = pn.iloc[0].values, pn.iloc[-1].values
    rho.append((DO.get(f, f), np.corrcoef(lo, hi)[0, 1],
                float(np.abs(lo - hi).max())))
print('  correlation of the normalised profile between the smallest and the')
print('  largest budget, per firm:')
for f, r, mx in rho:
    print(f'    {f:<10} corr {r:+.4f}   max shift {mx:.4f}')

# ------------------------------------------------------------------ plot ---
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))

ax = axes[0]
cmap = plt.get_cmap('viridis')
for j, cat in enumerate(CORE):
    s = sub[(sub.firm == 'openai') & (sub.cat == cat)].sort_values('c')
    ax.plot(s.c, s.a, color=cmap(j / (len(CORE) - 1)), lw=1.5, label=DC[cat])
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('budget $c_i$')
ax.set_ylabel('$a_{i,k}$')
ax.set_title('OpenAI: $a_{i,k}$ across the budget sweep',
             fontsize=10.5, color=INK_2, pad=6)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper right')

ax = axes[1]
c_ref = C_GRID[20]
s = sub[np.isclose(sub.c, c_ref)]
P = s.pivot_table(index='firm', columns='cat', values='a')[CORE]
P = P.div(P.mean(axis=1), axis=0)
P.index = [DO.get(f, f) for f in P.index]
im = ax.pcolormesh(P.values, cmap='RdBu_r', vmin=0.7, vmax=1.3,
                   edgecolors=SURFACE, lw=0.5)
ax.set_xticks(np.arange(len(CORE)) + 0.5)
ax.set_xticklabels([DC[c] for c in CORE], rotation=45, ha='right', fontsize=8)
ax.set_yticks(np.arange(len(P)) + 0.5)
ax.set_yticklabels(P.index, fontsize=8.5)
ax.set_ylim(len(P), 0)
ax.set_title(f'Normalised $a_{{i,k}}$ at $c={c_ref:.0f}$',
             fontsize=10.5, color=INK_2, pad=6)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
cb.outline.set_visible(False)
cb.ax.tick_params(length=2.5, labelsize=8)

ax = axes[2]
for i, f in enumerate(firms):
    s = sub[sub.firm == f]
    p = s.pivot_table(index='c', columns='cat', values='a')[CORE]
    pn = p.div(p.mean(axis=1), axis=0)
    ax.plot(p.index, pn.std(axis=1), lw=1.5, label=DO.get(f, f))
ax.set_xscale('log')
ax.set_xlabel('budget $c_i$')
ax.set_ylabel('sd of the normalised profile')
ax.set_title('Profile dispersion vs budget', fontsize=10.5, color=INK_2, pad=6)
ax.legend(frameon=False, fontsize=7.5, ncol=2)

for ax in (axes[0], axes[2]):
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
fig.tight_layout(pad=0.7, w_pad=1.8)
fig.savefig('production_identified_set.pdf')
plt.close(fig)
print('\nwrote production_identified_set.pdf')

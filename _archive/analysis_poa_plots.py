"""
Two PDFs:
1. plot_poa_scatter.pdf  — K=2 scatter: Nash vs Optimal allocations side by side
2. plot_poa_bars.pdf     — Bar chart: Nash SW vs Optimal SW per function
"""

import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from competition_simulation import CompetitionSimulation

# ── shared setup ──────────────────────────────────────────────────────────────
N = 8
B = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
w2 = np.array([2/3, 1/3])  # K=2 for scatter

w6 = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
w6 = w6 / w6.sum()         # K=6 for bar chart

FUNCTIONS = [
    ("log(x+1)",       lambda x: math.log(max(x,0)+1),       lambda x: 1/(max(x,0)+1)),
    ("log(x+10)",      lambda x: math.log(max(x,0)+10),      lambda x: 1/(max(x,0)+10)),
    ("sqrt(x)",        lambda x: math.sqrt(max(x,0)),         lambda x: 0.5/math.sqrt(max(x,1e-12))),
    ("x^0.99",         lambda x: max(x,1e-12)**0.99,          lambda x: 0.99*max(x,1e-12)**(-0.01)),
    ("1-e^{-10x}",     lambda x: 1-math.exp(-10*max(x,0)),   lambda x: 10*math.exp(-10*max(x,0))),
    ("1-(1+x)^{-2}",   lambda x: 1-(1+max(x,0))**(-2),       lambda x: 2*(1+max(x,0))**(-3)),
]

colors = plt.cm.plasma(np.linspace(0.1, 0.85, N))

# ── helpers ───────────────────────────────────────────────────────────────────
def social_welfare(X, w, f_fn):
    return sum(w[k] * max(f_fn(X[i, k]) for i in range(len(X)))
               for k in range(len(w)))

def optimal_allocation_given_assignment(assignment, B, w, f_fn):
    N, K = len(B), len(w)
    X = np.zeros((N, K))
    for i in range(N):
        dims = [k for k in range(K) if assignment[k] == i]
        if not dims:
            continue
        wi = np.array([w[k] for k in dims])
        bi = B[i]
        if len(dims) == 1:
            X[i, dims[0]] = bi
            continue
        def neg_obj(x):
            return -sum(wi[j] * f_fn(x[j]) for j in range(len(dims)))
        x0 = wi / wi.sum() * bi
        cons = {'type': 'eq', 'fun': lambda x: x.sum() - bi}
        bounds = [(0, None)] * len(dims)
        res = minimize(neg_obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-10, 'maxiter': 500})
        for j, k in enumerate(dims):
            X[i, k] = max(res.x[j], 0)
    return X

def find_optimal(B, w, f_fn):
    N, K = len(B), len(w)
    best_val = -np.inf
    best_X = None
    for assignment in itertools.product(range(N), repeat=K):
        X = optimal_allocation_given_assignment(assignment, B, w, f_fn)
        val = social_welfare(X, w, f_fn)
        if val > best_val:
            best_val = val
            best_X = X.copy()
    return best_X, best_val

# ═══════════════════════════════════════════════════════════════════════════════
# PDF 1: K=2 scatter — Nash vs Optimal
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing K=2 Nash + Optimal allocations...")
nf = len(FUNCTIONS)
ncols = 3
nrows = math.ceil(nf / ncols)
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(14, 4.5*nrows))
axes1 = axes1.flatten()
fig1.suptitle(r"Nash (circles) vs Optimal (stars): $K=2$, $w_1=2/3$, $w_2=1/3$, $N=8$ firms"
              "\n(dashed = proportional ray; faint diagonals = budget constraints)",
              fontsize=12, fontweight="bold")

for idx, (name, f_fn, fp_fn) in enumerate(FUNCTIONS):
    sim = CompetitionSimulation(N, 2, B, w2, w2.copy(), f_fn, fp_fn)
    X_nash, _ = sim.find_nash_br(max_iter=300, tol=1e-7, verbose=False)
    sw_nash = social_welfare(X_nash, w2, f_fn)
    X_opt, sw_opt = find_optimal(B, w2, f_fn)

    ax = axes1[idx]
    t = np.linspace(0, B.max(), 200)
    ax.plot(2/3*t, 1/3*t, 'k--', lw=1, alpha=0.4)
    for bi in B:
        ax.plot([bi, 0], [0, bi], '-', color='gray', lw=0.5, alpha=0.25)

    for i in range(N):
        # Nash: filled circle
        ax.scatter(X_nash[i,0], X_nash[i,1], color=colors[i], s=80,
                   marker='o', zorder=5)
        # Optimal: star, same color, larger
        ax.scatter(X_opt[i,0], X_opt[i,1], color=colors[i], s=220,
                   marker='*', zorder=6, edgecolors='k', linewidths=0.4)
        ax.annotate(f'B={B[i]:.0f}', (X_nash[i,0], X_nash[i,1]),
                    textcoords='offset points', xytext=(4,3), fontsize=7, color=colors[i])

    ax.set_xlabel('$x_{i,1}$  ($w_1=2/3$)', fontsize=9)
    ax.set_ylabel('$x_{i,2}$  ($w_2=1/3$)', fontsize=9)
    ax.set_title(f'$f(x)={name}$\nNash SW={sw_nash:.3f}  |  Opt SW={sw_opt:.3f}  '
                 f'(PoA={sw_nash/sw_opt:.2f})', fontsize=9, fontweight='bold')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

# legend in first panel
axes1[0].scatter([], [], color='gray', s=80,  marker='o', label='Nash')
axes1[0].scatter([], [], color='gray', s=220, marker='*', edgecolors='k',
                 linewidths=0.4, label='Optimal')
axes1[0].legend(fontsize=9, loc='upper left')

for idx in range(nf, len(axes1)):
    axes1[idx].set_visible(False)

plt.tight_layout()
plt.savefig("plot_poa_scatter.pdf", format="pdf")
plt.close()
print("Saved plot_poa_scatter.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# PDF 2: Bar chart — Nash SW vs Optimal SW (K=6)
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing K=6 Nash + Optimal social welfare...")
# Use smaller N for K=6 enumeration (N^K = 5^6 = 15625)
N6 = 5
B6 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

nash_sws, opt_sws, names = [], [], []
for name, f_fn, fp_fn in FUNCTIONS:
    sim = CompetitionSimulation(N6, 6, B6, w6, w6.copy(), f_fn, fp_fn)
    X_nash, _ = sim.find_nash_br(max_iter=300, tol=1e-7, verbose=False)
    nash_sw = social_welfare(X_nash, w6, f_fn)

    _, opt_sw = find_optimal(B6, w6, f_fn)

    nash_sws.append(nash_sw)
    opt_sws.append(opt_sw)
    names.append(name)
    print(f"  {name:<18}  Nash={nash_sw:.4f}  Opt={opt_sw:.4f}  PoA={nash_sw/opt_sw:.3f}")

fig2, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(names))
w_bar = 0.35
bars1 = ax.bar(x - w_bar/2, nash_sws, w_bar, label='Nash SW',    color='steelblue',  alpha=0.85)
bars2 = ax.bar(x + w_bar/2, opt_sws,  w_bar, label='Optimal SW', color='firebrick',  alpha=0.85)

for bar, val in zip(bars1, nash_sws):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
for bar, val, nash in zip(bars2, opt_sws, nash_sws):
    poa = nash/val
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f'{val:.3f}\n(PoA={poa:.2f})', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f'$f(x)={n}$' for n in names], fontsize=9)
ax.set_ylabel('Social Welfare', fontsize=11)
ax.set_title(f'Social Welfare: Nash vs Optimal  ($N={N6}$, $K=6$)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("plot_poa_bars.pdf", format="pdf")
plt.close()
print("Saved plot_poa_bars.pdf")

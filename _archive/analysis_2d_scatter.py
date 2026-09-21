"""
K=2 dims, w=[2/3, 1/3]. Plot Nash allocation as 2D scatter for each f(x).
Each point = one firm at (x_{i,1}, x_{i,2}). Budget constraint is the line x1+x2=B_i.
Proportional allocation = (2/3 B_i, 1/3 B_i), i.e., slope 1/2 from origin.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from competition_simulation import CompetitionSimulation

N = 8
B = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
w = np.array([2/3, 1/3])

functions = [
    ("log(x+1)",      lambda x: math.log(max(x,0)+1),         lambda x: 1/(max(x,0)+1)),
    ("log(x+10)",     lambda x: math.log(max(x,0)+10),        lambda x: 1/(max(x,0)+10)),
    ("sqrt(x)",       lambda x: math.sqrt(max(x,0)),           lambda x: 0.5/math.sqrt(max(x,1e-12))),
    ("x^0.99",        lambda x: max(x,1e-12)**0.99,            lambda x: 0.99*max(x,1e-12)**(-0.01)),
    ("1-e^{-10x}",    lambda x: 1-math.exp(-10*max(x,0)),      lambda x: 10*math.exp(-10*max(x,0))),
    ("1-(1+x)^{-2}",  lambda x: 1-(1+max(x,0))**(-2),         lambda x: 2*(1+max(x,0))**(-3)),
]

colors = plt.cm.plasma(np.linspace(0.1, 0.85, N))

ncols = 3
nrows = math.ceil(len(functions) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5*nrows))
axes = axes.flatten()

for ax_idx, (name, f_fn, fp_fn) in enumerate(functions):
    ax = axes[ax_idx]
    sim = CompetitionSimulation(N, 2, B, w, w.copy(), f_fn, fp_fn)
    X, _ = sim.find_nash_br(max_iter=300, tol=1e-7, verbose=False)

    # Draw budget lines and proportional ray
    B_max = B.max()
    t = np.linspace(0, B_max, 200)
    ax.plot(2/3 * t, 1/3 * t, 'k--', lw=1, alpha=0.4, label='Proportional (2:1)')

    # Budget constraint lines (faint)
    for bi in B:
        ax.plot([bi, 0], [0, bi], '-', color='gray', lw=0.5, alpha=0.3)

    # Plot each firm
    for i in range(N):
        ax.scatter(X[i,0], X[i,1], color=colors[i], s=90, zorder=5,
                   label=f'B={B[i]:.0f}')
        ax.annotate(f'B={B[i]:.0f}', (X[i,0], X[i,1]),
                    textcoords='offset points', xytext=(5,4), fontsize=7, color=colors[i])

    ax.set_xlabel('$x_{i,1}$  ($w_1=2/3$)', fontsize=10)
    ax.set_ylabel('$x_{i,2}$  ($w_2=1/3$)', fontsize=10)
    ax.set_title(f'$f(x)={name}$', fontsize=10, fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if ax_idx == 0:
        ax.legend(fontsize=7, loc='upper left')

# Hide unused axes
for ax_idx in range(len(functions), len(axes)):
    axes[ax_idx].set_visible(False)

fig.suptitle(r'Nash allocations in 2D: $w_1=2/3,\ w_2=1/3$, $N=8$ firms'
             '\n(dashed = proportional ray; faint lines = budget constraints)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_2d_scatter.pdf', format='pdf')
plt.close()
print('Saved plot_2d_scatter.pdf')

"""RQ3: How does evaluation design affect firm behavior?

Deliberately small: ONE figure (control) + ONE table (identification), on a
calibrated semi-synthetic environment (real theta_i^0, real B_0/beta0/resid0,
bootstrap-resampled real release residuals -- same construction as expT/expU).

FIGURE (evaluator control): an interpretable one-dimensional perturbation
    w(lambda) = (1-lambda) w^A + lambda w^B,   lambda in [0,1]
shifting evaluator weight from Coding (w^A) to Creative Writing (w^B), held
FIXED for T periods (steady state) under each of 3 plausible F_i families from
RQ2. Plots lambda vs the REAL (not residualized) equilibrium monoculture
    H(theta*(w(lambda))) = mean pairwise cosine of firms' raw theta vectors.
Real H, not the two-way-residual H used for the empirical (noisy, common-
quality-dominated) leaderboard data -- here theta is fully observed and
noise-free apart from calibrated release noise, so the raw cosine is exactly
the theoretical object Theorem 9 / the control result are about.

TABLE 5 (identification): does varying w induce different enough investment
responses to learn F_i? Same T (number of evaluations) across designs; only
the RANK of w's variation differs. Response rank r_i is the empirical
(participation-ratio, time-centered) effective rank of each firm's realized
regressor matrix -- see Proposition (Evaluation Identification): unidentified
dimensions = K(K - r_i). Range of H induced ties the two results together,
computed on the fully-heterogeneous F_i (the family for which identification
is actually a live question).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'experiments')
from common import load_panel, build_events, firm_profiles, CORE, FRONTIER

INK, INK_2, MUTED, GRID, BASELINE, SURFACE = \
    '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 16, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 18,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.linewidth': 1.0, 'pdf.fonttype': 42,
})


def strip_axes(ax, keep=('left', 'bottom')):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=4, width=0.7)


K, N = len(CORE), len(FRONTIER)

# ---- real calibration (identical construction to expT/expU) ----
d = load_panel(cats=CORE)
ev = build_events(d, CORE)
G_ev = ev.pivot_table(index='event', columns='cat', values='gap')[CORE].values
D_ev = ev.pivot_table(index='event', columns='cat', values='dtheta')[CORE].values
lam_ridge = 1000.0
B0 = np.linalg.solve(G_ev.T @ G_ev + lam_ridge * np.eye(K), G_ev.T @ D_ev).T
beta0 = np.diag(B0).mean()
resid0 = B0 - beta0 * np.eye(K)
U, S, Vt = np.linalg.svd(resid0)
u1, s1, v1 = U[:, 0], S[0], Vt[0, :]
resid_pool = D_ev - G_ev @ B0.T

firm_of = ev.drop_duplicates('event').set_index('event').firm
firm_of = firm_of.reindex(ev.pivot_table(index='event', columns='cat', values='gap').index)
X1 = beta0 * G_ev
proj = G_ev @ v1
X2 = s1 * proj[:, None] * u1[None, :]
firms = sorted(firm_of.unique())
a_real = {}
for f in firms:
    idx = np.where(firm_of.values == f)[0]
    x1, x2, y = X1[idx].reshape(-1), X2[idx].reshape(-1), D_ev[idx].reshape(-1)
    num = x1 @ (y - x2) + 1000 * 1.0
    den = x1 @ x1 + 1000
    a_real[f] = num / den

dts = sorted(d.dt.unique())
theta0 = None
for t in dts[::-1]:
    P, V, names = firm_profiles(d, CORE, t)
    if len(names) == N:
        theta0, order = P, names
        break
a_vec = np.array([a_real.get(f, 1.0) for f in order])


def gt_homogeneous():
    return {f: B0.copy() for f in order}


def gt_scale_a_i():
    return {f: a_vec[j] * beta0 * np.eye(K) + resid0 for j, f in enumerate(order)}


def gt_fully_het(rng, scale=0.3):
    sigma = scale * np.linalg.norm(B0) / K
    return {f: B0 + rng.standard_normal((K, K)) * sigma for f in order}


FAMILIES = {
    'homogeneous ($F_i$ all equal)': lambda rng: gt_homogeneous(),
    'parallel, scaled ($a_i$ only)': lambda rng: gt_scale_a_i(),
    'fully heterogeneous': lambda rng: gt_fully_het(rng),
}
FAMILY_COLOR = {'homogeneous ($F_i$ all equal)': MUTED,
                'parallel, scaled ($a_i$ only)': BLUE,
                'fully heterogeneous': ORANGE}


def simulate(B_star, w_path, rng, T):
    theta = theta0.copy()
    rows_G, rows_f = [], []
    for t in range(T):
        wt = w_path[t] if w_path.ndim == 2 else w_path
        new = theta.copy()
        for j, f in enumerate(order):
            rivals = [k for k in range(N) if k != j]
            gap = theta[rivals].max(axis=0) - theta[j]
            g = K * wt * gap
            eps = resid_pool[rng.integers(len(resid_pool))]
            new[j] = theta[j] + B_star[f] @ g + eps
            rows_G.append(g); rows_f.append(f)
        theta = new
    return theta, np.array(rows_G), np.array(rows_f)


def H_raw(theta_mat):
    """Real monoculture H: mean pairwise cosine of firms' RAW theta vectors."""
    cs = []
    for a in range(len(theta_mat)):
        for b in range(a + 1, len(theta_mat)):
            u, v = theta_mat[a], theta_mat[b]
            nn = np.linalg.norm(u) * np.linalg.norm(v)
            if nn > 0:
                cs.append(u @ v / nn)
    return np.mean(cs)


# ============================================================= FIGURE ======
CAT_A, CAT_B = 'coding', 'creative_writing'
iA, iB = CORE.index(CAT_A), CORE.index(CAT_B)


def make_w(cat_idx, weight=0.7):
    w = np.full(K, (1 - weight) / (K - 1))
    w[cat_idx] = weight
    return w


wA, wB = make_w(iA), make_w(iB)
lambdas = np.linspace(0, 1, 16)
T, reps = 30, 25

fig, ax = plt.subplots(figsize=(7.4, 5.6))
for fam_name, fam_fn in FAMILIES.items():
    color = FAMILY_COLOR[fam_name]
    med, lo, hi = [], [], []
    for lam in lambdas:
        w = (1 - lam) * wA + lam * wB
        w = w / w.sum()
        hs = []
        for rep in range(reps):
            rng = np.random.default_rng(5000 * reps + rep)
            B_star = fam_fn(rng)
            theta_T, _, _ = simulate(B_star, w, rng, T)
            hs.append(H_raw(theta_T))
        med.append(np.median(hs)); lo.append(np.percentile(hs, 10)); hi.append(np.percentile(hs, 90))
    med, lo, hi = np.array(med), np.array(lo), np.array(hi)
    ax.plot(lambdas, med, color=color, lw=3.0, zorder=3, label=fam_name)
    ax.fill_between(lambdas, lo, hi, color=color, alpha=0.15, zorder=1, lw=0)

ax.set_xlabel(r'$\lambda$   (evaluator weight: coding $\to$ creative writing)')
ax.set_ylabel(r'$H(\theta^*(\mathbf{w}(\lambda)))$')
ax.grid(color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
ax.legend(frameon=False, fontsize=13.5, loc='lower center', handlelength=1.6)
fig.tight_layout(pad=0.8)
fig.savefig('rq3_evaluator_control.pdf')
plt.close(fig)
print('wrote rq3_evaluator_control.pdf')

# ============================================================= TABLE 5 =====
# Response rank r_i: effective (participation-ratio) rank of firm i's realized
# investment-signal sequence g_t = K w_t (.) gap_t across the T evaluations.
# Two fixes relative to a naive SVD, both load-bearing:
#  (1) row-normalize each g_t before the SVD. Under heterogeneous technology
#      theta can grow (or, for unstable B_i draws, blow up) over T periods;
#      raw magnitude then swamps the decomposition with one "how big did this
#      get" direction regardless of w's design. Cosine-normalizing isolates
#      DIRECTIONAL variation, which is what identifies B_i's off-diagonal
#      structure -- and is exactly what makes H itself scale-invariant too.
#  (2) i.i.d. draws for w's perturbation coefficients, not a random walk. A
#      random walk's rank estimate converges to its target very slowly (checked
#      directly: even T=3000 undershoots); i.i.d. draws in a designed r-dim
#      subspace converge fast, so T can stay modest and match the figure.
rng_dir = np.random.default_rng(42)
raw = rng_dir.standard_normal((K, K))
raw -= raw.mean(0, keepdims=True)
Qfull, _ = np.linalg.qr(raw[:, :K - 1])
PERTURB_AMP = 0.08


def make_w_path(T, r, rng):
    basis = Qfull[:, :r]
    coefs = rng.uniform(-PERTURB_AMP, PERTURB_AMP, size=(T, r))
    w = 1.0 / K + coefs @ basis.T
    w = np.clip(w, 1e-3, None)
    return w / w.sum(axis=1, keepdims=True)


def eff_rank(Z):
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.where(norm == 0, 1, norm)
    Zc = Zn - Zn.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    denom = (s ** 2 ** 2).sum()
    return (s ** 2).sum() ** 2 / denom if denom > 0 else np.nan


def simulate_H_trajectory(B_star, w_path, rng, T):
    """Like simulate(), but records H at every step -- the range of monoculture
    states VISITED under a design, not just where it ends up."""
    theta = theta0.copy()
    hs = [H_raw(theta)]
    rows_G, rows_f = [], []
    for t in range(T):
        wt = w_path[t] if w_path.ndim == 2 else w_path
        new = theta.copy()
        for j, f in enumerate(order):
            rivals = [k for k in range(N) if k != j]
            gap = theta[rivals].max(axis=0) - theta[j]
            g = K * wt * gap
            eps = resid_pool[rng.integers(len(resid_pool))]
            new[j] = theta[j] + B_star[f] @ g + eps
            rows_G.append(g); rows_f.append(f)
        theta = new
        hs.append(H_raw(theta))
    return hs, np.array(rows_G), np.array(rows_f)


T_table = 30
reps_table = 30
designs = [('Small perturbations', 1), ('Moderate variation', 2), ('Diverse variation', K - 1)]

# one shared representative technology draw for the H-range column, so the
# range reflects each DESIGN's control power, not which random B_star it hit
B_STAR_REF = gt_fully_het(np.random.default_rng(42))

table_rows = []
# row 0: single (static) evaluation -- analytic rank (one w -> rank <=1), H is
# the range of monoculture states visited under that one fixed evaluation
w_static = np.tile(np.ones(K) / K, (T_table, 1))
hs0 = []
for rep in range(reps_table):
    rng = np.random.default_rng(9000 * reps_table + rep)
    h_traj, _, _ = simulate_H_trajectory(B_STAR_REF, w_static, rng, T_table)
    hs0.append(h_traj[-1])  # where the design lands, not the shared t=0 start
table_rows.append(dict(design='Single evaluation', n_w=1, r_i=1.0,
                        unident=K * (K - 1), h_lo=np.min(hs0), h_hi=np.max(hs0)))

for name, r in designs:
    ranks, hs = [], []
    for rep in range(reps_table):
        rng = np.random.default_rng(7000 * reps_table + rep)
        w_path = make_w_path(T_table, r, rng)
        h_traj, Gs, fs = simulate_H_trajectory(B_STAR_REF, w_path, rng, T_table)
        hs.append(h_traj[-1])
        firm_ranks = [eff_rank(Gs[fs == f]) for f in order]
        ranks.append(np.mean(firm_ranks))
    r_i = np.mean(ranks)
    table_rows.append(dict(design=name, n_w=T_table, r_i=r_i,
                            unident=K * (K - r_i), h_lo=np.min(hs), h_hi=np.max(hs)))

tab = pd.DataFrame(table_rows)
print('\nTable 5 (identification):')
print(tab.round(3).to_string(index=False))

lines = [r'\begin{tabular}{lccccc}', r'\toprule',
         r'Evaluation design & \# distinct $\mathbf{w}$\ & Response rank $\hat r_i$ & '
         r'Unidentified dims. $K(K-\hat r_i)$ & Range of $H$ induced & $F_i$ identified? \\',
         r'\midrule']
for _, row in tab.iterrows():
    ident = 'No' if row.r_i < K - 0.5 else 'Yes'
    lines.append(f'{row.design} & {row.n_w} & {row.r_i:.2f} & {row.unident:.1f} & '
                 f'[{row.h_lo:.2f}, {row.h_hi:.2f}] & {ident} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
with open('experiments/table5_identification.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\nwrote experiments/table5_identification.tex')

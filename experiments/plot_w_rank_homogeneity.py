"""Does the RANK of evaluator-weight (w) variation change output monoculture over
time -- not just identification of A_i (that's expU), but the actual trajectory of
H = mean pairwise cosine of firms' two-way-residual profiles, in [-1,1].
H=1: perfect monoculture (relative profiles fully aligned). H=-1/(N-1): the
mechanical permutation-null floor with zero shared structure (not H=-1).

Same calibration and simulator as expU_w_rank_identification.py (real theta_i^0,
real B_0/beta0/resid0, bootstrap-resampled real release residuals; rank-r w-shock
process on the simplex tangent space, total shock variance held fixed across r).

Outputs:
  w_rank_H_trajectory.pdf   H_t over time, one ribbon per rank r, for the
                             homogeneous vs fully-heterogeneous ground truth
  table_w_rank_H.tex        rank r x {avg H, max H, min H, empirical identification
                             rank r_i} for both ground-truth families
  table_w_specific_H.tex    a handful of NAMED, economically legible w vectors held
                             fixed for T periods -> steady-state-ish H
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'experiments')
from common import load_panel, build_events, firm_profiles, two_way_resid, CORE, FRONTIER

INK, INK_2, MUTED, GRID, BASELINE, SURFACE = \
    '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
RCOLOR = {1: '#2a78d6', 2: '#eb6834', 4: '#1baf7a', 7: '#4a3aa7', 'empirical': '#e34948'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 9.5,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})


def strip_axes(ax, keep=('left', 'bottom')):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=3, width=0.6)


K, N = len(CORE), len(FRONTIER)

# ---- real calibration (identical to expU) ----
d = load_panel(cats=CORE)
ev = build_events(d, CORE)
G_ev = ev.pivot_table(index='event', columns='cat', values='gap')[CORE].values
D_ev = ev.pivot_table(index='event', columns='cat', values='dtheta')[CORE].values
lam = 1000.0
B0 = np.linalg.solve(G_ev.T @ G_ev + lam * np.eye(K), G_ev.T @ D_ev).T
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

rng_dir = np.random.default_rng(42)
raw = rng_dir.standard_normal((K, K))
raw -= raw.mean(0, keepdims=True)
Qfull, _ = np.linalg.qr(raw[:, :K - 1])
TOTAL_VAR = 3e-4


def make_w_path(T, r, rng):
    basis = Qfull if r == K - 1 else Qfull[:, :r]
    per_dir_var = TOTAL_VAR / r
    coefs = rng.standard_normal((T, r)) * np.sqrt(per_dir_var)
    dw = coefs @ basis.T
    w = np.zeros((T + 1, K))
    w[0] = 1.0 / K
    for t in range(T):
        w[t + 1] = np.clip(w[t] + dw[t], 1e-3, None)
        w[t + 1] /= w[t + 1].sum()
    return w[1:]


w_share = pd.read_parquet('experiments/_w_share_panel.parquet')[CORE]
dw_emp_raw = w_share.diff().dropna().values
_emp_trace = np.trace(np.cov(dw_emp_raw.T))
dw_emp = dw_emp_raw * np.sqrt(TOTAL_VAR / _emp_trace)


def make_w_path_empirical(T, rng):
    w = np.zeros((T + 1, K))
    w[0] = 1.0 / K
    idx = rng.integers(len(dw_emp), size=T)
    for t in range(T):
        w[t + 1] = np.clip(w[t] + dw_emp[idx[t]], 1e-3, None)
        w[t + 1] /= w[t + 1].sum()
    return w[1:]


def gt_family_i():
    return {f: B0.copy() for f in order}


def gt_family_ii():
    return {f: a_vec[j] * beta0 * np.eye(K) + resid0 for j, f in enumerate(order)}


def gt_family_iv(rng, scale=0.3):
    sigma = scale * np.linalg.norm(B0) / K
    return {f: B0 + rng.standard_normal((K, K)) * sigma for f in order}


def simulate_panel_full(B_star, w_path, rng, T):
    """Like expU's simulate_panel, but also returns the theta trajectory."""
    theta = theta0.copy()
    theta_hist = [theta.copy()]
    rows_G, rows_f = [], []
    for t in range(T):
        wt = w_path[t]
        new = theta.copy()
        for j, f in enumerate(order):
            rivals = [k for k in range(N) if k != j]
            gap = theta[rivals].max(axis=0) - theta[j]
            g = K * wt * gap
            eps = resid_pool[rng.integers(len(resid_pool))]
            dtheta = B_star[f] @ g + eps
            new[j] = theta[j] + dtheta
            rows_G.append(g); rows_f.append(f)
        theta = new
        theta_hist.append(theta.copy())
    return np.array(theta_hist), np.array(rows_G), np.array(rows_f)


def H_of(theta_mat):
    """H = mean pairwise cosine of the two-way-residual profile, in [-1,1].
    H=1: firms' relative (firm- and category-demeaned) profiles perfectly aligned
    -- homogenization/monoculture. H=-1/(N-1): the mechanical permutation-null
    floor for N firms with NO shared structure at all (not H=-1 -- two-way
    demeaning itself induces that floor; see exp0b_estimand_null.py)."""
    R = two_way_resid(theta_mat)
    cs = []
    for a in range(len(R)):
        for b in range(a + 1, len(R)):
            u, v = R[a], R[b]
            nn = np.linalg.norm(u) * np.linalg.norm(v)
            if nn > 0:
                cs.append(u @ v / nn)
    return np.mean(cs)


def eff_rank(Z):
    """Participation-ratio effective rank of the TIME-VARIATION in Z (the T x K
    matrix of realized g_t's for one firm) -- the empirical analogue of
    r_i = rank(Z_i) in the identification proposition. Z is centered (column
    means removed) before the SVD: a constant-across-t component of the
    regressor carries zero identifying information (it's collinear with the
    firm's own intercept/fixed effect, exactly as in fe_regress's `within`
    transform) and, left uncentered, swamps the rank estimate -- g_t's mean
    level (driven by w's mean ~1/K times the average gap) is large relative to
    the shocks, so an uncentered SVD reports ~rank 1 regardless of the target r."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    denom = (s ** 2 ** 2).sum()
    return (s ** 2).sum() ** 2 / denom if denom > 0 else np.nan


T = 30
reps = 40
r_conditions = [1, 2, 4, 7, 'empirical']
families = {'homogeneous': gt_family_i, 'real_scale_a_i': gt_family_ii,
            'fully_heterogeneous': gt_family_iv}

traj = {}     # (family, r) -> array (reps, T+1) of H_t
rank_emp = {}  # (family, r) -> array (reps,) of mean-over-firms effective rank

for fam_name, fam_fn in families.items():
    for r in r_conditions:
        Hs, ranks = [], []
        for rep in range(reps):
            rng = np.random.default_rng(2000 * reps + rep)
            B_star = gt_family_iv(rng) if fam_name == 'fully_heterogeneous' else fam_fn()
            w_path = make_w_path_empirical(T, rng) if r == 'empirical' else make_w_path(T, r, rng)
            theta_hist, Gs, fs = simulate_panel_full(B_star, w_path, rng, T)
            Hs.append([H_of(theta_hist[t]) for t in range(T + 1)])
            firm_ranks = [eff_rank(Gs[fs == f]) for f in order]
            ranks.append(np.mean(firm_ranks))
        traj[(fam_name, r)] = np.array(Hs)
        rank_emp[(fam_name, r)] = np.array(ranks)
        print(f'{fam_name:22s} r={str(r):>10}  mean H (final)={np.mean(np.array(Hs)[:, -1]):.4f}  '
              f'mean r_i={np.mean(ranks):.2f}')

# ============================================================= FIGURE ======
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
for ax, fam_name in zip(axes, families):
    for r in r_conditions:
        Hs = traj[(fam_name, r)]
        med = np.median(Hs, axis=0)
        lo, hi = np.percentile(Hs, [10, 90], axis=0)
        color = RCOLOR[r]
        label = f'r={r}' if r != 'empirical' else 'empirical Δw'
        ax.plot(range(T + 1), med, color=color, lw=1.8, zorder=3, label=label)
        ax.fill_between(range(T + 1), lo, hi, color=color, alpha=0.13, lw=0, zorder=1)
    ax.set_title(fam_name.replace('_', ' '), fontsize=9.5, loc='left', color=INK)
    ax.set_xlabel('release step $t$')
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_axes(ax)
axes[0].set_ylabel(r'$H_t = \overline{\cos}$(two-way residual profiles)')
axes[-1].legend(frameon=False, fontsize=7.6, loc='upper left', handlelength=1.4)
fig.tight_layout(pad=0.7)
fig.savefig('w_rank_H_trajectory.pdf')
plt.close(fig)

# --------------------------------------------------------- TABLE 1 (LaTeX) --
rows = []
for fam_name in families:
    for r in r_conditions:
        Hs = traj[(fam_name, r)]
        rows.append(dict(family=fam_name, r=r, avg_H=Hs.mean(), max_H=Hs.max(),
                          min_H=Hs.min(), r_i=rank_emp[(fam_name, r)].mean()))
tab1 = pd.DataFrame(rows)
print('\n' + tab1.to_string(index=False))

lines = [r'\begin{tabular}{llcccc}', r'\toprule',
         r'Ground truth & Target rank $r$ & avg.\ $H$ & max $H$ & min $H$ & '
         r'identification rank $\hat r_i$ \\', r'\midrule']
for _, row in tab1.iterrows():
    lines.append(f'{row.family.replace("_"," ")} & {row.r} & {row.avg_H:.3f} & '
                 f'{row.max_H:.3f} & {row.min_H:.3f} & {row.r_i:.2f} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
with open('experiments/table_w_rank_H.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')

# --------------------------------------------------------- TABLE 2 (LaTeX) --
NAMED_W = {
    'Uniform': np.ones(K) / K,
}
for k in CORE:
    w = np.full(K, 0.3 / (K - 1))
    w[CORE.index(k)] = 0.7
    NAMED_W[f'{k}-heavy (0.7)'] = w
NAMED_W['Coding+Math split'] = np.array(
    [0.4 if k in ('coding', 'math') else 0.2 / (K - 2) for k in CORE])

rng2 = np.random.default_rng(7)
rows2 = []
reps2 = 20
for name, w in NAMED_W.items():
    w = w / w.sum()
    w_path = np.tile(w, (T, 1))
    Hs_final = []
    for rep in range(reps2):
        rng = np.random.default_rng(3000 * reps2 + rep)
        B_star = gt_family_iv(rng)
        theta_hist, _, _ = simulate_panel_full(B_star, w_path, rng, T)
        Hs_final.append(H_of(theta_hist[-1]))
    rows2.append(dict(w_name=name, avg_H=np.mean(Hs_final), sd_H=np.std(Hs_final)))
tab2 = pd.DataFrame(rows2)
print('\n' + tab2.to_string(index=False))

lines2 = [r'\begin{tabular}{lcc}', r'\toprule',
          r'Evaluation weight $\bw$ (held fixed) & avg.\ $H$ (t=' + str(T) + r') & sd \\',
          r'\midrule']
for _, row in tab2.iterrows():
    lines2.append(f'{row.w_name} & {row.avg_H:.3f} & {row.sd_H:.3f} \\\\')
lines2 += [r'\bottomrule', r'\end{tabular}']
with open('experiments/table_w_specific_H.tex', 'w') as f:
    f.write('\n'.join(lines2) + '\n')

print('\nwrote w_rank_H_trajectory.pdf, table_w_rank_H.tex, table_w_specific_H.tex')

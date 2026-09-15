"""RQ3. Does more dynamic data buy a better production function?

expR shows that a firm-specific scale a_i on a shared response matrix beats the
pooled model out of sample.  The open question is whether that advantage grows
with the number of release events observed, i.e. whether the panel is
information-limited or the model is simply near its ceiling.

Design.  For a training size n, draw n events at random, fit the three families
on them, and score leave-one-out-style on the held-out events.  Repeat over many
draws and average.  Families:

  Model 0  dtheta = B_ridge @ gap                     shared response, 0 params/firm
  Model 1  dtheta = a_i * (B_ridge @ gap)             firm scale,      1 param/firm
  Model 2  dtheta = a_i * beta0 * gap + b_i * rank-1  firm scale+shape, 2 params/firm

a_i and b_i are ridge-shrunk toward the pooled value 1, with the ridge chosen on
the training split only.  Firms with no training event fall back to a_i = 1.

Writes production_learning_curve.pdf.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'experiments')
from common import load_panel, build_events, CORE, FRONTIER

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 10.5,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})

K = len(CORE)
d = load_panel(cats=CORE)
ev = build_events(d, CORE)
G = ev.pivot_table(index='event', columns='cat', values='gap')[CORE]
Y = ev.pivot_table(index='event', columns='cat', values='dtheta')[CORE]
firm = ev.drop_duplicates('event').set_index('event').firm
events = [e for e in G.index if G.loc[e].notna().all() and Y.loc[e].notna().all()]
G, Y, firm = G.loc[events].values, Y.loc[events].values, firm.loc[events].values
E = len(events)
print(f'{E} events, {firm_n} firms' if False else f'{E} events, {len(set(firm))} firms')


def fit_B(idx, lam=1.0):
    """Ridge response matrix on the training events."""
    X, Yt = G[idx], Y[idx]
    return np.linalg.solve(X.T @ X + lam * np.eye(K), X.T @ Yt)


def fit_scale(idx, B, lam):
    """a_i shrunk toward 1: a_i = (sum <p,y> + lam) / (sum <p,p> + lam)."""
    P = G[idx] @ B
    num, den = {}, {}
    for r, f in zip(range(len(idx)), firm[idx]):
        num[f] = num.get(f, 0.0) + float(P[r] @ Y[idx][r])
        den[f] = den.get(f, 0.0) + float(P[r] @ P[r])
    return {f: (num[f] + lam) / (den[f] + lam) for f in num}


def fit_shape(idx, lam):
    """a_i on the isotropic part, b_i on the shared rank-1 correction."""
    B = fit_B(idx)
    beta0 = np.trace(B) / K
    R = B - beta0 * np.eye(K)
    u, s, vt = np.linalg.svd(R)
    p1 = G[idx] @ (s[0] * np.outer(u[:, 0], vt[0]))
    p0 = beta0 * G[idx]
    ab = {}
    for f in set(firm[idx]):
        m = firm[idx] == f
        A = np.column_stack([p0[m].ravel(), p1[m].ravel()])
        t = Y[idx][m].ravel()
        M = A.T @ A + lam * np.eye(2)
        ab[f] = np.linalg.solve(M, A.T @ t + lam * np.ones(2))
    return beta0, s[0] * np.outer(u[:, 0], vt[0]), ab


def score(pred, idx):
    y = Y[idx]
    ss = ((y - pred) ** 2).sum()
    return 1 - ss / (y ** 2).sum()


LAMS = [1, 10, 100, 1000, 1e6]
SIZES = [12, 16, 20, 26, 32, 40, 48, 56, 60]
N_DRAW = 120
rng = np.random.default_rng(0)

rows = []
for n in SIZES:
    if n >= E - 4:
        continue
    acc = {0: [], 1: [], 2: []}
    for _ in range(N_DRAW):
        perm = rng.permutation(E)
        tr, te = perm[:n], perm[n:]
        B = fit_B(tr)
        acc[0].append(score(G[te] @ B, te))
        # pick the ridge by 4-fold CV inside the training split only
        best, best_lam = -np.inf, LAMS[-1]
        folds = np.array_split(rng.permutation(tr), 4)
        for lam in LAMS:
            preds, tgts = [], []
            for k in range(4):
                inner_te = folds[k]
                inner_tr = np.concatenate([folds[j] for j in range(4) if j != k])
                Bi = fit_B(inner_tr)
                a = fit_scale(inner_tr, Bi, lam)
                preds.append(np.array([a.get(f, 1.0) * (G[i] @ Bi)
                                       for i, f in zip(inner_te, firm[inner_te])]))
                tgts.append(Y[inner_te])
            P_, T_ = np.vstack(preds), np.vstack(tgts)
            s_ = 1 - ((T_ - P_) ** 2).sum() / (T_ ** 2).sum()
            if s_ > best:
                best, best_lam = s_, lam
        a = fit_scale(tr, B, best_lam)
        p1 = np.array([a.get(f, 1.0) * (G[i] @ B)
                       for i, f in zip(te, firm[te])])
        acc[1].append(score(p1, te))
        beta0, R1, ab = fit_shape(tr, best_lam)
        p2 = np.array([(ab.get(f, np.ones(2))[0] * beta0 * G[i]
                        + ab.get(f, np.ones(2))[1] * (G[i] @ R1)) for i, f in
                       zip(te, firm[te])])
        acc[2].append(score(p2, te))
    rows.append(dict(n=n,
                     m0=np.mean(acc[0]), m0_se=np.std(acc[0]) / np.sqrt(N_DRAW),
                     m1=np.mean(acc[1]), m1_se=np.std(acc[1]) / np.sqrt(N_DRAW),
                     m2=np.mean(acc[2]), m2_se=np.std(acc[2]) / np.sqrt(N_DRAW)))
L = pd.DataFrame(rows)
L['gain1'] = L.m1 - L.m0
L['gain2'] = L.m2 - L.m0
print('\nheld-out R^2 by training size (mean over 120 random splits)')
print(L.to_string(index=False, float_format=lambda x: f'{x:8.4f}'))

for c in ('m0', 'm1', 'gain1'):
    b = np.polyfit(np.log(L.n), L[c], 1)[0]
    print(f'  {c:<6} slope per e-fold of training events: {b:+.4f}')

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
ax = axes[0]
for c, se, col, lab in (('m0', 'm0_se', MUTED, 'shared response'),
                        ('m1', 'm1_se', BLUE, r'+ firm scale $a_i$'),
                        ('m2', 'm2_se', ORANGE, r'+ firm shape $b_i$')):
    ax.errorbar(L.n, L[c], yerr=1.96 * L[se], color=col, lw=1.6, marker='o',
                ms=3.5, capsize=2, label=lab)
ax.set_xlabel('training events')
ax.set_ylabel('held-out $R^2$')
ax.set_title('Prediction vs amount of history', fontsize=10.5, color=INK_2, pad=6)
ax.legend(frameon=False, fontsize=9, loc='lower right')

ax = axes[1]
ax.axhline(0, color=BASELINE, lw=0.9, ls=(0, (3, 2)))
ax.plot(L.n, L.gain1, color=BLUE, lw=1.7, marker='o', ms=3.5,
        label=r'$a_i$ over shared')
ax.plot(L.n, L.gain2, color=ORANGE, lw=1.4, ls=(0, (3, 2)), marker='s', ms=3,
        label=r'$a_i,b_i$ over shared')
ax.set_xlabel('training events')
ax.set_ylabel('gain in held-out $R^2$')
ax.set_title('Value of firm-specific parameters', fontsize=10.5, color=INK_2, pad=6)
ax.legend(frameon=False, fontsize=9)

for ax in axes:
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
    ax.tick_params(length=3, width=0.6)
fig.tight_layout(pad=0.7, w_pad=1.8)
fig.savefig('production_learning_curve.pdf')
plt.close(fig)
print('\nwrote production_learning_curve.pdf')

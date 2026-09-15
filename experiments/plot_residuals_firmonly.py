"""RQ1 static similarity with FIRM demeaning ONLY (no category demeaning).

R_i,k = theta_i,k - mean_k(theta_i,.)   [row-demean only]

Contrast with plot_residuals.py, which uses the two-way residual
R = P - rowmean - colmean.  Dropping the column demeaning leaves the common
across-firm category profile in R, so cosines are no longer pinned near the
-1/(N-1) permutation null.  Outputs:

  residual_cosine_kde_firmonly.pdf       pooled cos(R_i,R_i') vs permutation null
  residual_cosine_heatmap_firmonly.pdf   pairwise cosine, top-23 labs, one snapshot
  residual_category_contrib.pdf          which categories drive the similarity

and prints the category decomposition.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde

sys.path.insert(0, 'experiments')
from common import load_panel, two_way_resid, flagships, CORE

INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BASELINE, SURFACE = '#c3c2b7', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'

DISPLAY_ORG = {
    'openai': 'OpenAI', 'anthropic': 'Anthropic', 'google': 'Google',
    'alibaba': 'Alibaba', 'meta': 'Meta', 'mistral': 'Mistral', 'xai': 'xAI',
    'deepseek': 'DeepSeek', 'nvidia': 'NVIDIA', 'amazon': 'Amazon',
    'stepfun': 'StepFun', 'cohere': 'Cohere', 'microsoft': 'Microsoft',
    'ibm': 'IBM', 'minimax': 'MiniMax', 'moonshot': 'Moonshot',
    'zai': 'Z.ai', 'meituan': 'Meituan', 'tencent': 'Tencent',
    'ant-group': 'Ant Group', 'baidu': 'Baidu', 'xiaomi': 'Xiaomi',
    'bytedance': 'ByteDance', 'allenai': 'AI2',
}
DISPLAY_CAT = {
    'coding': 'Coding', 'math': 'Math', 'hard_prompts': 'Hard Prompts',
    'instruction_following': 'Instr. Following',
    'creative_writing': 'Creative Writing', 'longer_query': 'Longer Query',
    'multi_turn': 'Multi-Turn', 'english': 'English',
}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 11, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 12,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
})


def strip_axes(ax, keep=('left', 'bottom')):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=3, width=0.6)


def row_resid(P):
    """Firm demeaning only."""
    return P - P.mean(1, keepdims=True)


# ---------------------------------------------------------------- data -----
# identical panel / lab set / snapshot selection to plot_residuals.py
d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
flag = flagships(d)
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
dates = sorted(d.dt.unique())

last = d.dt.max()
snap_last = d[(d.dt == last) & (d.category == 'overall')].dropna(subset=['rating'])
ranked = (snap_last.sort_values('rating', ascending=False)
                    .groupby('org').head(1).sort_values('rating', ascending=False))
ORGS_BROAD = [o for o in ranked.org if o and o not in ('inception-ai',)][:23]


def profile(dt_, orgs):
    out = {}
    for org in orgs:
        m = flag.get((dt_, org))
        if m is None:
            continue
        try:
            r = rat.loc[(dt_, m)][CORE]
        except KeyError:
            continue
        if r.isna().any():
            continue
        out[org] = r.values.astype(float)
    return out


def pairwise_cos(R):
    cs = []
    for a in range(len(R)):
        for b in range(a + 1, len(R)):
            u, v = R[a], R[b]
            n = np.linalg.norm(u) * np.linalg.norm(v)
            if n > 0:
                cs.append(u @ v / n)
    return cs


# ------------------------- pooled distribution, row-demean vs two-way ------
rng = np.random.default_rng(0)
obs_cos, null_cos, obs_two = [], [], []
contrib = []          # per-category contribution to each pair's cosine
for t in dates:
    prof = profile(t, ORGS_BROAD)
    if len(prof) < 4:
        continue
    names = sorted(prof)
    P = np.array([prof[n] for n in names])
    R = row_resid(P)
    obs_cos.extend(pairwise_cos(R))
    obs_two.extend(pairwise_cos(two_way_resid(P)))
    nrm = np.linalg.norm(R, axis=1)
    for a in range(len(R)):
        for b in range(a + 1, len(R)):
            if nrm[a] > 0 and nrm[b] > 0:
                contrib.append(R[a] * R[b] / (nrm[a] * nrm[b]))
    # null: permute firms within each category -- preserves the common
    # across-firm category profile, destroys firm-specific alignment
    for _ in range(20):
        Rp = np.column_stack([rng.permutation(R[:, k]) for k in range(R.shape[1])])
        null_cos.extend(pairwise_cos(Rp))

obs_cos = np.array(obs_cos); null_cos = np.array(null_cos)
obs_two = np.array(obs_two); contrib = np.array(contrib)

print(f'snapshots={len(dates)}  labs<=|{len(ORGS_BROAD)}|')
print(f'FIRM-DEMEANED  pooled pairwise cosine: n={len(obs_cos)}  '
      f'mean={obs_cos.mean():+.4f}  median={np.median(obs_cos):+.4f}  '
      f'sd={obs_cos.std():.4f}  frac>0={(obs_cos > 0).mean():.3f}')
print(f'  permutation null (within-category shuffle): mean={null_cos.mean():+.4f}')
print(f'TWO-WAY (for reference): mean={obs_two.mean():+.4f}')

# ============================================================== FIGURE 1 ===
fig, ax = plt.subplots(figsize=(4.575, 3.05))
xs = np.linspace(-1, 1, 500)
kde_obs = gaussian_kde(obs_cos, bw_method=0.15)(xs)
kde_null = gaussian_kde(null_cos, bw_method=0.15)(xs)
ax.fill_between(xs, kde_null, color=MUTED, alpha=0.18, lw=0)
ax.plot(xs, kde_null, color=MUTED, lw=1.3, ls=(0, (3, 1.5)), label='permutation null')
ax.fill_between(xs, kde_obs, color=BLUE, alpha=0.16, lw=0)
ax.plot(xs, kde_obs, color=BLUE, lw=1.6, label='observed')
ax.axvline(obs_cos.mean(), color=BLUE, lw=0.9, ls=':')
ax.axvline(null_cos.mean(), color=MUTED, lw=0.9, ls=':')
ax.set_xlim(-1, 1)
ax.set_ylim(0, max(kde_obs.max(), kde_null.max()) * 1.28)
ax.set_xlabel(r"$\cos(R_i, R_{i\prime})$")
ax.set_ylabel('density')
ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
ax.legend(frameon=False, fontsize=10.5, loc='upper center', ncol=2,
          bbox_to_anchor=(0.5, 1.10), handlelength=1.6, columnspacing=1.2)
fig.tight_layout(pad=0.6)
fig.savefig('residual_cosine_kde_firmonly.pdf')
plt.close(fig)


# ============================================================== FIGURE 2 ===
def cosine_heatmap(names, P, outpath, figsize, n_clusters=4, resid=row_resid):
    R = resid(P)
    norm = np.linalg.norm(R, axis=1, keepdims=True)
    Rn = R / np.where(norm == 0, 1, norm)
    C = Rn @ Rn.T
    np.fill_diagonal(C, 1.0)
    dist = np.clip(1 - C, 0, 2)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method='average')
    order = dendrogram(Z, no_plot=True)['leaves']
    Co = C[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=figsize)
    n = Co.shape[0]
    edge = SURFACE if n <= 30 else 'none'
    im = ax.pcolormesh(Co, cmap='RdBu_r', vmin=-1, vmax=1,
                       edgecolors=edge, linewidth=0.4)
    ax.set_xlim(0, n); ax.set_ylim(n, 0)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03, aspect=22)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2.5, labelsize=10, color=MUTED)
    cb.set_label('cosine similarity', fontsize=11, color=INK_2)
    fig.tight_layout(pad=0.5)
    fig.savefig(outpath)
    plt.close(fig)

    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    groups = {}
    for name, c in zip(names, clusters):
        groups.setdefault(c, []).append(name)
    return R, order, [names[i] for i in order], groups


best_t, best_prof = None, {}
for t in reversed(dates):
    prof = profile(t, ORGS_BROAD)
    if len(prof) > len(best_prof):
        best_t, best_prof = t, prof
    if len(prof) >= 18:
        best_t, best_prof = t, prof
        break

names = sorted(best_prof)
P = np.array([best_prof[n] for n in names])
R, order, ordered_names, groups = cosine_heatmap(
    names, P, 'residual_cosine_heatmap_firmonly.pdf', (3.35, 3.05))
print(f'\nheatmap snapshot: {pd.Timestamp(best_t).date()}  K={len(names)}')
print('dendrogram order:', [DISPLAY_ORG.get(m, m) for m in ordered_names])
print('\ncluster membership + mean FIRM-DEMEANED profile per category:')
for c, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    idx = [names.index(m) for m in members]
    prof_mean = R[idx].mean(axis=0)
    disp = [DISPLAY_ORG.get(m, m) for m in members]
    pstr = ', '.join(f'{DISPLAY_CAT[c2]}={v:+.0f}' for c2, v in zip(CORE, prof_mean))
    print(f'  cluster {c} ({len(members)}): {disp}')
    print(f'      mean profile: {pstr}')

# ==================== which categories dominate the similarity? ============
print('\n' + '=' * 78)
print('WHICH CATEGORIES DOMINATE  (pooled over all snapshots, K<=23 labs)')
print('=' * 78)

# (a) additive decomposition of the mean cosine: sum_k contrib_k = mean cosine
cmean = contrib.mean(0)
print('\n(a) per-category contribution to the mean pairwise cosine '
      f'(sums to {cmean.sum():+.4f}):')
for k in np.argsort(-cmean):
    print(f'    {DISPLAY_CAT[CORE[k]]:<18} {cmean[k]:+.4f}   '
          f'({100 * cmean[k] / cmean.sum():5.1f}% of total)')

# (b) leave-one-category-out: mean cosine with category k removed
print('\n(b) leave-one-out: mean pairwise cosine with category k dropped '
      f'(full = {obs_cos.mean():+.4f}):')
loo = {}
for k, cat in enumerate(CORE):
    keep = [j for j in range(len(CORE)) if j != k]
    cs = []
    for t in dates:
        prof = profile(t, ORGS_BROAD)
        if len(prof) < 4:
            continue
        Pt = np.array([prof[n] for n in sorted(prof)])
        cs.extend(pairwise_cos(row_resid(Pt)[:, keep]))
    loo[cat] = np.mean(cs)
for cat, v in sorted(loo.items(), key=lambda kv: kv[1]):
    print(f'    drop {DISPLAY_CAT[cat]:<18} {v:+.4f}   '
          f'(delta {v - obs_cos.mean():+.4f})')

# (c) common vs idiosyncratic variance, per category, at the heatmap snapshot
print('\n(c) variance share per category at the heatmap snapshot '
      '(common = across-firm category mean, idio = deviation from it):')
colmean = R.mean(0)
tot = (R ** 2).sum()
rows = []
for k, cat in enumerate(CORE):
    common = len(R) * colmean[k] ** 2
    idio = ((R[:, k] - colmean[k]) ** 2).sum()
    rows.append((DISPLAY_CAT[cat], colmean[k], 100 * (common + idio) / tot,
                 100 * common / tot, 100 * idio / tot))
tab = pd.DataFrame(rows, columns=['category', 'mean_resid', 'var_share_%',
                                  'common_%', 'idio_%'])
print(tab.sort_values('var_share_%', ascending=False).to_string(
    index=False, float_format=lambda x: f'{x:+.2f}'))
print(f'  TOTAL common share = {tab["common_%"].sum():.1f}%  '
      f'idiosyncratic = {tab["idio_%"].sum():.1f}%')

# (d) leading principal direction of the firm-demeaned profiles
U, S, Vt = np.linalg.svd(R, full_matrices=False)
ev = S ** 2 / (S ** 2).sum()
print(f'\n(d) SVD of the K x 8 firm-demeaned matrix at the heatmap snapshot:')
print(f'    variance shares: ' + ', '.join(f'{100*e:.1f}%' for e in ev))
print(f'    participation ratio (effective dims) = '
      f'{(S**2).sum()**2 / (S**4).sum():.2f} of {len(CORE)}')
for j in range(2):
    load = ', '.join(f'{DISPLAY_CAT[c]}={v:+.2f}'
                     for c, v in sorted(zip(CORE, Vt[j]), key=lambda kv: -abs(kv[1])))
    print(f'    PC{j+1} ({100*ev[j]:.1f}%): {load}')

# ============================================================== FIGURE 3 ===
fig, ax = plt.subplots(figsize=(4.9, 3.05))
ordk = np.argsort(cmean)
ypos = np.arange(len(CORE))
cols = [BLUE if cmean[k] >= 0 else ORANGE for k in ordk]
ax.barh(ypos, cmean[ordk], color=cols, height=0.62, lw=0)
ax.axvline(0, color=BASELINE, lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels([DISPLAY_CAT[CORE[k]] for k in ordk])
ax.set_xlabel('contribution to mean $\\cos(R_i, R_{i\\prime})$')
ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax, keep=('bottom',))
fig.tight_layout(pad=0.6)
fig.savefig('residual_category_contrib.pdf')
plt.close(fig)
print('\nwrote residual_cosine_kde_firmonly.pdf, residual_cosine_heatmap_firmonly.pdf, '
      'residual_category_contrib.pdf')

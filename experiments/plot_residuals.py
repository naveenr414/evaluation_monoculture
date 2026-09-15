"""Three PDF figures on the two-way (firm x category) FE residual: theta_i,k
demeaned by firm and by category, R = P - rowmean - colmean (see common.two_way_resid).

  1. residual_cosine_kde.pdf        overall distribution of pairwise cosine(R_i, R_i')
                                     pooled across snapshots, vs. the permutation null
  2. residual_cosine_heatmap.pdf    pairwise cosine matrix for the top-23 labs at one
                                     snapshot, reordered by hierarchical clustering
  2b. residual_cosine_heatmap_top100.pdf   same, at MODEL granularity, top-100 models
  3. residual_timeseries.pdf        smoothed (org, category) residual trajectories,
                                     old->new bottom->top, 2 outliers highlighted
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde

sys.path.insert(0, 'experiments')
from common import load_panel, two_way_resid, flagships, CORE

# ---------------------------------------------------------------- style ----
INK       = '#0b0b0b'
INK_2     = '#52514e'
MUTED     = '#898781'
GRID      = '#e1e0d9'
BASELINE  = '#c3c2b7'
SURFACE   = '#ffffff'
BLUE, ORANGE, AQUA, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#e34948'
GOLD, TEAL = '#c98500', '#0d7d6e'

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
    'instruction_following': 'Instruction Following',
    'creative_writing': 'Creative Writing', 'longer_query': 'Longer Query',
    'multi_turn': 'Multi-Turn', 'english': 'English',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'text.color': INK,
    'axes.edgecolor': BASELINE,
    'axes.labelcolor': INK,
    'axes.labelsize': 12,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.linewidth': 0.7,
    'pdf.fonttype': 42,
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


cov = pd.Series({org: sum(org in profile(t, [org]) for t in dates) for org in ORGS_BROAD})
ORGS_STABLE = cov[cov >= 0.85 * len(dates)].index.tolist()
print(f'ORGS_BROAD  (K={len(ORGS_BROAD)}): {ORGS_BROAD}')
print(f'ORGS_STABLE (K={len(ORGS_STABLE)}, >=85% snapshot coverage): {ORGS_STABLE}')


def pairwise_cos(R, names):
    cs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            u, v = R[a], R[b]
            n = np.linalg.norm(u) * np.linalg.norm(v)
            if n > 0:
                cs.append(u @ v / n)
    return cs


rng = np.random.default_rng(0)
obs_cos, null_cos = [], []
for t in dates:
    prof = profile(t, ORGS_BROAD)
    if len(prof) < 4:
        continue
    names = sorted(prof)
    P = np.array([prof[n] for n in names])
    R = two_way_resid(P)
    obs_cos.extend(pairwise_cos(R, names))
    for _ in range(20):
        Rp = np.column_stack([rng.permutation(R[:, k]) for k in range(R.shape[1])])
        null_cos.extend(pairwise_cos(Rp, names))

obs_cos, null_cos = np.array(obs_cos), np.array(null_cos)
print(f'pooled pairwise cosines: n={len(obs_cos)}  mean={obs_cos.mean():+.4f}  '
      f'null mean={null_cos.mean():+.4f}')

# ============================================================= FIGURE 1 ====
# same height as the heatmap (3.05in) so the two sit side by side at equal
# height; 3:2 width:height for the density curve rather than forcing it square
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
ymax = max(kde_obs.max(), kde_null.max())
ax.set_ylim(0, ymax * 1.28)
ax.set_xlabel(r"$\cos(R_i, R_{i\prime})$")
ax.set_ylabel('density')
ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
ax.legend(frameon=False, fontsize=10.5, loc='upper center', ncol=2,
          bbox_to_anchor=(0.5, 1.10), handlelength=1.6, columnspacing=1.2)
fig.tight_layout(pad=0.6)
fig.savefig('residual_cosine_kde.pdf')
plt.close(fig)


def cosine_heatmap(names, P, outpath, figsize, n_clusters=4):
    R = two_way_resid(P)
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
    # pcolormesh draws each cell as a real vector quad -- crisp at any zoom/DPI.
    # imshow instead embeds a KxK-pixel raster in the PDF that PDF viewers/
    # rasterizers then resample (blurry at K=23, blocky-then-blurry at K=100).
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
    return R, order, groups


# ------------------------------------------------------- FIGURE 2 (K=23) ---
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
R, order, groups = cosine_heatmap(names, P, 'residual_cosine_heatmap.pdf', (3.35, 3.05))
print(f'\nheatmap snapshot: {best_t}  K={len(names)}')
print('cluster membership + mean residual profile per category (blank=fully-overlapping cats only):')
for c, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    idx = [names.index(m) for m in members]
    prof_mean = R[idx].mean(axis=0)
    disp = [DISPLAY_ORG.get(m, m) for m in members]
    profile_str = ', '.join(f'{DISPLAY_CAT[c2]}={v:+.0f}' for c2, v in zip(CORE, prof_mean))
    print(f'  cluster {c} ({len(members)}): {disp}')
    print(f'      mean profile: {profile_str}')

# ------------------------------------------------- FIGURE 2b (top-100 models) --
snap_pool = d[(d.dt == best_t) & (d.category == 'overall')].dropna(subset=['rating'])
top_models = snap_pool.sort_values('rating', ascending=False).model_name.head(150).tolist()
mnames = []
for m in top_models:
    try:
        r = rat.loc[(best_t, m)][CORE]
    except KeyError:
        continue
    if r.isna().any():
        continue
    mnames.append(m)
    if len(mnames) == 100:
        break
Pm = np.array([rat.loc[(best_t, m)][CORE].values.astype(float) for m in mnames])
cosine_heatmap(mnames, Pm, 'residual_cosine_heatmap_top100.pdf', (3.6, 3.3), n_clusters=6)
print(f'\ntop-100-models heatmap snapshot: {best_t}  K={len(mnames)}')

# ============================================================= FIGURE 3 ====
cols = pd.MultiIndex.from_product([ORGS_STABLE, CORE], names=['org', 'cat'])
M = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=cols, dtype=float)
for t in dates:
    prof = profile(t, ORGS_STABLE)
    if len(prof) < 4:
        continue
    names_t = sorted(prof)
    Pt = np.array([prof[n] for n in names_t])
    Rt = two_way_resid(Pt)
    for i, org in enumerate(names_t):
        for j, cat in enumerate(CORE):
            M.loc[t, (org, cat)] = Rt[i, j]

# -------- persistence: is any (org, cat) residual consistently one-signed? --
def longest_streak(vals):
    sign = np.sign(vals)
    best_pos = best_neg = cur_pos = cur_neg = 0
    for x in sign:
        if x > 0:
            cur_pos += 1; cur_neg = 0
        elif x < 0:
            cur_neg += 1; cur_pos = 0
        else:
            cur_pos = cur_neg = 0
        best_pos, best_neg = max(best_pos, cur_pos), max(best_neg, cur_neg)
    return best_pos, best_neg


for label, freq in [('month', 'MS'), ('quarter', 'QS')]:
    Mp = M.resample(freq).mean()
    rows = []
    for col in Mp.columns:
        s = Mp[col].dropna()
        if len(s) < 4:
            continue
        bp, bn = longest_streak(s.values)
        rows.append((*col, len(s), bp, bn, max(bp, bn), np.nanmean(np.abs(s.values))))
    tab = pd.DataFrame(rows, columns=['org', 'cat', f'n_{label}s', 'longest_pos', 'longest_neg',
                                       'best', 'mean_abs']).sort_values(
        ['best', 'mean_abs'], ascending=False)
    print(f'\npersistence table ({label}-aggregated, top 12 by longest same-sign run, '
          f'ties broken by |residual|):')
    print(tab.head(12).to_string(index=False))
    a_c = tab[(tab.org == 'anthropic') & (tab.cat == 'coding')]
    if len(a_c):
        print(f"  anthropic/coding: {a_c.iloc[0].to_dict()}")
    if label == 'month':
        month_tab = tab


# -------- is the persistence statistically real, or just noise/autocorrelation? --
# Streak length alone isn't a test: a mean-zero but autocorrelated series (e.g. a
# flagship model rated repeatedly between releases) produces long runs by chance.
# The right test is whether the TIME-MEAN of the residual is nonzero, using
# autocorrelation-robust (Newey-West/HAC) standard errors so the many redundant
# weekly snapshots of an unchanged flagship don't masquerade as independent draws.
import statsmodels.api as sm


def hac_test(s, maxlags):
    s = s.dropna()
    if len(s) < 8:
        return None
    y = s.values
    model = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model.params[0], model.bse[0], model.tvalues[0], model.pvalues[0], len(y)


def bh_fdr(pvals):
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q


hac_rows = []
for col in M.columns:
    r = hac_test(M[col], maxlags=12)
    if r is None:
        continue
    mean, se, t, p, n = r
    hac_rows.append((*col, mean, se, t, p, n))
hac_tab = pd.DataFrame(hac_rows, columns=['org', 'cat', 'mean', 'se', 't', 'p', 'n'])
hac_tab['q_bh'] = bh_fdr(hac_tab['p'].values)
print(f'\nHAC (Newey-West) t-test on mean residual != 0, weekly snapshots, maxlags=12:')
print(f'  {len(hac_tab)} (org,cat) pairs tested; significant at raw p<0.05: '
      f'{(hac_tab.p < 0.05).sum()}; at BH q<0.05: {(hac_tab.q_bh < 0.05).sum()}')
print(hac_tab.sort_values('p').head(10).to_string(index=False,
      formatters={'mean': '{:+.2f}'.format, 't': '{:+.2f}'.format,
                  'p': '{:.4f}'.format, 'q_bh': '{:.4f}'.format}))
hac_lookup = hac_tab.set_index(['org', 'cat'])
a_c = hac_lookup.loc[('anthropic', 'coding')]
print(f"  anthropic/coding: mean={a_c['mean']:+.2f}  t={a_c.t:+.2f}  p={a_c.p:.5f}  q_bh={a_c.q_bh:.5f}")

# smooth by literal monthly aggregation (fewer, less noisy points) rather than
# a rolling window over the raw ~weekly snapshots
Ms = M.resample('MS').mean()

# highlight anthropic/coding directly (the pair we were asked about) plus
# whichever other pair has the longest same-sign run (ties broken by |residual|)
n_out = 2
outliers = [('anthropic', 'coding')]
for _, row in month_tab.iterrows():
    pair = (row.org, row['cat'])
    if pair not in outliers:
        outliers.append(pair)
    if len(outliers) == n_out:
        break
import matplotlib.patheffects as pe

OUTLIER_COLORS = [BLUE, RED]

# transposed (time on x, value on y) -- more compact than the time-on-y version
fig, ax = plt.subplots(figsize=(7.0, 3.1))
vals = Ms.to_numpy(dtype=float)

# a plain, uniform gray backdrop -- a magnitude-colored gradient here would need
# its own colorbar/legend to mean anything, and without one just competes with
# the two colors that actually carry meaning
for col in Ms.columns:
    if col in outliers:
        continue
    s = Ms[col].dropna()
    if len(s) < 2:
        continue
    ax.plot(s.index, s.values, color=MUTED, lw=0.6, alpha=0.30, zorder=1)

def streak_span(vals):
    """(start_idx, end_idx, sign) of the longest same-sign run."""
    sign = np.sign(vals)
    best = (0, 0, 0, 0)  # length, start, end, sign
    cur_len, cur_start, cur_sign = 0, 0, 0
    for i, x in enumerate(sign):
        s = 1 if x > 0 else (-1 if x < 0 else 0)
        if s != 0 and s == cur_sign:
            cur_len += 1
        else:
            cur_len, cur_start, cur_sign = 1, i, s
        if cur_len > best[0]:
            best = (cur_len, cur_start, i, cur_sign)
    return best


ymax = np.nanmax(np.abs(vals))
halo = [pe.withStroke(linewidth=3.2, foreground=SURFACE)]
for k, col in enumerate(outliers):
    s = Ms[col].dropna()
    if len(s) < 2:
        continue
    color = OUTLIER_COLORS[k % len(OUTLIER_COLORS)]
    length, i0, i1, sign = streak_span(s.values)
    ax.plot(s.index, s.values, color=color, lw=2.0, zorder=3,
             solid_capstyle='round', path_effects=halo)
    # label right at the line's own peak, above it if the line runs positive,
    # below if it runs negative -- just names it, no streak length/p-value
    peak_i = np.argmax(np.abs(s.values))
    lab_x, lab_y = s.index[peak_i], s.values[peak_i]
    org, cat = col
    text = f'{DISPLAY_ORG.get(org, org)} · {DISPLAY_CAT[cat]}'
    va = 'bottom' if sign > 0 else 'top'
    dy = 7 if sign > 0 else -7
    ax.annotate(text, xy=(lab_x, lab_y), xytext=(0, dy), textcoords='offset points',
                fontsize=11, color=color, ha='center', va=va, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2.6, foreground=SURFACE)])

ax.axhline(0, color=INK_2, lw=1.4, ls=(0, (5, 3)), zorder=4)
ax.set_ylim(-ymax * 1.18, ymax * 1.18)
ax.set_xlim(Ms.index.min(), Ms.index.max())
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
ax.set_ylabel(r'$r_{i,k}$')
ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax, keep=('left', 'bottom'))
fig.tight_layout(pad=0.7)
fig.savefig('residual_timeseries.pdf')
plt.close(fig)
print(f'\ntimeseries highlighted (longest monthly same-sign streak): {outliers}')

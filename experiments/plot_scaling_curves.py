"""RQ (production functions via scaling): theta_{i,k} = a_{i,k} + A_{i,k} log(params).

Same data/filters as expG_production_functions.py (open-weight labs, Open LLM
Leaderboard's 6 objective dims, MIN_N/MIN_SPAN screen for a real scaling ladder).

  1. scaling_curves_raw.pdf         small multiples (1 panel/category), RAW scores,
                                      one color/fit-line per firm
  2. scaling_curves_normalized.pdf  same, but y is standardized per category with a
                                      SHARED y-range across panels -- lets you compare
                                      slopes across categories directly
  3. scaling_by_category.pdf        one panel: pooled (all-firms) fit + 95% CI ribbon
                                      per category, overlaid -- "scaling curves by
                                      category" instead of by firm
  4. scaling_overall_permodel.pdf   one panel: each model's mean standardized score
                                      across the 6 categories ("overall capability")
                                      vs log(params), one color/fit-line per firm
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK_2, MUTED, GRID, BASELINE, SURFACE = \
    '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
DIM_LABEL = {'ifeval': 'IFEval', 'bbh': 'BBH', 'math': 'MATH', 'gpqa': 'GPQA',
             'musr': 'MuSR', 'mmlu_pro': 'MMLU-Pro'}
LABS = ['qwen', 'google', 'meta-llama', 'mistralai', 'deepseek-ai', 'microsoft',
        '01-ai', 'ibm-granite', 'allenai', 'huggingfacetb', 'tiiuae', 'eleutherai',
        'nvidia', 'cohereforai', 'nousresearch', 'bigscience', 'arcee-ai']
LAB_LABEL = {'qwen': 'Qwen', 'google': 'Google', 'meta-llama': 'Meta', 'mistralai': 'Mistral',
             'deepseek-ai': 'DeepSeek', 'microsoft': 'Microsoft', '01-ai': '01.AI',
             'ibm-granite': 'IBM Granite', 'allenai': 'AI2', 'huggingfacetb': 'HuggingFaceTB',
             'tiiuae': 'TII', 'eleutherai': 'EleutherAI', 'nvidia': 'NVIDIA',
             'cohereforai': 'Cohere For AI', 'nousresearch': 'NousResearch',
             'bigscience': 'BigScience', 'arcee-ai': 'Arcee AI'}
MIN_N, MIN_SPAN = 7, 0.6

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 28, 'text.color': INK,
    'axes.edgecolor': BASELINE, 'axes.labelcolor': INK, 'axes.labelsize': 33,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 26, 'ytick.labelsize': 26,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.linewidth': 1.2, 'pdf.fonttype': 42,
})


def strip_axes(ax, keep=('left', 'bottom')):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(length=3, width=0.6)


b = pd.read_parquet('data/benchmarks.parquet')
b['org'] = b.hf_model_id.str.split('/').str[0].str.lower()
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged]
b = b.dropna(subset=DIMS)
b['lp'] = np.log10(b.params_b)
b_raw = b.copy()

for k in DIMS:
    b[k] = (b[k] - b[k].mean()) / b[k].std()

b = b[b.org.isin(LABS)].copy()
b_raw = b_raw[b_raw.org.isin(LABS)].copy()

keep = []
for org, g in b.groupby('org'):
    span = np.log10(g.params_b.max() / g.params_b.min())
    if len(g) >= MIN_N and span >= MIN_SPAN and g.params_b.nunique() >= 4:
        keep.append(org)
b = b[b.org.isin(keep)].copy()
b_raw = b_raw[b_raw.org.isin(keep)].copy()
print(f'labs kept: {len(keep)} -> {keep}')

CAT_COLOR = dict(zip(DIMS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#e34948']))

# highlight only the 3-4 most-represented/well-known labs; everyone else stays
# gray -- the question each panel should answer is "how much do firms deviate
# from the common category curve", not "which color is which of 15 firms"
HIGHLIGHT = ['qwen', 'google', 'meta-llama', 'mistralai']
HIGHLIGHT_COLOR = {'qwen': '#2a78d6', 'google': '#eb6834',
                    'meta-llama': '#1baf7a', 'mistralai': '#4a3aa7'}


def fit_line(x, y):
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # intercept, slope


# ================================================== FIGURE A (1 & 2) =======
# "does scaling exist, and does it vary by category" -- NOT "which color is
# which of 15 firms". Raw points + firm fits are all gray (deemphasized);
# only the pooled category-level fit is bold; a handful of major labs are
# picked out in color so the eye has somewhere to rest without a 15-entry key.
for label, data, ylab_fn, shared_ylim in [
        ('raw', b_raw, lambda k: DIM_LABEL[k], False),
        ('normalized', b, lambda k: DIM_LABEL[k] + ' (z-score)', True)]:

    fig, axes = plt.subplots(2, 3, figsize=(24.5, 10.3))
    axes = axes.ravel()
    for ax, k in zip(axes, DIMS):
        ax.scatter(data.lp, data[k], s=5, color=MUTED, alpha=0.18,
                   edgecolor='none', zorder=1)
        for org in sorted(keep):
            if org in HIGHLIGHT:
                continue
            g = data[data.org == org]
            if len(g) < 2:
                continue
            icpt, slope = fit_line(g.lp.values, g[k].values)
            xs = np.linspace(g.lp.min(), g.lp.max(), 20)
            ax.plot(xs, icpt + slope * xs, color=MUTED, lw=0.8, alpha=0.40, zorder=2)
        for org in HIGHLIGHT:
            g = data[data.org == org]
            if len(g) < 2:
                continue
            icpt, slope = fit_line(g.lp.values, g[k].values)
            xs = np.linspace(g.lp.min(), g.lp.max(), 20)
            ax.plot(xs, icpt + slope * xs, color=HIGHLIGHT_COLOR[org], lw=3.2,
                    alpha=0.95, zorder=3)
        icpt_p, slope_p = fit_line(data.lp.values, data[k].values)
        xs_p = np.linspace(data.lp.min(), data.lp.max(), 40)
        ax.plot(xs_p, icpt_p + slope_p * xs_p, color=INK, lw=4.6, zorder=4)
        ax.set_title(ylab_fn(k), fontsize=28, loc='left', color=INK)
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        strip_axes(ax)
    if shared_ylim:
        lo = min(ax.get_ylim()[0] for ax in axes)
        hi = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(lo, hi)
    for ax in axes[3:]:
        ax.set_xlabel(r'$\log_{10}$(params, B)')
    axes[0].set_ylabel('score')
    axes[3].set_ylabel('score')

    handles = ([plt.Line2D([0], [0], color=INK, lw=4.6, label='pooled (all labs)')]
               + [plt.Line2D([0], [0], color=HIGHLIGHT_COLOR[o], lw=3.2, label=LAB_LABEL[o])
                  for o in HIGHLIGHT]
               + [plt.Line2D([0], [0], color=MUTED, lw=1.4, alpha=0.6, label='other labs')])
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=len(handles), frameon=False, fontsize=26, handlelength=1.6,
               columnspacing=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f'scaling_curves_{label}.pdf', bbox_inches='tight')
    plt.close(fig)

# ============================================================= FIGURE 3 ====
fig, ax = plt.subplots(figsize=(6.4, 4.4))
xs = np.linspace(b.lp.min(), b.lp.max(), 60)
rng = np.random.default_rng(0)
for k in DIMS:
    x, y = b.lp.values, b[k].values
    icpt, slope = fit_line(x, y)
    boots = []
    for _ in range(400):
        idx = rng.integers(0, len(x), len(x))
        i2, s2 = fit_line(x[idx], y[idx])
        boots.append(i2 + s2 * xs)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ax.fill_between(xs, lo, hi, color=CAT_COLOR[k], alpha=0.15, lw=0, zorder=1)
    ax.plot(xs, icpt + slope * xs, color=CAT_COLOR[k], lw=2.0, zorder=3,
            label=f'{DIM_LABEL[k]}  (slope={slope:+.2f})')

ax.set_xlabel(r'$\log_{10}$(params, B)')
ax.set_ylabel('standardized score (z-score, pooled across labs)')
ax.grid(color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
ax.legend(frameon=False, fontsize=8, loc='upper left', handlelength=1.4)
fig.tight_layout(pad=0.7)
fig.savefig('scaling_by_category.pdf')
plt.close(fig)

# ============================================================= FIGURE 4 ====
b['overall'] = b[DIMS].mean(axis=1)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.scatter(b.lp, b.overall, s=8, color=MUTED, alpha=0.20, edgecolor='none', zorder=1)
for org in sorted(keep):
    if org in HIGHLIGHT:
        continue
    g = b[b.org == org]
    if len(g) < 2:
        continue
    icpt, slope = fit_line(g.lp.values, g.overall.values)
    xs = np.linspace(g.lp.min(), g.lp.max(), 20)
    ax.plot(xs, icpt + slope * xs, color=MUTED, lw=0.8, alpha=0.40, zorder=2)
for org in HIGHLIGHT:
    g = b[b.org == org]
    if len(g) < 2:
        continue
    icpt, slope = fit_line(g.lp.values, g.overall.values)
    xs = np.linspace(g.lp.min(), g.lp.max(), 20)
    ax.plot(xs, icpt + slope * xs, color=HIGHLIGHT_COLOR[org], lw=2.2, zorder=3,
            label=f'{LAB_LABEL[org]}  ({slope:+.2f})')
icpt_p, slope_p = fit_line(b.lp.values, b.overall.values)
xs_p = np.linspace(b.lp.min(), b.lp.max(), 40)
ax.plot(xs_p, icpt_p + slope_p * xs_p, color=INK, lw=3.2, zorder=4,
        label=f'pooled ({slope_p:+.2f})')

ax.set_xlabel(r'$\log_{10}$(params, B)')
ax.set_ylabel('overall capability (mean z-score across 6 categories)')
ax.grid(color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
strip_axes(ax)
ax.legend(frameon=False, fontsize=9, loc='upper left', handlelength=1.4)
fig.tight_layout(pad=0.7)
fig.savefig('scaling_overall_permodel.pdf')
plt.close(fig)

# ============================================================= FIGURE B ====
# firm x category heatmap of beta_hat_{i,k} -- does scaling differ MOSTLY by
# firm, mostly by category, or specifically by the firm x category cell?
# (ties directly to Table 3: firm-only heterogeneity buys ~0.8% held-out
# improvement; firm x category heterogeneity buys ~31.8%.)
beta = pd.DataFrame(index=sorted(keep), columns=DIMS, dtype=float)
for org in sorted(keep):
    for k in DIMS:
        g = b[b.org == org]
        if len(g) >= 2:
            beta.loc[org, k] = fit_line(g.lp.values, g[k].values)[1]

# transposed: categories as rows (only 6 -> horizontal labels, no rotation
# needed), firms as columns (15, spread across the full figure width) -- this
# lets the figure use the WHOLE page width instead of a narrow tall strip, so
# the same 90 cells get much more room per label/cell at a given font size.
col_order = beta.mean(axis=1).sort_values(ascending=False).index
beta_t = beta.loc[col_order].T  # rows=DIMS order (already DIMS columns), cols=firms
vmax = np.nanmax(np.abs(beta_t.values))

fig, ax = plt.subplots(figsize=(24.5, 7.2))
im = ax.pcolormesh(beta_t.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    edgecolors=SURFACE, linewidth=1.6)
ax.set_xticks(np.arange(len(col_order)) + 0.5)
ax.set_xticklabels([LAB_LABEL[o] for o in col_order], rotation=40, ha='right', fontsize=27)
ax.set_yticks(np.arange(len(DIMS)) + 0.5)
ax.set_yticklabels([DIM_LABEL[k] for k in DIMS], fontsize=29)
ax.set_xlim(0, len(col_order)); ax.set_ylim(len(DIMS), 0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015, aspect=16)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=26)
cb.set_label(r'$\hat\beta_{i,k}$  (z-score per log$_{10}$(params, B))', fontsize=27)
fig.tight_layout(pad=0.7)
fig.savefig('scaling_beta_heatmap.pdf', bbox_inches='tight')
plt.close(fig)

print('wrote scaling_curves_raw.pdf, scaling_curves_normalized.pdf, '
      'scaling_by_category.pdf, scaling_overall_permodel.pdf, scaling_beta_heatmap.pdf')

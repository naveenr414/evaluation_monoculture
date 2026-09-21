"""Parameters vs. performance curves per benchmark category, overlaid with CI bands.
Pooled across all open-weight labs (not per-lab) to see whether the SHAPE of the
scaling curve looks similar across the 6 Open LLM Leaderboard dimensions.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
COLORS = {'ifeval': '#4285F4', 'bbh': '#EA4335', 'math': '#FBBC05',
          'gpqa': '#34A853', 'musr': '#9C27B0', 'mmlu_pro': '#00ACC1'}

b = pd.read_parquet('data/benchmarks.parquet')
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged]
b = b.dropna(subset=DIMS)
b['lp'] = np.log10(b.params_b)
print(f'n models = {len(b)}, param range = {b.params_b.min():.2f}B - {b.params_b.max():.1f}B')

N_BINS = 14
bins = np.linspace(b.lp.min(), b.lp.max(), N_BINS + 1)
b['bin'] = pd.cut(b.lp, bins)
bin_centers = (bins[:-1] + bins[1:]) / 2

fig, ax = plt.subplots(figsize=(11, 7))
for k in DIMS:
    stats = b.groupby('bin', observed=True)[k].agg(['mean', 'std', 'count'])
    stats = stats[stats['count'] >= 3].sort_index()
    x = np.array([iv.mid for iv in stats.index])
    y = stats['mean'].values
    sem = stats['std'].values / np.sqrt(stats['count'].values)
    ci = 1.96 * sem
    ax.plot(x, y, '-o', color=COLORS[k], label=k, lw=2, markersize=4)
    ax.fill_between(x, y - ci, y + ci, color=COLORS[k], alpha=0.15)

ax.set_xlabel('log10(parameters, billions)')
ax.set_ylabel('benchmark score (0-100)')
ax.set_title('Parameters vs. performance, by category (pooled across labs, 95% CI bands)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('experiments/scratch_persistence/params_vs_perf_raw.png', dpi=130, bbox_inches='tight')
print('saved raw-scale version')

# ---- normalized version: each dim rescaled to its own 5th-95th pct range, so
# curve SHAPE is comparable even though absolute difficulty differs a lot ----
fig2, ax2 = plt.subplots(figsize=(11, 7))
for k in DIMS:
    lo, hi = b[k].quantile([0.02, 0.98])
    bk = (b[k] - lo) / (hi - lo)
    stats = bk.groupby(b['bin'], observed=True).agg(['mean', 'std', 'count'])
    stats = stats[stats['count'] >= 3].sort_index()
    x = np.array([iv.mid for iv in stats.index])
    y = stats['mean'].values
    sem = stats['std'].values / np.sqrt(stats['count'].values)
    ci = 1.96 * sem
    ax2.plot(x, y, '-o', color=COLORS[k], label=k, lw=2, markersize=4)
    ax2.fill_between(x, y - ci, y + ci, color=COLORS[k], alpha=0.15)

ax2.set_xlabel('log10(parameters, billions)')
ax2.set_ylabel('normalized score (2nd-98th pctile rescaled to 0-1)')
ax2.set_title('Parameters vs. performance, NORMALIZED per category (shape comparison, 95% CI bands)')
ax2.legend(loc='upper left')
ax2.axhline(0, color='gray', lw=0.5)
ax2.axhline(1, color='gray', lw=0.5)
plt.tight_layout()
plt.savefig('experiments/scratch_persistence/params_vs_perf_normalized.png', dpi=130, bbox_inches='tight')
print('saved normalized version')

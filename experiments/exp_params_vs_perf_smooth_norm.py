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

# normalize each dim to its own 2nd-98th percentile range (0-1)
for k in DIMS:
    lo, hi = b[k].quantile([0.02, 0.98])
    b[k] = (b[k] - lo) / (hi - lo)

BANDWIDTH = 0.18
grid = np.linspace(b.lp.min() + 0.15, b.lp.max() - 0.05, 150)

def nw_smooth(x, y, grid, bw):
    x = x.values; y = y.values
    out = np.full(len(grid), np.nan)
    neff = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((x - g) / bw) ** 2)
        sw = w.sum()
        if sw < 1e-6:
            continue
        out[i] = (w * y).sum() / sw
        neff[i] = sw ** 2 / (w ** 2).sum()
    return out, neff

N_BOOT = 300
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(11, 7))
for k in DIMS:
    y0, neff = nw_smooth(b.lp, b[k], grid, BANDWIDTH)
    boots = np.zeros((N_BOOT, len(grid)))
    idx_all = np.arange(len(b))
    for t in range(N_BOOT):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        yb, _ = nw_smooth(b.lp.iloc[idx], b[k].iloc[idx], grid, BANDWIDTH)
        boots[t] = yb
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    valid = neff >= 15
    ax.plot(grid[valid], y0[valid], '-', color=COLORS[k], label=k, lw=2.2)
    ax.fill_between(grid[valid], lo[valid], hi[valid], color=COLORS[k], alpha=0.15)

ax.axhline(0, color='gray', lw=0.5)
ax.axhline(1, color='gray', lw=0.5)
ax.set_xlabel('log10(parameters, billions)')
ax.set_ylabel('normalized score (2nd-98th pctile rescaled to 0-1)')
ax.set_title('Parameters vs. performance, NORMALIZED, continuous kernel smoothing (bw=0.18 dex, 95% bootstrap CI)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('experiments/scratch_persistence/params_vs_perf_smooth_normalized.png', dpi=130, bbox_inches='tight')
print('saved')

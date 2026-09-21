"""
C1 Analysis: Has Arena ELO - benchmark correlation declined over time?

Analysis 1: First-appearance cohorts - Spearman(ELO_first, benchmark) by 6-month cohort
Analysis 2: Rolling correlation over snapshots across all present models
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/matched.parquet')
matched = df[df['avg_score'].notna()].copy()
matched['leaderboard_publish_date'] = pd.to_datetime(matched['leaderboard_publish_date'])

def spearman_ci(x, y, n_boot=1000, ci=95):
    """Spearman rho with bootstrap CI."""
    x, y = np.array(x), np.array(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, len(x)
    rho, _ = stats.spearmanr(x, y)
    boot = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot.append(r)
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return rho, lo, hi, len(x)

# ── Analysis 1: First-appearance cohorts ─────────────────────────────────────
print("=" * 65)
print("ANALYSIS 1: First-appearance ELO vs benchmarks by cohort")
print("=" * 65)

# For each model, get its ELO at first appearance
first = (matched.sort_values('leaderboard_publish_date')
         .groupby('model_name').first().reset_index())
first['cohort'] = pd.cut(
    first['leaderboard_publish_date'],
    bins=pd.to_datetime([
        '2023-01-01','2023-07-01','2024-01-01',
        '2024-07-01','2025-01-01','2025-07-01','2026-01-01'
    ]),
    labels=['2023H1','2023H2','2024H1','2024H2','2025H1','2025H2'],
    right=False
)

print(f"\n{'Cohort':<10} {'N':>4}  {'Spearman ρ':>10}  {'95% CI':>20}  {'p-value':>9}")
print("-" * 60)

cohort_results = []
for cohort in ['2023H1','2023H2','2024H1','2024H2','2025H1','2025H2']:
    sub = first[first['cohort'] == cohort]
    rho, lo, hi, n = spearman_ci(sub['rating'], sub['avg_score'])
    if np.isnan(rho):
        print(f"{cohort:<10} {n:>4}  {'too few':>10}")
        continue
    _, pval = stats.spearmanr(sub['rating'].dropna(), sub['avg_score'].dropna())
    sig = '*' if pval < 0.05 else ''
    print(f"{cohort:<10} {n:>4}  {rho:>10.3f}  [{lo:>6.3f}, {hi:>6.3f}]  {pval:>9.4f} {sig}")
    cohort_results.append({'cohort': cohort, 'n': n, 'rho': rho, 'lo': lo, 'hi': hi, 'pval': pval})

# Per-benchmark breakdown for first+last cohort
print("\n--- Per-benchmark Spearman ρ: 2023H1 vs 2024H2 vs 2025H1 ---")
benchmarks = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro', 'avg_score']
print(f"{'Benchmark':<12}", end='')
for c in ['2023H1', '2023H2', '2024H1', '2024H2', '2025H1', '2025H2']:
    print(f"  {c:>8}", end='')
print()
print("-" * 70)
for b in benchmarks:
    print(f"{b:<12}", end='')
    for cohort in ['2023H1','2023H2','2024H1','2024H2','2025H1','2025H2']:
        sub = first[first['cohort'] == cohort]
        if len(sub) < 5:
            print(f"  {'—':>8}", end='')
            continue
        rho, _, _, _ = spearman_ci(sub['rating'], sub[b])
        print(f"  {rho:>8.3f}", end='')
    print()

# ── Analysis 2: Rolling correlation over all snapshots ────────────────────────
print("\n" + "=" * 65)
print("ANALYSIS 2: Rolling Spearman correlation over snapshots")
print("=" * 65)

snap_results = []
for date, grp in matched.groupby('leaderboard_publish_date'):
    rho, lo, hi, n = spearman_ci(grp['rating'], grp['avg_score'])
    snap_results.append({'date': date, 'rho': rho, 'lo': lo, 'hi': hi, 'n': n})

snaps = pd.DataFrame(snap_results).dropna(subset=['rho'])
snaps['year'] = snaps['date'].dt.year

print(f"\n{'Date':<14} {'N':>4}  {'Spearman ρ':>10}  {'95% CI':>20}")
print("-" * 55)
# Print every ~10th snapshot to avoid wall of text
step = max(1, len(snaps) // 30)
for _, row in snaps.iloc[::step].iterrows():
    print(f"{str(row['date'].date()):<14} {int(row['n']):>4}  {row['rho']:>10.3f}"
          f"  [{row['lo']:>6.3f}, {row['hi']:>6.3f}]")

print(f"\n--- Annual summary ---")
print(f"{'Year':<6} {'Snapshots':>9}  {'Mean ρ':>8}  {'Min ρ':>8}  {'Max ρ':>8}  {'Mean N':>7}")
print("-" * 55)
for yr, g in snaps.groupby('year'):
    print(f"{yr:<6} {len(g):>9}  {g['rho'].mean():>8.3f}  "
          f"{g['rho'].min():>8.3f}  {g['rho'].max():>8.3f}  {g['n'].mean():>7.1f}")

# Trend test: is there a significant linear trend in rho over time?
snaps['t'] = (snaps['date'] - snaps['date'].min()).dt.days
slope, intercept, r, p, se = stats.linregress(snaps['t'], snaps['rho'])
print(f"\n--- Trend test (linear regression of ρ on time) ---")
print(f"  Slope: {slope:.6f} per day ({slope*365:.4f} per year)")
print(f"  R²: {r**2:.4f},  p-value: {p:.4f}")
print(f"  Interpretation: ρ {'decreases' if slope < 0 else 'increases'} by "
      f"{abs(slope*365):.3f} per year {'(significant)' if p < 0.05 else '(not significant)'}")

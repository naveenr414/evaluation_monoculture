"""
Temporal analysis of LLM error correlation.
Extends Kim et al. (ICML 2025) cross-sectional finding to a longitudinal analysis.

Test 1: Benchmark profile similarity by release cohort
  - Do models released later have more correlated capability profiles?
  - Measure: pairwise cosine similarity of benchmark score vectors

Test 2: Arena battle outcome variability 2023 vs 2025
  - Are battle outcomes becoming more uniform across prompts?
  - Measure: entropy / variance of win rates per model pair across prompt subsets

Test 3: Within-cohort benchmark correlation structure
  - Is the average inter-model correlation on each benchmark increasing?
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from itertools import combinations
import glob, os

# ── Test 1: Benchmark profile similarity by release cohort ────────────────────
print("=" * 65)
print("TEST 1: Benchmark profile similarity by release cohort")
print("(Do newer models have more similar capability profiles?)")
print("=" * 65)

bench = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/benchmarks.parquet')
bench = bench.dropna(subset=['ifeval','bbh','math','gpqa','musr','mmlu_pro','upload_date'])
bench = bench[~bench['flagged'] & ~bench['merged']]
bench['upload_year'] = bench['upload_date'].dt.year

benchmarks = ['ifeval','bbh','math','gpqa','musr','mmlu_pro']

# Normalise benchmark vectors (z-score across all models)
for b in benchmarks:
    bench[b+'_z'] = (bench[b] - bench[b].mean()) / bench[b].std()

z_cols = [b+'_z' for b in benchmarks]

def mean_pairwise_cosine_sim(vectors, max_pairs=2000, seed=42):
    """Mean pairwise cosine similarity among a set of vectors (sampled)."""
    if len(vectors) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    n = len(vectors)
    all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    if len(all_pairs) > max_pairs:
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs
    sims = []
    for i, j in pairs:
        v1, v2 = vectors[i], vectors[j]
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            sims.append(1 - cosine(v1, v2))
    return np.mean(sims)

def mean_pairwise_spearman(df, cols, max_pairs=2000, seed=42):
    """Mean pairwise Spearman correlation of model score vectors (sampled)."""
    if len(df) < 2:
        return np.nan, 0
    rng = np.random.default_rng(seed)
    n = len(df)
    all_pairs = list(combinations(range(n), 2))
    if len(all_pairs) > max_pairs:
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs
    rhos = []
    vals = df[cols].values.astype(float)
    for i, j in pairs:
        vi, vj = vals[i], vals[j]
        if np.std(vi) > 0 and np.std(vj) > 0:
            r, _ = stats.spearmanr(vi, vj)
            rhos.append(r)
    return np.mean(rhos), len(rhos)

print("\n--- All models by release year ---")
print(f"{'Year':<6}  {'N':>5}  {'Mean pairwise cosine sim':>25}  {'Mean pairwise Spearman ρ':>25}")
print("-" * 65)
for yr in [2022, 2023, 2024, 2025]:
    sub = bench[bench['upload_year'] == yr]
    if len(sub) < 5:
        continue
    vectors = sub[z_cols].values
    cos_sim = mean_pairwise_cosine_sim(vectors)
    sp_rho, n_pairs = mean_pairwise_spearman(sub, z_cols)
    print(f"{yr:<6}  {len(sub):>5}  {cos_sim:>25.4f}  {sp_rho:>25.4f}  (n_pairs={n_pairs})")

# Focus: top-K models by avg_score in each year — most comparable to frontier
print("\n--- Top-50 models per year (frontier proxy) ---")
print(f"{'Year':<6}  {'N':>5}  {'Cosine sim':>12}  {'Spearman ρ':>12}  {'Avg score':>10}")
print("-" * 50)
for yr in [2022, 2023, 2024, 2025]:
    sub = bench[bench['upload_year'] == yr].nlargest(50, 'avg_score')
    if len(sub) < 5:
        continue
    vectors = sub[z_cols].values
    cos_sim = mean_pairwise_cosine_sim(vectors)
    sp_rho, _ = mean_pairwise_spearman(sub, z_cols)
    print(f"{yr:<6}  {len(sub):>5}  {cos_sim:>12.4f}  {sp_rho:>12.4f}  {sub['avg_score'].mean():>10.2f}")

# Correlation by avg_score: do better models have more correlated profiles?
print("\n--- Profile similarity vs. model quality ---")
print("(Replicating Kim et al. cross-sectional finding)")
bench['score_quartile'] = pd.qcut(bench['avg_score'], q=4, labels=['Q1 (weakest)','Q2','Q3','Q4 (best)'])
for q in ['Q1 (weakest)','Q2','Q3','Q4 (best)']:
    sub = bench[bench['score_quartile'] == q]
    if len(sub) < 5:
        continue
    sp_rho, n_pairs = mean_pairwise_spearman(sub, z_cols)
    print(f"  {q}  n={len(sub):>4}  mean pairwise ρ={sp_rho:.4f}  (n_pairs={n_pairs})")

# ── Test 2: Arena battle outcome variability 2023 vs 2025 ─────────────────────
print("\n" + "=" * 65)
print("TEST 2: Battle outcome variability 2023 vs 2025")
print("(Are outcomes becoming more uniform = more correlated behavior?)")
print("=" * 65)

def load_battles_2023():
    path = '/Users/naveenr/Downloads/cyclic_optimization/data/data/train-00000-of-00001-cced8514c7ed782a.parquet'
    df = pd.read_parquet(path)
    df = df[df['winner'].isin(['model_a','model_b'])].copy()
    df['a_wins'] = (df['winner'] == 'model_a').astype(int)
    return df

def load_battles_2025():
    shard_dir = '/Users/naveenr/Downloads/cyclic_optimization/data/140k'
    shards = sorted(glob.glob(os.path.join(shard_dir, '**/*.parquet'), recursive=True))
    dfs = []
    for s in shards:
        df = pd.read_parquet(s, columns=['model_a','model_b','winner'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['winner'].isin(['model_a','model_b'])].copy()
    df['a_wins'] = (df['winner'] == 'model_a').astype(int)
    return df

b23 = load_battles_2023()
b25 = load_battles_2025()

def outcome_variability(battles, min_battles=15):
    """
    For each model pair, compute variance of win rates across random subsamples.
    High variance = outcomes are prompt-dependent (uncorrelated errors).
    Low variance = outcomes are uniform (correlated behavior).
    """
    results = []
    for (ma, mb), grp in battles.groupby(['model_a','model_b']):
        if len(grp) < min_battles:
            continue
        wr = grp['a_wins'].mean()
        # Bootstrap variance of win rate
        rng = np.random.default_rng(42)
        bootstrap_wrs = []
        for _ in range(200):
            sample = grp['a_wins'].sample(min_battles, replace=True, random_state=rng.integers(0,9999))
            bootstrap_wrs.append(sample.mean())
        results.append({
            'model_a': ma, 'model_b': mb,
            'n_battles': len(grp),
            'win_rate': wr,
            'win_rate_var': np.var(bootstrap_wrs),
            'win_rate_std': np.std(bootstrap_wrs),
        })
    return pd.DataFrame(results)

print("\nComputing variability metrics...")
var23 = outcome_variability(b23, min_battles=10)
var25 = outcome_variability(b25, min_battles=20)

print(f"\n2023 ({len(b23):,} battles, {b23['model_a'].nunique()} models):")
print(f"  Model pairs with ≥10 battles: {len(var23)}")
print(f"  Mean win rate (higher model): {var23['win_rate'].apply(lambda x: max(x,1-x)).mean():.3f}")
print(f"  Mean bootstrap std of win rate: {var23['win_rate_std'].mean():.4f}")
print(f"  % pairs with win rate in [0.45,0.55]: {((var23['win_rate']>=0.45)&(var23['win_rate']<=0.55)).mean():.1%}")

print(f"\n2025 ({len(b25):,} battles, {b25['model_a'].nunique()} models):")
print(f"  Model pairs with ≥20 battles: {len(var25)}")
print(f"  Mean win rate (higher model): {var25['win_rate'].apply(lambda x: max(x,1-x)).mean():.3f}")
print(f"  Mean bootstrap std of win rate: {var25['win_rate_std'].mean():.4f}")
print(f"  % pairs with win rate in [0.45,0.55]: {((var25['win_rate']>=0.45)&(var25['win_rate']<=0.55)).mean():.1%}")

# KS test on win rate distributions
ks, p = stats.ks_2samp(
    var23['win_rate'].apply(lambda x: max(x,1-x)),
    var25['win_rate'].apply(lambda x: max(x,1-x))
)
print(f"\nKS test (win rate distributions 2023 vs 2025): stat={ks:.3f}  p={p:.4f}")

# ── Test 3: Model type correlation structure ──────────────────────────────────
print("\n" + "=" * 65)
print("TEST 3: Base vs instruct — who is more correlated?")
print("(Kim et al.: better models more correlated; does instruct > base?)")
print("=" * 65)

for mtype in ['base','instruct','other']:
    sub = bench[bench['model_type'] == mtype]
    if len(sub) < 10:
        continue
    sp_rho, n_pairs = mean_pairwise_spearman(sub.head(100), z_cols)
    print(f"  {mtype:<12}  n={len(sub):>4}  mean pairwise ρ={sp_rho:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY: Connecting to Kim et al. and our C4 findings")
print("=" * 65)

# Rank correlation between avg_score and profile similarity
sub = bench.dropna(subset=['avg_score']+z_cols)
sub = sub.sample(min(200, len(sub)), random_state=42)  # sample for speed
pairs = [(i,j) for i,j in combinations(range(len(sub)), 2)]
pair_sim = []
pair_score_diff = []
for i,j in pairs[:2000]:
    vi = sub.iloc[i][z_cols].values.astype(float)
    vj = sub.iloc[j][z_cols].values.astype(float)
    if np.std(vi) > 0 and np.std(vj) > 0:
        sim = stats.spearmanr(vi, vj)[0]
        avg_q = (sub.iloc[i]['avg_score'] + sub.iloc[j]['avg_score']) / 2
        pair_sim.append(sim)
        pair_score_diff.append(avg_q)

rho_q_sim, p_q_sim = stats.spearmanr(pair_score_diff, pair_sim)
print(f"\nSpearman(avg model quality, profile similarity): rho={rho_q_sim:.3f}  p={p_q_sim:.4f}")
print(f"  → {'Higher quality models have MORE similar profiles' if rho_q_sim > 0 else 'Higher quality models have LESS similar profiles'}")
print(f"  → {'Consistent with Kim et al.' if rho_q_sim > 0 and p_q_sim < 0.05 else 'Inconsistent with Kim et al.'}")

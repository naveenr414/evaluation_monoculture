"""
Dissecting v-inflation vs. genuine convergence.

A. Organization diversity and intra/inter-org ELO compression over time
B. Response length vs. ELO in 2025 battle data (verbosity as v-proxy)
C. Benchmark profile similarity among top matched models
   (RLHF-sensitive benchmarks compress more than RLHF-insensitive ones?)
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import glob, os, ast

arena   = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
matched = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/matched.parquet')
arena['leaderboard_publish_date']   = pd.to_datetime(arena['leaderboard_publish_date'])
matched['leaderboard_publish_date'] = pd.to_datetime(matched['leaderboard_publish_date'])

# ── A. Organization diversity and intra/inter-org compression ─────────────────
print("=" * 65)
print("A. Organization diversity and intra/inter-org ELO compression")
print("=" * 65)

K = 20
results = []
for date, snap in arena.groupby('leaderboard_publish_date'):
    top = snap.nlargest(K, 'rating').reset_index(drop=True)
    if len(top) < K:
        continue
    top['org'] = top['organization'].fillna('unknown').str.lower().str.strip()

    n_orgs = top['org'].nunique()
    # Largest org share (org with most models in top-K)
    org_counts = top['org'].value_counts()
    max_share = org_counts.iloc[0] / K

    # Intra-org ELO pairs (same org)
    intra_gaps, inter_gaps = [], []
    for i, j in combinations(range(K), 2):
        gap = abs(top.loc[i,'rating'] - top.loc[j,'rating'])
        if top.loc[i,'org'] == top.loc[j,'org']:
            intra_gaps.append(gap)
        else:
            inter_gaps.append(gap)

    results.append({
        'date': date,
        'n_orgs': n_orgs,
        'max_org_share': max_share,
        'mean_intra_gap': np.mean(intra_gaps) if intra_gaps else np.nan,
        'mean_inter_gap': np.mean(inter_gaps) if inter_gaps else np.nan,
        'n_intra_pairs': len(intra_gaps),
    })

org_df = pd.DataFrame(results)
org_df['year'] = org_df['date'].dt.year
org_df['t'] = (org_df['date'] - org_df['date'].min()).dt.days

print(f"\n{'Year':<6} {'Snaps':>5}  {'Orgs in top-{K}':>14}  "
      f"{'Max org share':>14}  {'Mean intra gap':>15}  {'Mean inter gap':>15}")
print("-" * 75)
for yr, g in org_df.groupby('year'):
    print(f"{yr:<6} {len(g):>5}  {g['n_orgs'].mean():>14.1f}  "
          f"{g['max_org_share'].mean():>14.1%}  "
          f"{g['mean_intra_gap'].mean():>15.1f}  "
          f"{g['mean_inter_gap'].mean():>15.1f}")

# Trend tests
for col, lbl in [('n_orgs','N orgs'), ('mean_intra_gap','Intra-org gap'),
                 ('mean_inter_gap','Inter-org gap'), ('max_org_share','Max org share')]:
    sub = org_df.dropna(subset=[col])
    sl, _, r, p, _ = stats.linregress(sub['t'], sub[col])
    print(f"\n{lbl} trend: slope={sl*365:.4f}/yr  R²={r**2:.3f}  p={p:.4f}  "
          f"({'significant' if p<0.05 else 'not sig.'})")

# Latest snapshot: who's in top-20?
latest = arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]
top20  = latest.nlargest(20, 'rating')[['model_name','organization','rating']].reset_index(drop=True)
print(f"\nTop-20 at latest snapshot by organization:")
print(top20.groupby('organization')['model_name'].apply(list).to_string())

# Intra vs inter ratio — if intra/inter < 1, same-org models are CLOSER than cross-org
org_df['intra_inter_ratio'] = org_df['mean_intra_gap'] / org_df['mean_inter_gap']
print(f"\nIntra/inter ELO gap ratio (< 1 means same-org models cluster together):")
for yr, g in org_df.groupby('year'):
    r = g['intra_inter_ratio'].mean()
    print(f"  {yr}: {r:.3f}  {'← same-org cluster' if r < 1 else '← cross-org cluster'}")

# ── B. Response length vs. ELO (verbosity as v-proxy) ─────────────────────────
print(f"\n{'='*65}")
print("B. Response length vs. ELO — verbosity as a v-inflation proxy")
print(f"{'='*65}")

shard_dir = '/Users/naveenr/Downloads/cyclic_optimization/data/140k'
shards = sorted(glob.glob(os.path.join(shard_dir, '**/*.parquet'), recursive=True))
b25_list = []
for s in shards:
    df = pd.read_parquet(s, columns=['model_a','model_b','winner',
                                      'conversation_a','conversation_b'])
    b25_list.append(df)
b25 = pd.concat(b25_list, ignore_index=True)

def get_response_len(conv):
    """Extract total assistant response length from conversation."""
    try:
        if isinstance(conv, str):
            turns = ast.literal_eval(conv)
        else:
            turns = conv
        return sum(len(str(t.get('content','')))
                   for t in turns if t.get('role') == 'assistant')
    except:
        return np.nan

print("\nComputing response lengths (this may take a moment)...")
b25['len_a'] = b25['conversation_a'].apply(get_response_len)
b25['len_b'] = b25['conversation_b'].apply(get_response_len)

# Per-model mean response length
len_a = b25.groupby('model_a')['len_a'].mean()
len_b = b25.groupby('model_b')['len_b'].mean()
model_len = pd.concat([len_a.rename('len'), len_b.rename('len')]).groupby(level=0).mean()
model_len = model_len.reset_index().rename(columns={'index':'model','model_a':'model'})
# Fix column name
model_len.columns = ['model','mean_response_len']

# Get Arena ELO for latest snapshot
latest_elo = (arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]
              .set_index('model_name')['rating'])

# Match model names
def norm(s): return str(s).lower().replace('-','').replace('_','').replace('.','').replace(' ','')
elo_norm = {norm(m): v for m, v in latest_elo.items()}

model_len['elo'] = model_len['model'].apply(
    lambda m: elo_norm.get(norm(m))
)
model_len = model_len.dropna(subset=['elo'])

rho, pval = stats.spearmanr(model_len['elo'], model_len['mean_response_len'])
print(f"\nSpearman(ELO, mean response length): ρ={rho:.3f}  p={pval:.4f}  n={len(model_len)}")
print(f"{'significant' if pval < 0.05 else 'not significant'} — "
      f"{'verbosity inflates ELO' if rho > 0 and pval < 0.05 else 'verbosity does not drive ELO'}")

print(f"\nTop models by response length:")
top_len = model_len.merge(
    latest_elo.reset_index().rename(columns={'model_name':'model'}), on='model', how='inner'
).sort_values('mean_response_len', ascending=False)
print(top_len[['model','mean_response_len','rating']].head(15).to_string(index=False))

print(f"\nCorrelation by ELO tier:")
top_len['elo_tier'] = pd.qcut(top_len['rating'], q=3, labels=['bottom','mid','top'])
for tier, g in top_len.groupby('elo_tier'):
    if len(g) < 4: continue
    r, p = stats.spearmanr(g['rating'], g['mean_response_len'])
    print(f"  {tier} third (n={len(g)}): ρ={r:.3f}  p={p:.4f}")

# ── C. Benchmark profile similarity — RLHF-sensitive vs. insensitive ──────────
print(f"\n{'='*65}")
print("C. Benchmark profile: RLHF-sensitive vs. RLHF-insensitive compression")
print(f"{'='*65}\n")

has_bench = matched[matched['avg_score'].notna()].copy()

# RLHF-sensitive: IFEval (directly trained), BBH (often in SFT data)
# RLHF-insensitive: GPQA (hard reasoning, PhD-level, hard to target), MATH (competition)
rlhf_sens   = ['ifeval', 'bbh']
rlhf_insens = ['gpqa',   'math']

first = (has_bench.sort_values('leaderboard_publish_date')
         .groupby('model_name').first().reset_index())
first['cohort'] = pd.cut(
    first['leaderboard_publish_date'],
    bins=pd.to_datetime(['2023-01-01','2024-01-01','2024-07-01',
                         '2025-01-01','2026-01-01']),
    labels=['2023','2024H1','2024H2','2025'],
    right=False
)

print(f"Std of benchmark scores by cohort and benchmark type")
print(f"(if RLHF-sensitive compresses faster → v-inflation signal)\n")
print(f"{'Cohort':<10}  {'N':>3}  {'ifeval std':>10}  {'bbh std':>8}  "
      f"{'gpqa std':>9}  {'math std':>9}  {'IFEval/GPQA ratio':>18}")
print("-" * 75)
for cohort in ['2023','2024H1','2024H2','2025']:
    sub = first[first['cohort'] == cohort].dropna(subset=['ifeval','gpqa','bbh','math'])
    if len(sub) < 4:
        print(f"{cohort:<10}  {len(sub):>3}  too few")
        continue
    ifev_s = sub['ifeval'].std()
    bbh_s  = sub['bbh'].std()
    gpqa_s = sub['gpqa'].std()
    math_s = sub['math'].std()
    ratio  = ifev_s / gpqa_s if gpqa_s > 0 else np.nan
    print(f"{cohort:<10}  {len(sub):>3}  {ifev_s:>10.2f}  {bbh_s:>8.2f}  "
          f"{gpqa_s:>9.2f}  {math_s:>9.2f}  {ratio:>18.2f}")

# Over time: std of RLHF-sensitive vs insensitive among top matched models
print(f"\nAmong top-20 matched models at each snapshot:")
print(f"{'Year':<6}  {'IFEval std':>10}  {'BBH std':>8}  {'GPQA std':>9}  "
      f"{'MATH std':>9}  {'IFEval/GPQA':>12}")
print("-" * 60)
sens_results = []
for date, grp in has_bench.groupby('leaderboard_publish_date'):
    top = grp.nlargest(20, 'rating')
    if len(top) < 10: continue
    sens_results.append({
        'date': date,
        'ifeval_std': top['ifeval'].std(),
        'bbh_std':    top['bbh'].std(),
        'gpqa_std':   top['gpqa'].std(),
        'math_std':   top['math'].std(),
    })

sdf = pd.DataFrame(sens_results)
sdf['year'] = sdf['date'].dt.year
sdf['ratio'] = sdf['ifeval_std'] / sdf['gpqa_std']
for yr, g in sdf.groupby('year'):
    print(f"{yr:<6}  {g['ifeval_std'].mean():>10.2f}  {g['bbh_std'].mean():>8.2f}  "
          f"{g['gpqa_std'].mean():>9.2f}  {g['math_std'].mean():>9.2f}  "
          f"{g['ratio'].mean():>12.2f}")

sdf['t'] = (sdf['date'] - sdf['date'].min()).dt.days
for col, lbl in [('ifeval_std','IFEval std (RLHF-sensitive)'),
                 ('gpqa_std','GPQA std (RLHF-insensitive)'),
                 ('math_std','MATH std (RLHF-insensitive)')]:
    sl, _, r, p, _ = stats.linregress(sdf['t'], sdf[col])
    print(f"\n{lbl}: slope={sl*365:.4f}/yr  R²={r**2:.3f}  p={p:.4f}  "
          f"({'sig.' if p<0.05 else 'not sig.'})")

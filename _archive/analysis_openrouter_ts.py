"""
OpenRouter time-series analysis:
1. Convergence: is token usage concentrating toward fewer models over time?
2. Inversions: are Pareto-dominated models (higher price, lower Arena ELO) still getting traffic?
3. Arena ELO vs actual usage rank — does Arena predict what developers actually use?
"""

import pandas as pd
import numpy as np
import re
import requests
from scipy import stats

# Load data
df = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/openrouter_daily.parquet')
arena = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
arena['leaderboard_publish_date'] = pd.to_datetime(arena['leaderboard_publish_date'])

# Get Arena ELO at latest snapshot + pricing
latest_arena = arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]
arena_elo = latest_arena.set_index('model_name')['rating'].to_dict()

# Fetch current pricing
r = requests.get('https://openrouter.ai/api/v1/models', timeout=15)
price_data = {m['id']: float(m['pricing'].get('prompt', 0) or 0) * 1e6
              for m in r.json()['data']}

def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())

# Match OpenRouter model slugs to Arena model names
def match_to_arena(slug):
    short = slug.split('/')[-1] if '/' in slug else slug
    kn = norm(short)
    for an in arena_elo:
        if len(kn) >= 5 and (kn in norm(an) or norm(an) in kn):
            return an
    return None

slug_to_arena = {s: match_to_arena(s) for s in df['model_permaslug'].unique()}
df['arena_model'] = df['model_permaslug'].map(slug_to_arena)
df['arena_elo'] = df['arena_model'].map(arena_elo)
df['input_price'] = df['model_permaslug'].map(price_data)
df['month'] = df['date'].dt.to_period('M')

matched_df = df[df['arena_elo'].notna() & df['input_price'].notna() & (df['input_price'] > 0)]
print(f"Rows with Arena ELO + pricing: {len(matched_df)}")
print(f"Unique matched models: {matched_df['model_permaslug'].nunique()}")

# ── 1. CONVERGENCE: is usage concentrating? ───────────────────────────────────
print("\n" + "=" * 65)
print("1. CONVERGENCE: is OpenRouter usage concentrating over time?")
print("=" * 65)

# Monthly: top-1 share, top-5 share, HHI
monthly = []
for month, grp in df.groupby('month'):
    total = grp['total_tokens'].sum()
    sorted_tok = grp.groupby('model_permaslug')['total_tokens'].sum().sort_values(ascending=False)
    top1  = sorted_tok.iloc[0] / total if len(sorted_tok) > 0 else np.nan
    top5  = sorted_tok.head(5).sum() / total if len(sorted_tok) >= 5 else np.nan
    top10 = sorted_tok.head(10).sum() / total if len(sorted_tok) >= 10 else np.nan
    hhi   = ((sorted_tok / total) ** 2).sum()
    n_models = (sorted_tok > 0).sum()
    monthly.append({'month': month, 'top1_share': top1, 'top5_share': top5,
                    'top10_share': top10, 'hhi': hhi, 'n_models': n_models,
                    'total_tokens': total})

mdf = pd.DataFrame(monthly).sort_values('month')
mdf['t'] = range(len(mdf))

print(f"\n{'Month':<10}  {'Top-1%':>8}  {'Top-5%':>8}  {'Top-10%':>9}  {'HHI':>8}  {'N models':>9}")
print("-" * 60)
for _, row in mdf.iterrows():
    print(f"{str(row['month']):<10}  {row['top1_share']:>8.1%}  {row['top5_share']:>8.1%}  "
          f"{row['top10_share']:>9.1%}  {row['hhi']:>8.4f}  {int(row['n_models']):>9}")

for col, lbl in [('top1_share','Top-1 share'), ('hhi','HHI'), ('n_models','N models')]:
    sl, _, r, p, _ = stats.linregress(mdf['t'], mdf[col])
    print(f"\n{lbl} trend: slope={sl*12:.4f}/yr  p={p:.4f}  "
          f"({'sig.' if p<0.05 else 'not sig.'}) {'↑' if sl>0 else '↓'}")

# ── 2. INVERSIONS: do Pareto-dominated models still get traffic? ───────────────
print("\n" + "=" * 65)
print("2. INVERSIONS: Pareto-dominated models vs their dominators")
print("=" * 65)

# Aggregate total tokens per model across all time
model_totals = df.groupby('model_permaslug')['total_tokens'].sum().sort_values(ascending=False)
model_totals_arena = matched_df.groupby('arena_model')['total_tokens'].sum()

# Get per-model data: Arena ELO + price + usage
model_data = []
for slug, arena_model in slug_to_arena.items():
    if arena_model is None: continue
    price = price_data.get(slug, 0)
    if price == 0: continue
    elo = arena_elo.get(arena_model, np.nan)
    tokens = model_totals.get(slug, 0)
    if np.isnan(elo) or tokens == 0: continue
    model_data.append({'slug': slug, 'arena_model': arena_model,
                       'arena_elo': elo, 'price': price, 'total_tokens': tokens})

mdata = pd.DataFrame(model_data).sort_values('arena_elo', ascending=False)
mdata = mdata.drop_duplicates(subset='arena_model')

# Find dominated pairs: model_j has higher ELO AND lower price than model_i
inversions = []
for i, ri in mdata.iterrows():
    for j, rj in mdata.iterrows():
        if i == j: continue
        # rj dominates ri (rj is better AND cheaper)
        if rj['arena_elo'] > ri['arena_elo'] + 10 and rj['price'] < ri['price'] * 0.5:
            inversions.append({
                'dominated_model': ri['arena_model'],
                'dominated_elo': round(ri['arena_elo'], 1),
                'dominated_price': ri['price'],
                'dominated_tokens_B': ri['total_tokens'] / 1e9,
                'dominator_model': rj['arena_model'],
                'dominator_elo': round(rj['arena_elo'], 1),
                'dominator_price': rj['price'],
                'dominator_tokens_B': rj['total_tokens'] / 1e9,
                'price_ratio': round(ri['price'] / rj['price'], 1),
                'usage_ratio': ri['total_tokens'] / rj['total_tokens'],
            })

inv_df = pd.DataFrame(inversions)
# Filter to most dramatic: dominated model still heavily used despite having a dominator
if len(inv_df):
    inv_df = inv_df[inv_df['usage_ratio'] > 0.1].sort_values('dominated_tokens_B', ascending=False)
    inv_df = inv_df.drop_duplicates(subset='dominated_model')
    print(f"\n{'Dominated model':<40} {'ELO':>6}  {'$/1M':>6}  {'Tokens(B)':>10}  "
          f"{'Dominator':<35} {'ELO':>6}  {'$/1M':>6}  {'Usage ratio':>12}")
    print("-" * 130)
    for _, row in inv_df.head(15).iterrows():
        print(f"{row['dominated_model']:<40} {row['dominated_elo']:>6.0f}  "
              f"{row['dominated_price']:>6.2f}  {row['dominated_tokens_B']:>10.1f}  "
              f"{row['dominator_model']:<35} {row['dominator_elo']:>6.0f}  "
              f"{row['dominator_price']:>6.2f}  {row['usage_ratio']:>12.2f}x")

# ── 3. ARENA ELO vs actual usage rank ─────────────────────────────────────────
print("\n" + "=" * 65)
print("3. Does Arena ELO predict actual developer usage?")
print("=" * 65)

mdata['usage_rank'] = mdata['total_tokens'].rank(ascending=False)
mdata['elo_rank']   = mdata['arena_elo'].rank(ascending=False)
rho, p = stats.spearmanr(mdata['arena_elo'], mdata['total_tokens'])
print(f"\nSpearman(Arena ELO, total tokens): rho={rho:.3f}  p={p:.4f}  n={len(mdata)}")

# Controlling for price
# Partial correlation: Arena ELO vs usage controlling for price
from scipy.stats import spearmanr
def partial_spearman(x, y, z):
    """Partial Spearman correlation of x and y controlling for z."""
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    rz = pd.Series(z).rank()
    res_x = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    res_y = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(res_x, res_y)

rho_partial, p_partial = partial_spearman(
    mdata['arena_elo'], mdata['total_tokens'], np.log(mdata['price'])
)
print(f"Partial Spearman(Arena ELO, tokens | price): rho={rho_partial:.3f}  p={p_partial:.4f}")
print(f"  → {'Arena ELO predicts usage beyond price' if rho_partial > 0.2 and p_partial < 0.05 else 'Arena ELO does NOT predict usage after controlling for price'}")

# Top usage models vs Arena ELO
print(f"\nTop-20 models by OpenRouter usage (Jan 2025 – Jun 2026):")
print(f"{'Model':<45} {'Tokens(B)':>10}  {'$/1M':>6}  {'Arena ELO':>10}  {'ELO rank':>9}")
print("-" * 85)
for _, row in mdata.nlargest(20, 'total_tokens').iterrows():
    print(f"{row['arena_model']:<45} {row['total_tokens']/1e9:>10.1f}  "
          f"{row['price']:>6.2f}  {row['arena_elo']:>10.1f}  {row['elo_rank']:>9.0f}")

# Monthly trend: does Arena ELO-usage correlation change over time?
print(f"\nMonthly Spearman(Arena ELO, daily tokens) trend:")
monthly_rho = []
for month, grp in matched_df.groupby('month'):
    day_totals = grp.groupby('arena_model').agg(
        tokens=('total_tokens','sum'), elo=('arena_elo','first'), price=('input_price','first')
    ).dropna()
    if len(day_totals) < 10: continue
    r, p = stats.spearmanr(day_totals['elo'], day_totals['tokens'])
    monthly_rho.append({'month': month, 'rho': r, 'n': len(day_totals)})

rho_df = pd.DataFrame(monthly_rho)
rho_df['t'] = range(len(rho_df))
print(f"\n{'Month':<10}  {'rho(ELO, usage)':>16}  {'n':>4}")
print("-" * 35)
for _, row in rho_df.iterrows():
    print(f"{str(row['month']):<10}  {row['rho']:>16.3f}  {int(row['n']):>4}")

sl, _, r2, p2, _ = stats.linregress(rho_df['t'], rho_df['rho'])
print(f"\nTrend: slope={sl*12:.4f}/yr  R²={r2**2:.3f}  p={p2:.4f}  "
      f"({'ELO predicting usage less over time' if sl<0 else 'ELO predicting usage more over time'})")

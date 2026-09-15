import pandas as pd, numpy as np, requests, re
from scipy import stats

r = requests.get('https://openrouter.ai/api/v1/models', timeout=15)
data = r.json()['data']
arena = pd.read_parquet('data/arena_leaderboard.parquet')
latest = arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]

def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
arena_norm = {norm(m): (m, row['rating']) for m, row in latest.set_index('model_name').iterrows()}

models = []
for m in data:
    pricing = m.get('pricing', {})
    inp = float(pricing.get('prompt', 0) or 0) * 1e6
    if inp == 0:
        continue
    model_id = m.get('id', '')
    short = model_id.split('/')[-1] if '/' in model_id else model_id
    kn = norm(short)
    match = None
    for an, (am, aelo) in arena_norm.items():
        if len(kn) >= 5 and (kn in an or an in kn):
            match = (am, aelo)
            break
    if not match:
        continue
    models.append({
        'id': model_id, 'name': m.get('name', ''),
        'arena_model': match[0], 'arena_elo': match[1],
        'input_price': inp,
        'out_price': float(pricing.get('completion', 0) or 0) * 1e6,
        'org': model_id.split('/')[0] if '/' in model_id else 'unknown'
    })

df = pd.DataFrame(models).drop_duplicates(subset=['arena_model', 'input_price'])
print(f'Matched models: {len(df)}')

# ── Pareto dominance ──────────────────────────────────────────────────────────
rows = []
for i, ri in df.iterrows():
    for j, rj in df.iterrows():
        if i == j:
            continue
        if ri['arena_elo'] > rj['arena_elo'] + 5 and ri['input_price'] < rj['input_price'] * 0.5:
            rows.append({
                'dominated': rj['name'], 'd_elo': round(rj['arena_elo'], 1),
                'd_price': rj['input_price'],
                'dominates': ri['name'], 'dom_elo': round(ri['arena_elo'], 1),
                'dom_price': ri['input_price'],
                'elo_diff': round(ri['arena_elo'] - rj['arena_elo'], 1),
                'price_ratio': round(rj['input_price'] / ri['input_price'], 1)
            })

dom_df = pd.DataFrame(rows).sort_values('price_ratio', ascending=False)
print('\n=== PARETO DOMINATED MODELS (beaten on BOTH ELO and price) ===')
print(dom_df.head(15).to_string(index=False))

# ── ELO spread within price buckets ──────────────────────────────────────────
print('\n=== ELO spread within price buckets (top 80 models) ===')
top80 = df.nlargest(80, 'arena_elo').copy()
top80['price_bucket'] = pd.cut(
    top80['input_price'],
    bins=[0, 0.5, 2, 10, 200],
    labels=['<$0.50', '$0.50-$2', '$2-$10', '>$10']
)
print(top80.groupby('price_bucket', observed=True)['arena_elo'].agg(
    ['mean', 'std', 'min', 'max', 'count']).round(1))

# ── Statistically tied with #1 ────────────────────────────────────────────────
print()
top_elo = df['arena_elo'].max()
ci_half = 25
tied = df[df['arena_elo'] >= top_elo - ci_half].sort_values('arena_elo', ascending=False)
print(f'=== Statistically tied with #1 (within {ci_half} ELO): {len(tied)} models ===')
print(tied[['name', 'arena_elo', 'input_price']].to_string(index=False))
cheap = tied.nsmallest(1, 'input_price').iloc[0]
exp   = tied.nlargest(1, 'input_price').iloc[0]
print(f'\nCheapest: {cheap["name"]}  ${cheap["input_price"]:.3f}/1M')
print(f'Priciest: {exp["name"]}  ${exp["input_price"]:.1f}/1M')
print(f'Price ratio within tied band: {exp["input_price"]/cheap["input_price"]:.0f}x')

# ── Overall ELO-price correlation ─────────────────────────────────────────────
rho, p = stats.spearmanr(df['arena_elo'], df['input_price'])
print(f'\nSpearman(Arena ELO, price): rho={rho:.3f}  p={p:.4f}')
print('Moderate correlation — price explains some but not all of ELO signal')

# ── Notable inversions ────────────────────────────────────────────────────────
print('\n=== Notable inversions: high Arena ELO, very low price ===')
df['elo_rank'] = df['arena_elo'].rank(ascending=False)
df['price_rank'] = df['input_price'].rank(ascending=True)  # rank 1 = cheapest
df['value_gap'] = df['price_rank'] - df['elo_rank']  # positive = cheaper than ELO suggests
print(df.nlargest(15, 'value_gap')[
    ['name', 'arena_elo', 'input_price', 'elo_rank', 'price_rank', 'value_gap']
].round(1).to_string(index=False))

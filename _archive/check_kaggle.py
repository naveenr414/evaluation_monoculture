import pandas as pd
import re

base = '/Users/naveenr/Downloads/cyclic_optimization/data/kaggle_llm_tracker'
frontier = pd.read_csv(f'{base}/frontier_models1.csv')
arena = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
bench = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/benchmarks.parquet')

frontier['release_date'] = pd.to_datetime(frontier['release_date'], errors='coerce')
latest = arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]
arena_elo_map = latest.set_index('model_name')['rating'].to_dict()

def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
arena_norm = {norm(m): m for m in latest['model_name']}

# Flag 1: ELO sanity check
print('=== Sanity check: Kaggle ELO vs Arena leaderboard ===')
has_elo = frontier[frontier['arena_elo'].notna()][['model_name', 'arena_elo', 'release_date']]
rows = []
for _, row in has_elo.iterrows():
    kn = norm(row['model_name'])
    match = None
    for an, am in arena_norm.items():
        if kn in an or an in kn:
            match = am
            break
    arena_elo = arena_elo_map.get(match) if match else None
    rows.append({
        'kaggle_model': row['model_name'],
        'kaggle_elo': row['arena_elo'],
        'arena_match': match or 'no match',
        'arena_elo': round(arena_elo, 1) if arena_elo else 'no match',
        'diff': round(row['arena_elo'] - arena_elo, 1) if arena_elo else None
    })

check = pd.DataFrame(rows)
matched = check[check['arena_elo'] != 'no match'].copy()
print(matched[['kaggle_model', 'kaggle_elo', 'arena_match', 'arena_elo', 'diff']].to_string(index=False))

print()
matched_diff = matched['diff'].dropna().astype(float)
print(f'Matched: {len(matched)} models')
print(f'Mean ELO diff (Kaggle - Arena): {matched_diff.mean():.1f}')
print(f'Max abs diff: {matched_diff.abs().max():.1f}')

# Flag 2: Future models
print()
print('=== Models with release_date after June 2026 ===')
future = frontier[frontier['release_date'] > pd.Timestamp('2026-06-04')]
if len(future) == 0:
    print('  None')
else:
    print(future[['model_name', 'release_date', 'arena_elo']].to_string(index=False))

print()
print('=== Models released 2025-2026 (potentially speculative) ===')
recent = frontier[frontier['release_date'] >= pd.Timestamp('2025-01-01')].sort_values('release_date')
print(recent[['model_name', 'release_date', 'arena_elo', 'benchmark_source']].to_string(index=False))

# Flag 3: GPQA scale comparison
print()
print('=== GPQA scale comparison ===')
print(f'  OLLMv2 GPQA : min={bench["gpqa"].min():.1f}  max={bench["gpqa"].max():.1f}  mean={bench["gpqa"].mean():.1f}')
print(f'  Kaggle GPQA : min={frontier["gpqa_score"].min():.1f}  max={frontier["gpqa_score"].max():.1f}  mean={frontier["gpqa_score"].dropna().mean():.1f}')

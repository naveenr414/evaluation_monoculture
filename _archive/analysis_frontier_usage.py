import pandas as pd, numpy as np, requests, re
from scipy import stats

df = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/openrouter_daily.parquet')
arena = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
latest = arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]
arena_elo = latest.set_index('model_name')['rating'].to_dict()

r = requests.get('https://openrouter.ai/api/v1/models', timeout=15)
price_data = {m['id']: float(m['pricing'].get('prompt',0) or 0)*1e6 for m in r.json()['data']}

def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
arena_norm = {norm(m):(m,v) for m,v in arena_elo.items()}

def match(slug):
    short = slug.split('/')[-1] if '/' in slug else slug
    kn = norm(short)
    for an,(am,ae) in arena_norm.items():
        if len(kn)>=5 and (kn in an or an in kn):
            return am, ae
    return None, None

rows = []
for slug in df['model_permaslug'].unique():
    if slug == 'other': continue
    am, ae = match(slug)
    if not am or not ae: continue
    price = price_data.get(slug, 0)
    if price == 0: continue
    tok = df[df['model_permaslug']==slug]['total_tokens'].sum()
    first = df[df['model_permaslug']==slug]['date'].min()
    days = (pd.Timestamp('2026-06-05') - first).days + 1
    rows.append({'slug':slug,'model':am,'elo':ae,'price':price,
                 'total_tok':tok,'daily_tok':tok/days})

mdata = pd.DataFrame(rows).drop_duplicates(subset='model').sort_values('elo',ascending=False)
print(f'Total matched: {len(mdata)}, ELO range: {mdata["elo"].min():.0f}-{mdata["elo"].max():.0f}\n')

# What's in each price tier
for tier_name, lo, hi in [('cheap (<$0.30)',0,0.30),('budget ($0.30-$1)',0.30,1.0),
                           ('mid ($1-$5)',1.0,5.0),('premium (>$5)',5.0,999)]:
    sub = mdata[(mdata['price']>=lo)&(mdata['price']<hi)].sort_values('elo',ascending=False)
    if len(sub) < 2: continue
    rng = sub['elo'].max() - sub['elo'].min()
    print(f'--- {tier_name}: n={len(sub)}, ELO {sub["elo"].min():.0f}-{sub["elo"].max():.0f} (range={rng:.0f}) ---')
    print(sub[['model','elo','price','daily_tok']].to_string(index=False))
    print()

# Core test: top-K ELO correlation
print('='*60)
print('Spearman(ELO, daily_tok) within top-K by ELO')
print('='*60)
print(f'\n{"Top-K":<8}  {"ELO range":>10}  {"ELO std":>8}  {"rho":>7}  {"p":>8}')
print('-'*50)
for K in [5, 10, 15, 20, 30, len(mdata)]:
    sub = mdata.head(K)
    elo_range = sub['elo'].max()-sub['elo'].min()
    elo_std = sub['elo'].std()
    if len(sub) < 4: continue
    rho, p = stats.spearmanr(sub['elo'], sub['daily_tok'])
    sig = '*' if p<0.05 else ''
    print(f'Top-{K:<5}  {elo_range:>10.1f}  {elo_std:>8.1f}  {rho:>7.3f}  {p:>8.4f}{sig}')

# Explicit reconciliation
mid = mdata[(mdata['price']>=1)&(mdata['price']<5)]
print(f'\nReconciliation:')
print(f'  C4 top-20 Arena ELO std (2026):     ~17 pts  (within 17 pts of each other)')
print(f'  Mid-tier ($1-5) ELO range:          {mid["elo"].max()-mid["elo"].min():.0f} pts')
print(f'  Mid-tier ELO span:                  {mid["elo"].min():.0f} to {mid["elo"].max():.0f}')
print(f'  -> Mid tier is NOT the compressed zone; it includes older/cheaper frontier models')
print()

# What IS the compressed zone? Top-15 matched models
top15 = mdata.head(15)
print(f'Top-15 matched models (the C4-compressed zone):')
print(top15[['model','elo','price','daily_tok']].to_string(index=False))
rho15, p15 = stats.spearmanr(top15['elo'], top15['daily_tok'])
print(f'\nSpearman(ELO, daily_tok) within top-15: rho={rho15:.3f}  p={p15:.4f}')
print('ELO std:', round(top15['elo'].std(),1), 'ELO range:', round(top15['elo'].max()-top15['elo'].min(),1))
sig15 = 'SIGNIFICANT' if p15<0.05 else 'NOT significant'
print(f'-> {sig15}')
print()

# Final clean table: how does correlation decay as we include more of the frontier
print('Summary: ELO-usage rho vs. how far down the leaderboard we include')
print(f'{"Models included":<30}  {"ELO range":>10}  {"ELO std":>8}  {"rho":>8}')
print('-'*62)
cutoffs = [(5,'Top 5'),(10,'Top 10'),(15,'Top 15 (C4 zone)'),
           (20,'Top 20'),(30,'Top 30'),(len(mdata),f'All {len(mdata)}')]
for K, label in cutoffs:
    sub = mdata.head(K)
    if len(sub)<4: continue
    rho, p = stats.spearmanr(sub['elo'], sub['daily_tok'])
    sig = '*' if p<0.05 else ''
    print(f'{label:<30}  {sub["elo"].max()-sub["elo"].min():>10.1f}  '
          f'{sub["elo"].std():>8.1f}  {rho:>8.3f}{sig}')

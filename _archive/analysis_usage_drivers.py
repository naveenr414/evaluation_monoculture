"""
Decompose what drives OpenRouter usage beyond ELO and price.
Approach: OLS regression of log(daily_tokens) on measured features,
then variance decomposition via incremental R² and residual analysis.

Features:
  - Arena ELO (quality signal)
  - log(price) (cost)
  - Brand dummies (org fixed effects — trust, reliability, ecosystem)
  - Days in window (incumbency / switching costs)
  - Context window size
  - Free tier indicator

Goal: quantify what fraction of usage variance each factor explains,
and how much is genuinely unexplained.
"""

import pandas as pd, numpy as np, requests, re
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('data/openrouter_daily.parquet')
arena = pd.read_parquet('data/arena_leaderboard.parquet')
latest = arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]
arena_elo = latest.set_index('model_name')['rating'].to_dict()

r = requests.get('https://openrouter.ai/api/v1/models', timeout=15)
model_meta = {}
for m in r.json()['data']:
    model_meta[m['id']] = {
        'price':   float(m['pricing'].get('prompt', 0) or 0) * 1e6,
        'created': pd.to_datetime(m.get('created'), unit='s', errors='coerce'),
        'ctx':     m.get('context_length', 0) or 0,
        'org':     m['id'].split('/')[0] if '/' in m['id'] else 'unknown',
        'name':    m.get('name', ''),
    }

def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
arena_norm = {norm(m):(m,v) for m,v in arena_elo.items()}
def match(slug):
    short = slug.split('/')[-1] if '/' in slug else slug
    kn = norm(short)
    for an,(am,ae) in arena_norm.items():
        if len(kn)>=5 and (kn in an or an in kn): return am,ae
    return None, None

rows = []
for slug in df['model_permaslug'].unique():
    if slug == 'other': continue
    am, ae = match(slug)
    if not am or not ae: continue
    meta = model_meta.get(slug, {})
    price = meta.get('price', 0)
    if price == 0: continue
    tok = df[df['model_permaslug']==slug]['total_tokens'].sum()
    first = df[df['model_permaslug']==slug]['date'].min()
    days = (pd.Timestamp('2026-06-05') - first).days + 1
    rows.append({
        'slug': slug, 'model': am, 'elo': ae, 'price': price,
        'org':  meta.get('org','unknown'),
        'ctx':  meta.get('ctx', 0),
        'days_in_window': days,
        'total_tok': tok, 'daily_tok': tok / days,
    })

mdata = pd.DataFrame(rows).drop_duplicates(subset='model')
mdata['log_daily'] = np.log10(mdata['daily_tok'].clip(lower=1))
mdata['log_price'] = np.log10(mdata['price'])
mdata['log_ctx']   = np.log10(mdata['ctx'].clip(lower=1))
mdata['log_days']  = np.log10(mdata['days_in_window'].clip(lower=1))
mdata['elo_z']     = (mdata['elo'] - mdata['elo'].mean()) / mdata['elo'].std()

# Collapse small orgs to 'other'
org_counts = mdata['org'].value_counts()
major_orgs = org_counts[org_counts >= 3].index.tolist()
mdata['org_group'] = mdata['org'].apply(lambda x: x if x in major_orgs else 'other')

print(f"Dataset: {len(mdata)} models\n")
print("Organisation breakdown:")
print(mdata['org_group'].value_counts().to_string())

# ── 1. Incremental R² decomposition ──────────────────────────────────────────
y = mdata['log_daily'].values

def r2(X_cols, extra=None):
    X = mdata[X_cols].values if isinstance(X_cols, list) else mdata[[X_cols]].values
    if extra is not None:
        X = np.hstack([X, extra])
    reg = LinearRegression().fit(X, y)
    return reg.score(X, y), reg

# Brand dummies
brand_dummies = pd.get_dummies(mdata['org_group'], drop_first=True).values

models_seq = [
    ('Null (mean only)',          [],               None),
    ('+ Arena ELO',               ['elo_z'],        None),
    ('+ log(price)',              ['elo_z','log_price'], None),
    ('+ incumbency (log days)',   ['elo_z','log_price','log_days'], None),
    ('+ context window',         ['elo_z','log_price','log_days','log_ctx'], None),
    ('+ brand (org dummies)',     ['elo_z','log_price','log_days','log_ctx'], brand_dummies),
]

prev_r2 = 0
print(f"\n{'Model':<40}  {'R²':>6}  {'ΔR²':>6}  {'Variance explained':>20}")
print("-" * 78)
for label, cols, extra in models_seq:
    if not cols and extra is None:
        r2_val = 0.0
    else:
        r2_val, _ = r2(cols if cols else ['elo_z'], extra) if cols else (0.0, None)
        if not cols and extra is None:
            r2_val = 0.0
    delta = r2_val - prev_r2
    bar = '█' * int(r2_val * 30)
    print(f"{label:<40}  {r2_val:>6.3f}  {delta:>+6.3f}  {bar}")
    prev_r2 = r2_val

# Full model
r2_full, reg_full = r2(['elo_z','log_price','log_days','log_ctx'], brand_dummies)
print(f"\n{'Unexplained (residual)':<40}  {'':>6}  {1-r2_full:>+6.3f}  "
      f"{'░'*int((1-r2_full)*30)}")
print(f"\nTotal explained: {r2_full:.1%}   Unexplained: {1-r2_full:.1%}")

# ── 2. Individual feature importance (permutation) ───────────────────────────
print("\n" + "=" * 65)
print("2. Feature importance (drop-one R² loss)")
print("=" * 65)

feature_sets = {
    'Arena ELO':        (['log_price','log_days','log_ctx'], brand_dummies),
    'Price':            (['elo_z','log_days','log_ctx'],     brand_dummies),
    'Incumbency':       (['elo_z','log_price','log_ctx'],    brand_dummies),
    'Context window':   (['elo_z','log_price','log_days'],   brand_dummies),
    'Brand':            (['elo_z','log_price','log_days','log_ctx'], None),
}
print(f"\n{'Feature dropped':<20}  {'R² without':>12}  {'R² loss':>10}  {'Importance':>12}")
print("-" * 58)
for feat, (cols, extra) in feature_sets.items():
    r2_without, _ = r2(cols, extra)
    loss = r2_full - r2_without
    bar = '█' * max(0, int(loss * 100))
    print(f"{feat:<20}  {r2_without:>12.3f}  {loss:>10.3f}  {bar}")

# ── 3. Brand premium: per-org fixed effects ───────────────────────────────────
print("\n" + "=" * 65)
print("3. Brand premium — per-org fixed effects (vs. 'other' baseline)")
print("=" * 65)

# Simple: mean residual after controlling for ELO + price + incumbency
r2_base, reg_base = r2(['elo_z','log_price','log_days'], None)
mdata['resid'] = y - reg_base.predict(mdata[['elo_z','log_price','log_days']].values)
print(f"\nBaseline R² (ELO + price + incumbency): {r2_base:.3f}")
print(f"Mean residual by organisation (positive = gets MORE usage than predicted):\n")
org_resid = mdata.groupby('org_group')['resid'].agg(['mean','count']).sort_values('mean',ascending=False)
print(f"{'Org':<20}  {'Mean residual':>14}  {'N':>4}  {'Interpretation'}")
print("-" * 70)
for org, row in org_resid.iterrows():
    direction = 'over-predicted' if row['mean'] < -0.2 else \
                'under-predicted (brand premium)' if row['mean'] > 0.2 else 'well-predicted'
    print(f"{org:<20}  {row['mean']:>14.3f}  {int(row['count']):>4}  {direction}")

# ── 4. Most over/under-explained individual models ───────────────────────────
print("\n" + "=" * 65)
print("4. Individual model residuals (largest surprises)")
print("=" * 65)

mdata['predicted_log_daily'] = reg_base.predict(mdata[['elo_z','log_price','log_days']].values)
mdata['resid_abs'] = mdata['resid'].abs()

print("\nModels getting FAR MORE usage than ELO+price+incumbency predicts:")
over = mdata.nlargest(10, 'resid')[['model','elo','price','log_days','daily_tok','resid']]
over['predicted_daily'] = 10**mdata.loc[over.index,'predicted_log_daily']
over['actual_vs_pred'] = over['daily_tok'] / over['predicted_daily']
print(over[['model','elo','price','actual_vs_pred','resid']].to_string(index=False))

print("\nModels getting FAR LESS usage than ELO+price+incumbency predicts:")
under = mdata.nsmallest(10, 'resid')[['model','elo','price','log_days','daily_tok','resid']]
under['predicted_daily'] = 10**mdata.loc[under.index,'predicted_log_daily']
under['actual_vs_pred'] = under['daily_tok'] / under['predicted_daily']
print(under[['model','elo','price','actual_vs_pred','resid']].to_string(index=False))
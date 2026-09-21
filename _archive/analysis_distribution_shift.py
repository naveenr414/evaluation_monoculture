"""
Distribution shift analysis for OpenRouter findings.
Tests:
A. Age-normalized: within same release-cohort, does ELO predict usage?
B. Within-provider: does ELO predict usage controlling for brand?
C. Within-price-tier: does ELO add signal within tiers?
D. Event study: is the ELO-usage correlation decline step-function or gradual?
E. Incumbent effect: how much of brand premium disappears when controlling for age?
"""

import pandas as pd
import numpy as np
import requests, re
from scipy import stats

# Load data
df = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/openrouter_daily.parquet')
arena = pd.read_parquet('/Users/naveenr/Downloads/cyclic_optimization/data/arena_leaderboard.parquet')
arena['leaderboard_publish_date'] = pd.to_datetime(arena['leaderboard_publish_date'])
latest_arena = arena[arena['leaderboard_publish_date'] == arena['leaderboard_publish_date'].max()]
arena_elo = latest_arena.set_index('model_name')['rating'].to_dict()

# Fetch model metadata (pricing + created date)
r = requests.get('https://openrouter.ai/api/v1/models', timeout=15)
model_meta = {}
for m in r.json()['data']:
    model_meta[m['id']] = {
        'price': float(m['pricing'].get('prompt', 0) or 0) * 1e6,
        'created': pd.to_datetime(m.get('created'), unit='s', errors='coerce'),
        'name': m.get('name', ''),
        'org': m['id'].split('/')[0] if '/' in m['id'] else 'unknown',
    }

def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
arena_norm = {norm(m): (m, v) for m, v in arena_elo.items()}

def match_arena(slug):
    short = slug.split('/')[-1] if '/' in slug else slug
    kn = norm(short)
    for an, (am, aelo) in arena_norm.items():
        if len(kn) >= 5 and (kn in an or an in kn):
            return am, aelo
    return None, None

# Build model-level dataset
model_rows = []
for slug in df['model_permaslug'].unique():
    if slug == 'other': continue
    am, aelo = match_arena(slug)
    meta = model_meta.get(slug, {})
    price = meta.get('price', 0)
    created = meta.get('created')
    org = meta.get('org', 'unknown')
    total_tok = df[df['model_permaslug'] == slug]['total_tokens'].sum()
    # Days model was present in our data window (Jan 2025 - Jun 2026 = 521 days)
    first_in_data = df[df['model_permaslug'] == slug]['date'].min()
    days_in_window = (pd.Timestamp('2026-06-05') - first_in_data).days + 1
    avg_daily_tok = total_tok / days_in_window if days_in_window > 0 else 0
    model_rows.append({
        'slug': slug, 'arena_model': am, 'arena_elo': aelo,
        'price': price, 'created': created, 'org': org,
        'total_tokens': total_tok,
        'days_in_window': days_in_window,
        'avg_daily_tokens': avg_daily_tok,
    })

mdata = pd.DataFrame(model_rows)
mdata = mdata[mdata['arena_elo'].notna() & mdata['price'].notna() & (mdata['price'] > 0)]
mdata = mdata.drop_duplicates(subset='arena_model')
mdata['log_price'] = np.log10(mdata['price'])
mdata['log_tokens'] = np.log10(mdata['total_tokens'].clip(lower=1))
mdata['log_daily'] = np.log10(mdata['avg_daily_tokens'].clip(lower=1))
print(f"Analysis dataset: {len(mdata)} models\n")

# ── A. Age-normalized: within release cohort ──────────────────────────────────
print("=" * 65)
print("A. Age control: ELO-usage after controlling for days in window")
print("=" * 65)

# Raw correlation
rho_raw, p_raw = stats.spearmanr(mdata['arena_elo'], mdata['log_tokens'])
print(f"\nRaw Spearman(ELO, log total tokens): rho={rho_raw:.3f}  p={p_raw:.4f}")

# Controlling for days in window (proxy for model age in dataset)
rho_daily, p_daily = stats.spearmanr(mdata['arena_elo'], mdata['log_daily'])
print(f"Spearman(ELO, log avg daily tokens): rho={rho_daily:.3f}  p={p_daily:.4f}")
print(f"  → Age-normalizing {'reduces' if abs(rho_daily) < abs(rho_raw) else 'increases'} correlation "
      f"from {rho_raw:.3f} to {rho_daily:.3f}")

# Partial: ELO vs daily tokens controlling for log(price)
from scipy.stats import spearmanr
def partial_spearman(x, y, z):
    rx, ry, rz = pd.Series(x).rank(), pd.Series(y).rank(), pd.Series(z).rank()
    rx_res = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ry_res = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(rx_res, ry_res)

rho_partial, p_partial = partial_spearman(mdata['arena_elo'], mdata['log_daily'], mdata['log_price'])
print(f"Partial Spearman(ELO, daily | price): rho={rho_partial:.3f}  p={p_partial:.4f}")

# Triple partial: ELO vs daily tokens | price AND days
rho_full, p_full = partial_spearman(
    mdata['arena_elo'],
    mdata['log_daily'],
    mdata['log_price'] + mdata['days_in_window'] / 521  # crude combined control
)
print(f"Partial Spearman(ELO, daily | price + age): rho={rho_full:.3f}  p={p_full:.4f}")

# Release cohort analysis
mdata['release_cohort'] = pd.cut(
    mdata['days_in_window'],
    bins=[0, 90, 180, 300, 600],
    labels=['<3mo', '3-6mo', '6-10mo', '>10mo']
)
print(f"\nSpearman(ELO, avg_daily_tokens) within days-in-window cohorts:")
print(f"{'Cohort':<10}  {'N':>4}  {'rho':>7}  {'p':>8}")
print("-" * 35)
for cohort in ['<3mo', '3-6mo', '6-10mo', '>10mo']:
    sub = mdata[mdata['release_cohort'] == cohort].dropna(subset=['arena_elo','log_daily'])
    if len(sub) < 5: continue
    rho, p = stats.spearmanr(sub['arena_elo'], sub['log_daily'])
    print(f"{str(cohort):<10}  {len(sub):>4}  {rho:>7.3f}  {p:>8.4f}{'*' if p<0.05 else ''}")

# ── B. Within-provider: ELO predicts usage controlling for brand ──────────────
print(f"\n{'='*65}")
print("B. Within-provider: does ELO predict usage controlling for brand?")
print(f"{'='*65}\n")

# Only providers with ≥4 models
provider_counts = mdata['org'].value_counts()
multi_model_orgs = provider_counts[provider_counts >= 4].index.tolist()
print(f"Providers with ≥4 matched models: {multi_model_orgs}\n")

within_org_results = []
for org in multi_model_orgs:
    sub = mdata[mdata['org'] == org].dropna(subset=['arena_elo','log_daily'])
    if len(sub) < 4: continue
    rho, p = stats.spearmanr(sub['arena_elo'], sub['log_daily'])
    within_org_results.append({'org': org, 'n': len(sub), 'rho': rho, 'p': p})
    elo_range = sub['arena_elo'].max() - sub['arena_elo'].min()
    print(f"  {org:<20}  n={len(sub)}  rho={rho:.3f}  p={p:.4f}{'*' if p<0.05 else ''}  "
          f"ELO range={elo_range:.0f}")

wdf = pd.DataFrame(within_org_results)
if len(wdf) > 0:
    print(f"\n  Mean within-org rho: {wdf['rho'].mean():.3f}")
    sig = wdf[wdf['p'] < 0.05]
    print(f"  Orgs where ELO predicts usage (p<0.05): {len(sig)}/{len(wdf)}")

# ── C. Within-price-tier ──────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("C. Within price tier: does ELO add signal?")
print(f"{'='*65}\n")

mdata['price_tier'] = pd.cut(
    mdata['price'],
    bins=[0, 0.3, 1.0, 5.0, 200],
    labels=['cheap (<$0.30)', 'budget ($0.30-$1)', 'mid ($1-$5)', 'premium (>$5)']
)
print(f"{'Tier':<22}  {'N':>4}  {'rho(ELO,daily)':>15}  {'p':>8}  {'ELO range':>10}")
print("-" * 65)
for tier in ['cheap (<$0.30)', 'budget ($0.30-$1)', 'mid ($1-$5)', 'premium (>$5)']:
    sub = mdata[mdata['price_tier'] == tier].dropna(subset=['arena_elo','log_daily'])
    if len(sub) < 4: continue
    rho, p = stats.spearmanr(sub['arena_elo'], sub['log_daily'])
    elo_r = sub['arena_elo'].max() - sub['arena_elo'].min()
    print(f"{str(tier):<22}  {len(sub):>4}  {rho:>15.3f}  {p:>8.4f}{'*' if p<0.05 else ''}  "
          f"{elo_r:>10.1f}")

# ── D. Event study: is the correlation decline step-function or gradual? ──────
print(f"\n{'='*65}")
print("D. Event study: correlation decline vs major model releases")
print(f"{'='*65}")

# Monthly ELO-usage correlation using daily tokens (age-controlled)
monthly_rho = []
for month, grp in df[df['model_permaslug'] != 'other'].groupby(df['date'].dt.to_period('M')):
    sub = grp.copy()
    sub['arena_match'] = sub['model_permaslug'].map(lambda s: match_arena(s)[0])
    sub['arena_elo_val'] = sub['model_permaslug'].map(lambda s: match_arena(s)[1])
    sub = sub.dropna(subset=['arena_elo_val'])
    day_data = sub.groupby('arena_match').agg(
        tokens=('total_tokens','sum'), elo=('arena_elo_val','first')
    ).dropna()
    if len(day_data) < 8: continue
    rho, p = stats.spearmanr(day_data['elo'], day_data['tokens'])
    monthly_rho.append({'month': str(month), 'rho': rho, 'p': p, 'n': len(day_data)})

rho_df = pd.DataFrame(monthly_rho)
print(f"\n{'Month':<10}  {'rho':>7}  {'p':>8}  {'n':>4}  Note")
print("-" * 60)
events = {
    '2025-01': 'DeepSeek-R1 release',
    '2025-03': 'Gemini 2.5 Pro release',
    '2025-04': 'Qwen3 release / LLaMA-4',
    '2025-07': 'Grok-4 / Gemma-3 releases',
    '2025-09': 'Gemini-3 Pro release',
    '2026-01': 'GPT-5 series expansion',
    '2026-02': 'Claude Opus 4.7 / GPT-5.4',
}
for _, row in rho_df.iterrows():
    note = events.get(row['month'], '')
    print(f"{row['month']:<10}  {row['rho']:>7.3f}  {row['p']:>8.4f}  "
          f"{int(row['n']):>4}  {note}")

# Test for step-change vs gradual decline
rho_df['t'] = range(len(rho_df))
sl, _, r2, p2, _ = stats.linregress(rho_df['t'], rho_df['rho'])
print(f"\nLinear trend: slope={sl*12:.3f}/yr  R²={r2**2:.3f}  p={p2:.4f}")

# Chow-like test: split at middle, are slopes different?
mid = len(rho_df) // 2
first_half = rho_df.iloc[:mid]
second_half = rho_df.iloc[mid:]
sl1, _, _, p1, _ = stats.linregress(first_half['t'], first_half['rho'])
sl2, _, _, p2b, _ = stats.linregress(second_half['t'], second_half['rho'])
print(f"First half trend:  slope={sl1*12:.3f}/yr  p={p1:.4f}")
print(f"Second half trend: slope={sl2*12:.3f}/yr  p={p2b:.4f}")
print(f"  → {'Step-change pattern' if abs(sl2-sl1) > 0.2 else 'Gradual decline'}")

# ── E. Bounding: max explainable by distribution shift ───────────────────────
print(f"\n{'='*65}")
print("E. Bounding: how much can distribution shift explain?")
print(f"{'='*65}\n")

# Upper bound: if ALL correlation is age-driven, within-cohort rho should be ~0
# Lower bound: if ALL correlation is genuine, within-cohort rho should equal overall rho

overall_rho_daily, _ = stats.spearmanr(mdata['arena_elo'], mdata['log_daily'])
cohort_rhos = []
for cohort in mdata['release_cohort'].dropna().unique():
    sub = mdata[mdata['release_cohort'] == cohort].dropna(subset=['arena_elo','log_daily'])
    if len(sub) < 5: continue
    rho, _ = stats.spearmanr(sub['arena_elo'], sub['log_daily'])
    cohort_rhos.append(rho)

mean_within_cohort = np.mean(cohort_rhos)
print(f"Overall Spearman(ELO, avg_daily_tokens): {overall_rho_daily:.3f}")
print(f"Mean within-cohort Spearman:             {mean_within_cohort:.3f}")
print()
if overall_rho_daily > 0:
    pct_age = max(0, 1 - mean_within_cohort / overall_rho_daily) * 100
    print(f"Age confound upper bound: {pct_age:.0f}% of raw correlation")
    print(f"Residual genuine signal lower bound: {100-pct_age:.0f}%")
else:
    print("Overall correlation is near zero — most variation is noise/confound")

print()
print("Summary of bounding:")
print(f"  Raw ELO-total_tokens correlation:      rho={rho_raw:.3f}")
print(f"  Age-normalized (avg daily tokens):     rho={rho_daily:.3f}")
print(f"  + Price control:                       rho={rho_partial:.3f}")
print(f"  + Price + age control:                 rho={rho_full:.3f}")
print(f"  Within-provider average:               rho={wdf['rho'].mean():.3f}")
print()
print(f"Best estimate of genuine Arena ELO signal in usage: rho ≈ {rho_full:.2f}")
print(f"  {'Significant' if p_full < 0.05 else 'Not significant'} at p<0.05")

"""theta_std_{i,m,k} = alpha_i + alpha_{i,k} + beta_k*log(size)
  alpha_i    = lab i's OVERALL strength (mean across categories, standardized units)
  alpha_{i,k} = lab i's CATEGORY-SPECIFIC comparative advantage, net of alpha_i
               (sums to ~0 across k within each lab, by construction)
"""
import numpy as np, pandas as pd

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
LABS = ['qwen', 'google', 'meta-llama', 'mistralai', 'deepseek-ai', 'microsoft',
        '01-ai', 'ibm-granite', 'allenai', 'huggingfacetb', 'tiiuae', 'eleutherai',
        'nvidia', 'cohereforai', 'nousresearch', 'bigscience', 'arcee-ai']
MIN_N, MIN_SPAN = 7, 0.6

b = pd.read_parquet('data/benchmarks.parquet')
b['org'] = b.hf_model_id.str.split('/').str[0].str.lower()
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged]
b = b.dropna(subset=DIMS)
b['lp'] = np.log10(b.params_b)

# standardize each benchmark to mean 0, std 1 across the WHOLE pool first
for k in DIMS:
    b[k] = (b[k] - b[k].mean()) / b[k].std()

keep = []
for org, g in b.groupby('org'):
    span = np.log10(g.params_b.max() / g.params_b.min())
    if len(g) >= MIN_N and span >= MIN_SPAN and g.params_b.nunique() >= 4:
        keep.append(org)
b = b[b.org.isin(keep)].reset_index(drop=True)

# fit alpha_{i,k} (lab x category intercept) + shared beta_k, same as Model C but standardized
raw_alpha = {}
for k in DIMS:
    dk = b.dropna(subset=[k])
    labs_u = sorted(dk.org.unique())
    lab_idx = {l: j for j, l in enumerate(labs_u)}
    nL = len(labs_u)
    x = dk.lp.values
    y = dk[k].values
    X = np.zeros((len(x), nL + 1))
    for i, l in enumerate(dk.org.values):
        X[i, lab_idx[l]] = 1.0
    X[:, -1] = x
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    raw_alpha[k] = pd.Series(coef[:nL], index=labs_u)

A = pd.DataFrame(raw_alpha)  # alpha_{i,k}, standardized units
alpha_i = A.mean(axis=1)      # overall lab strength
delta_ik = A.sub(alpha_i, axis=0)  # category-specific residual, sums to ~0 per row

n_models = b.groupby('org').size()

pd.set_option('display.width', 140)
pd.set_option('display.max_rows', None)

print('=== alpha_i: OVERALL lab strength (standardized units, major labs) ===')
major = pd.DataFrame({'alpha_i': alpha_i, 'n_models': n_models}).loc[alpha_i.index.isin(LABS)]
print(major.sort_values('alpha_i', ascending=False).round(3).to_string())

print()
print('=== delta_{i,k}: CATEGORY-SPECIFIC advantage net of alpha_i (major labs) ===')
delta_major = delta_ik.loc[delta_ik.index.isin(LABS)]
print(delta_major.round(2).loc[major.sort_values('alpha_i', ascending=False).index].to_string())

print()
print('=== biggest single |delta_{i,k}| cells (major labs): where a lab is NOT just generally strong/weak,')
print('    but has a specific pull toward or away from one category ===')
flat = delta_major.stack().rename('delta').reset_index()
flat.columns = ['lab','cat','delta']
print(flat.reindex(flat.delta.abs().sort_values(ascending=False).index).head(15).to_string(index=False))

print()
print('=================== FULL PANEL (all 65 orgs) ===================')
print()
print('=== alpha_i: top 10 / bottom 10 overall strength ===')
full = pd.DataFrame({'alpha_i': alpha_i, 'n_models': n_models})
print(full.sort_values('alpha_i', ascending=False).head(10).round(3).to_string())
print('...')
print(full.sort_values('alpha_i', ascending=False).tail(10).round(3).to_string())

print()
print('=== universal pattern check: sign of mean delta_{i,k} per category, across ALL orgs ===')
print((delta_ik > 0).mean().round(2).rename('frac_orgs_positive'))

print()
print('=== biggest |delta_{i,k}| cells in the FULL panel ===')
flat_full = delta_ik.stack().rename('delta').reset_index()
flat_full.columns = ['lab','cat','delta']
print(flat_full.reindex(flat_full.delta.abs().sort_values(ascending=False).index).head(20).to_string(index=False))

print()
print('=== specialization: std of delta_{i,k} across categories, per org (high = spiky/specialized, low = generalist) ===')
spec = delta_ik.std(axis=1).rename('delta_std')
spec_df = pd.DataFrame({'delta_std': spec, 'alpha_i': alpha_i, 'n_models': n_models}).sort_values('delta_std', ascending=False)
print('most SPECIALIZED (top 10):')
print(spec_df.head(10).round(3).to_string())
print('most GENERALIST (bottom 10):')
print(spec_df.tail(10).round(3).to_string())

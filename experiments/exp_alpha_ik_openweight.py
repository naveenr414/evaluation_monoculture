"""Extract actual alpha_{i,k} (lab x category intercept, shared slope beta_k) from
Model C, fit on the FULL open-weight dataset (no CV -- point estimates for inspection).
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

keep = []
for org, g in b.groupby('org'):
    span = np.log10(g.params_b.max() / g.params_b.min())
    if len(g) >= MIN_N and span >= MIN_SPAN and g.params_b.nunique() >= 4:
        keep.append(org)
b = b[b.org.isin(keep)].reset_index(drop=True)
print(f'labs kept: {len(keep)}   models: {len(b)}')

def fit_model_c(dk, k):
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
    alpha = pd.Series(coef[:nL], index=labs_u)
    beta = coef[-1]
    resid = y - X @ coef
    r2 = 1 - np.sum(resid**2)/np.sum((y-y.mean())**2)
    return alpha, beta, r2, dk.groupby('org').size()

all_alpha = {}
betas = {}
for k in DIMS:
    dk = b.dropna(subset=[k])
    alpha, beta, r2, n = fit_model_c(dk, k)
    all_alpha[k] = alpha
    betas[k] = beta
    print(f'{k}: shared beta_k (per log10 param) = {beta:.2f}   in-sample R2 = {r2:.3f}')

A = pd.DataFrame(all_alpha)
n_models = b.groupby('org').size()
A['n_models'] = A.index.map(n_models)
A_major = A[A.index.isin(LABS)].copy()

pd.set_option('display.width', 140)
pd.set_option('display.max_rows', None)
print()
print('=== alpha_{i,k}: MAJOR/known labs only (predicted score AT params_b=1, i.e. log(params)=0) ===')
print(A_major.round(2).sort_values('bbh', ascending=False).to_string())

print()
print('=== full panel (65 orgs), top/bottom 10 by mean alpha across categories ===')
A['mean_alpha'] = A[DIMS].mean(axis=1)
print(A.sort_values('mean_alpha', ascending=False).head(10)[DIMS+['n_models','mean_alpha']].round(2).to_string())
print('...')
print(A.sort_values('mean_alpha', ascending=False).tail(10)[DIMS+['n_models','mean_alpha']].round(2).to_string())

print()
print('=== spread of alpha within major labs (std across the 17 known labs), per category ===')
print(A_major[DIMS].std().round(2))

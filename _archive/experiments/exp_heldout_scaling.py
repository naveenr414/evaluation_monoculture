"""Held-out prediction comparison for 4 nested scaling models on the open-weight
benchmark data:
  A: theta = alpha_k + beta_k*log(size)
  B: theta = alpha_k + beta_k*log(size) + gamma_k*t                    [+ shared time trend]
  C: theta = alpha_{i,k} + beta_k*log(size)                            [+ lab intercept]
  D: theta = alpha_{i,k} + beta_{i,k}*log(size) + gamma_k*t            [+ lab-specific slope]

5-fold CV, per category, held-out R^2 (baseline = training-fold mean).
"""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
LABS = ['qwen', 'google', 'meta-llama', 'mistralai', 'deepseek-ai', 'microsoft',
        '01-ai', 'ibm-granite', 'allenai', 'huggingfacetb', 'tiiuae', 'eleutherai',
        'nvidia', 'cohereforai', 'nousresearch', 'bigscience', 'arcee-ai']
MIN_N, MIN_SPAN = 7, 0.6

b = pd.read_parquet('data/benchmarks.parquet')
b['org'] = b.hf_model_id.str.split('/').str[0].str.lower()
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged & b.upload_date.notna()]
b = b.dropna(subset=DIMS)
b['lp'] = np.log10(b.params_b)
b['t'] = (b.upload_date - b.upload_date.min()).dt.days / 365.25

keep = []
for org, g in b.groupby('org'):
    span = np.log10(g.params_b.max() / g.params_b.min())
    if len(g) >= MIN_N and span >= MIN_SPAN and g.params_b.nunique() >= 4:
        keep.append(org)
b = b[b.org.isin(keep)].reset_index(drop=True)
print(f'labs kept: {len(keep)}   models: {len(b)}')

def dummy_design(labs, values):
    labs_u = sorted(labs.unique())
    idx = {l: j for j, l in enumerate(labs_u)}
    M = np.zeros((len(labs), len(labs_u)))
    for i, l in enumerate(labs):
        M[i, idx[l]] = values[i] if values is not None else 1.0
    return M, labs_u

def fit_predict(model_name, train, test):
    x_tr, y_tr, t_tr, lab_tr = train.lp.values, train[k].values, train.t.values, train.org.values
    x_te, t_te, lab_te = test.lp.values, test.t.values, test.org.values

    if model_name == 'A':
        X = np.column_stack([np.ones(len(x_tr)), x_tr])
        coef, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
        Xte = np.column_stack([np.ones(len(x_te)), x_te])
        return Xte @ coef

    if model_name == 'B':
        X = np.column_stack([np.ones(len(x_tr)), x_tr, t_tr])
        coef, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
        Xte = np.column_stack([np.ones(len(x_te)), x_te, t_te])
        return Xte @ coef

    if model_name == 'C':
        # lab intercept dummies + shared slope on x
        D_tr, labs_u = dummy_design(pd.Series(lab_tr), None)
        X = np.column_stack([D_tr, x_tr])
        coef, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
        lab_idx = {l: j for j, l in enumerate(labs_u)}
        global_int = coef[:len(labs_u)].mean()  # fallback for unseen labs
        pred = np.zeros(len(x_te))
        for i in range(len(x_te)):
            a = coef[lab_idx[lab_te[i]]] if lab_te[i] in lab_idx else global_int
            pred[i] = a + coef[-1]*x_te[i]
        return pred

    if model_name == 'D':
        # lab intercept + lab slope dummies + shared time trend
        labs_u = sorted(pd.Series(lab_tr).unique())
        lab_idx = {l: j for j, l in enumerate(labs_u)}
        nL = len(labs_u)
        X = np.zeros((len(x_tr), 2*nL + 1))
        for i in range(len(x_tr)):
            j = lab_idx[lab_tr[i]]
            X[i, j] = 1.0
            X[i, nL + j] = x_tr[i]
        X[:, -1] = t_tr
        coef, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
        global_int = coef[:nL].mean()
        global_slope = coef[nL:2*nL].mean()
        pred = np.zeros(len(x_te))
        for i in range(len(x_te)):
            if lab_te[i] in lab_idx:
                j = lab_idx[lab_te[i]]
                a, s = coef[j], coef[nL+j]
            else:
                a, s = global_int, global_slope
            pred[i] = a + s*x_te[i] + coef[-1]*t_te[i]
        return pred

results = {m: {} for m in ['A','B','C','D']}
N_SPLITS = 5
for k in DIMS:
    dk = b.dropna(subset=[k]).reset_index(drop=True)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    ss_res = {m: 0.0 for m in ['A','B','C','D']}
    ss_tot = 0.0
    for train_idx, test_idx in kf.split(dk):
        train, test = dk.iloc[train_idx], dk.iloc[test_idx]
        y_te = test[k].values
        baseline = train[k].mean()
        ss_tot += np.sum((y_te - baseline)**2)
        for m in ['A','B','C','D']:
            pred = fit_predict(m, train, test)
            ss_res[m] += np.sum((y_te - pred)**2)
    for m in ['A','B','C','D']:
        results[m][k] = 1 - ss_res[m]/ss_tot

res_df = pd.DataFrame(results)
res_df.loc['MEAN'] = res_df.mean()
pd.set_option('display.width', 120)
print()
print('Held-out R^2 (5-fold CV), by category:')
print(res_df.round(3).to_string())

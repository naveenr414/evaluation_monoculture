"""Same 4 nested models, but FORWARD-IN-TIME split: train on models uploaded
before a cutoff, predict scores for models uploaded after it (true forecasting,
not interpolation).
"""
import numpy as np, pandas as pd

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
MIN_N, MIN_SPAN = 7, 0.6
CUTOFF = pd.Timestamp('2025-01-22')  # ~75th percentile -> train on ~75%, test on newest ~25%

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
print(f'cutoff: {CUTOFF.date()}  train: {(b.upload_date < CUTOFF).sum()}  test: {(b.upload_date >= CUTOFF).sum()}')

def fit_predict(model_name, train, test, k):
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
        labs_u = sorted(pd.Series(lab_tr).unique())
        lab_idx = {l: j for j, l in enumerate(labs_u)}
        nL = len(labs_u)
        X = np.zeros((len(x_tr), nL + 1))
        for i, l in enumerate(lab_tr):
            X[i, lab_idx[l]] = 1.0
        X[:, -1] = x_tr
        coef, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
        global_int = coef[:nL].mean()
        pred = np.zeros(len(x_te))
        for i in range(len(x_te)):
            a = coef[lab_idx[lab_te[i]]] if lab_te[i] in lab_idx else global_int
            pred[i] = a + coef[-1]*x_te[i]
        return pred

    if model_name == 'D':
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
n_test = {}
for k in DIMS:
    dk = b.dropna(subset=[k])
    train = dk[dk.upload_date < CUTOFF]
    test = dk[dk.upload_date >= CUTOFF]
    n_test[k] = len(test)
    if len(test) < 10:
        continue
    baseline = train[k].mean()
    ss_tot = np.sum((test[k].values - baseline)**2)
    for m in ['A','B','C','D']:
        pred = fit_predict(m, train, test, k)
        ss_res = np.sum((test[k].values - pred)**2)
        results[m][k] = 1 - ss_res/ss_tot

res_df = pd.DataFrame(results)
res_df['n_test'] = pd.Series(n_test)
res_df.loc['MEAN'] = res_df.mean()
pd.set_option('display.width', 120)
print()
print(f'Forward-in-time held-out R^2 (train < {CUTOFF.date()}, test >= {CUTOFF.date()}):')
print(res_df.round(3).to_string())

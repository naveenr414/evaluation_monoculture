"""Split the A -> C forward-in-time R^2 jump into two steps:
  A:            theta = alpha_k + beta_k*log(size)
  A + alpha_i:  theta = alpha_k + alpha_i + beta_k*log(size)      [ONE shared lab shift, all categories]
  A + alpha_i + delta_{i,k} (= C):  theta = alpha_k + alpha_i + delta_{i,k} + beta_k*log(size)
                = full lab x category intercept (same predictions as before)
Forward-in-time: train on models before CUTOFF, test on models after.
"""
import numpy as np, pandas as pd

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
MIN_N, MIN_SPAN = 7, 0.6
CUTOFF = pd.Timestamp('2025-01-22')

b = pd.read_parquet('data/benchmarks.parquet')
b['org'] = b.hf_model_id.str.split('/').str[0].str.lower()
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged & b.upload_date.notna()]
b = b.dropna(subset=DIMS)
b['lp'] = np.log10(b.params_b)

keep = []
for org, g in b.groupby('org'):
    span = np.log10(g.params_b.max() / g.params_b.min())
    if len(g) >= MIN_N and span >= MIN_SPAN and g.params_b.nunique() >= 4:
        keep.append(org)
b = b[b.org.isin(keep)].reset_index(drop=True)

# long format: one row per (model, category)
long = b.melt(id_vars=['org','lp','upload_date'], value_vars=DIMS, var_name='cat', value_name='y').dropna(subset=['y'])
train_all = long[long.upload_date < CUTOFF]
test_all = long[long.upload_date >= CUTOFF]

def fit_predict_joint(model_name, train, test):
    labs_u = sorted(train.org.unique())
    lab_idx = {l: j for j, l in enumerate(labs_u)}
    nL = len(labs_u)
    cats_u = DIMS
    cat_idx = {k: j for j, k in enumerate(cats_u)}
    nC = len(cats_u)

    def build_X(df, include_alpha_i, include_delta_ik):
        n = len(df)
        cols = []
        # alpha_k: category intercept
        Ck = np.zeros((n, nC))
        for i, k in enumerate(df.cat.values):
            Ck[i, cat_idx[k]] = 1.0
        cols.append(Ck)
        # beta_k * log(size): category-specific slope
        Bk = np.zeros((n, nC))
        for i, (k, x) in enumerate(zip(df.cat.values, df.lp.values)):
            Bk[i, cat_idx[k]] = x
        cols.append(Bk)
        if include_alpha_i:
            Li = np.zeros((n, nL))
            for i, l in enumerate(df.org.values):
                if l in lab_idx:
                    Li[i, lab_idx[l]] = 1.0
            cols.append(Li)
        if include_delta_ik:
            Lik = np.zeros((n, nL*nC))
            for i, (l, k) in enumerate(zip(df.org.values, df.cat.values)):
                if l in lab_idx:
                    Lik[i, lab_idx[l]*nC + cat_idx[k]] = 1.0
            cols.append(Lik)
        return np.hstack(cols)

    inc_a = model_name in ('A+alpha_i', 'A+alpha_i+delta_ik')
    inc_d = model_name == 'A+alpha_i+delta_ik'
    Xtr = build_X(train, inc_a, inc_d)
    coef, *_ = np.linalg.lstsq(Xtr, train.y.values, rcond=None)
    Xte = build_X(test, inc_a, inc_d)
    return Xte @ coef

results = {m: {} for m in ['A','A+alpha_i','A+alpha_i+delta_ik']}
for k in DIMS:
    tr = train_all[train_all.cat == k]
    te = test_all[test_all.cat == k]
    baseline = tr.y.mean()
    ss_tot = np.sum((te.y.values - baseline)**2)
    for m in results:
        pred = fit_predict_joint(m, train_all, test_all)  # joint fit across ALL cats
        # extract just this category's test rows' predictions
        mask = test_all.cat.values == k
        ss_res = np.sum((te.y.values - pred[mask])**2)
        results[m][k] = 1 - ss_res/ss_tot

res_df = pd.DataFrame(results)
res_df.loc['MEAN'] = res_df.mean()
print('Forward-in-time held-out R^2, split by alpha_i then delta_ik:')
print(res_df.round(3).to_string())
print()
print('Incremental gain:')
print(f'  A alone                    : {res_df.loc["MEAN","A"]:.3f}')
print(f'  + alpha_i (shared lab shift): +{res_df.loc["MEAN","A+alpha_i"]-res_df.loc["MEAN","A"]:.3f}  -> {res_df.loc["MEAN","A+alpha_i"]:.3f}')
print(f'  + delta_ik (category-specific): +{res_df.loc["MEAN","A+alpha_i+delta_ik"]-res_df.loc["MEAN","A+alpha_i"]:.3f}  -> {res_df.loc["MEAN","A+alpha_i+delta_ik"]:.3f}')

"""Same rolling-origin (train < t, predict next slice) setup as exp_F1234_rolling.py,
adding the model the user actually asked about: a PARSIMONIOUS continuous rank-1
interaction, not F3's saturated per-cell dummy delta_{i,k}.

  F2:  a_k + alpha_i + b_k*log(x)                       (rank-0 interaction: A_i*A_k*log(1+x))
  F3:  a_k + alpha_i + delta_{i,k} + b_k*log(x)          (saturated/full-rank interaction)
  F3b: a_k + alpha_i + u_i*v_k + b_k*log(x)              (rank-1 continuous interaction:
                                                            A_i*A_k*exp(u_i*v_k)*log(1+x)
                                                            in log space)

F3b is bilinear in (u_i, v_k) so it's fit by ridge-regularized alternating least
squares on the training residual (after removing the a_k/alpha_i/b_k*log(x) part),
refit fresh within each fold on train data only. Unseen labs at test time get u_i=0
(collapse to F2's prediction), same fallback convention as F3/F4.
"""
import numpy as np, pandas as pd

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
MIN_N, MIN_SPAN = 7, 0.6
N_FOLDS = 5
MIN_TRAIN = 60
RIDGE = 1.0       # shrinkage on u_i, v_k toward 0 (toward F2)
N_ALS_ITER = 30
N_RESTARTS = 4

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
print(f'labs kept: {len(keep)}   models: {len(b)}')

long = b.melt(id_vars=['org', 'lp', 'upload_date'], value_vars=DIMS,
              var_name='cat', value_name='y').dropna(subset=['y']).reset_index(drop=True)
long = long.sort_values('upload_date').reset_index(drop=True)

edges = long.upload_date.quantile(np.linspace(0, 1, N_FOLDS + 1)).values
edges = np.unique(edges)

cat_idx_g = {k: j for j, k in enumerate(DIMS)}
nC = len(DIMS)


def build_baseline(df, lab_idx):
    n, nL = len(df), len(lab_idx)
    Ak = np.zeros((n, nC)); Bk = np.zeros((n, nC)); Ai = np.zeros((n, nL))
    for i, (k, l, x) in enumerate(zip(df.cat.values, df.org.values, df.lp.values)):
        j = cat_idx_g[k]
        Ak[i, j] = 1.0
        Bk[i, j] = x
        if l in lab_idx:
            Ai[i, lab_idx[l]] = 1.0
    return np.hstack([Ak, Bk, Ai])


def fit_predict_F2(train, test):
    labs_u = sorted(train.org.unique())
    lab_idx = {l: j for j, l in enumerate(labs_u)}
    Xtr = build_baseline(train, lab_idx)
    coef, *_ = np.linalg.lstsq(Xtr, train.y.values, rcond=None)
    Xte = build_baseline(test, lab_idx)
    return Xte @ coef, coef, lab_idx


def als_rank1(resid, lab_of, cat_of, nL, seed):
    rng = np.random.default_rng(seed)
    v = 1.0 + 0.1 * rng.standard_normal(nC)
    u = np.zeros(nL)
    for _ in range(N_ALS_ITER):
        num_u = np.zeros(nL); den_u = np.zeros(nL)
        np.add.at(num_u, lab_of, resid * v[cat_of])
        np.add.at(den_u, lab_of, v[cat_of] ** 2)
        u = num_u / (den_u + RIDGE)

        num_v = np.zeros(nC); den_v = np.zeros(nC)
        np.add.at(num_v, cat_of, resid * u[lab_of])
        np.add.at(den_v, cat_of, u[lab_of] ** 2)
        v = num_v / (den_v + RIDGE)
    pred = u[lab_of] * v[cat_of]
    sse = np.sum((resid - pred) ** 2)
    return u, v, sse


def fit_predict_F3b(train, test):
    pred_base, coef, lab_idx = fit_predict_F2(train, test)
    Xtr = build_baseline(train, lab_idx)
    resid_tr = train.y.values - Xtr @ coef
    lab_of = train.org.map(lab_idx).values.astype(int)
    cat_of = train.cat.map(cat_idx_g).values.astype(int)
    nL = len(lab_idx)

    best = None
    for s in range(N_RESTARTS):
        u, v, sse = als_rank1(resid_tr, lab_of, cat_of, nL, seed=s)
        if best is None or sse < best[2]:
            best = (u, v, sse)
    u, v, _ = best

    pred = pred_base.copy()
    for i, (l, k) in enumerate(zip(test.org.values, test.cat.values)):
        if l in lab_idx:
            pred[i] += u[lab_idx[l]] * v[cat_idx_g[k]]
    return pred


MODELS = ['F2', 'F3b']
fold_rows = []
for f in range(len(edges) - 1):
    t_start, t_end = edges[f], edges[f + 1]
    train = long[long.upload_date < t_start]
    test = long[(long.upload_date >= t_start) & (long.upload_date < t_end)]
    if len(train) < MIN_TRAIN or len(test) < 10:
        continue
    pred_by_model = {}
    pred_by_model['F2'], _, _ = fit_predict_F2(train, test)
    pred_by_model['F3b'] = fit_predict_F3b(train, test)
    for k in DIMS:
        te_k_mask = (test.cat.values == k)
        te_k = test[te_k_mask]
        if len(te_k) < 3:
            continue
        tr_k = train[train.cat == k]
        baseline = tr_k.y.mean()
        ss_tot = np.sum((te_k.y.values - baseline) ** 2)
        row = dict(fold=f, cat=k, n_test=len(te_k), n_train=len(tr_k),
                   t_start=pd.Timestamp(t_start).date(), t_end=pd.Timestamp(t_end).date())
        for m in MODELS:
            ss_res = np.sum((te_k.y.values - pred_by_model[m][te_k_mask]) ** 2)
            row[m] = 1 - ss_res / ss_tot
        fold_rows.append(row)

res = pd.DataFrame(fold_rows)
pd.set_option('display.width', 140)
print('\nPer fold x category held-out R^2:')
print(res.round(3).to_string(index=False))

print('\nMean held-out R^2 by fold (weighted by n_test):')
def wmean(g):
    w = g.n_test.values
    return pd.Series({m: np.average(g[m], weights=w) for m in MODELS})
by_fold = res.groupby('fold').apply(wmean, include_groups=False)
by_fold['n_test'] = res.groupby('fold').n_test.sum()
by_fold['t_end'] = res.groupby('fold').t_end.first()
print(by_fold.round(3).to_string())

print('\nOverall mean held-out R^2 (weighted, pooled):')
w = res.n_test.values
for m in MODELS:
    print(f'  {m}: {np.average(res[m], weights=w):.3f}')
print('\n(cf. exp_F1234_rolling.py: F1=0.334  F2=0.191  F3(saturated)=0.356  F4=0.040)')

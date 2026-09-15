"""Rolling-origin (train up to t, predict next slice) held-out comparison of
four nested production-function classes, in the user's F_{i,k}(x) notation:

  F1: a_k + b_k*log(x)
  F2: a_k + alpha_i + b_k*log(x)
  F3: a_k + alpha_i + delta_{i,k} + b_k*log(x)
  F4: a_k + alpha_i + delta_{i,k} + (b_k + s_{i,k})*log(x)

x = params_b (log10). Panel = open-weight benchmarks.parquet, long format
(one row per model x category). Time = upload_date, split into folds by
quantile so each fold has a comparable amount of NEW data, and we predict
each fold using only data strictly before it (no leakage, no interpolation).
"""
import numpy as np, pandas as pd

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
MIN_N, MIN_SPAN = 7, 0.6
N_FOLDS = 5          # rolling folds after an initial warm-up fold
MIN_TRAIN = 60        # don't score a fold until train has at least this many long-format rows

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

# fold edges by quantile of upload_date -> each fold boundary sees a similar
# amount of NEW rows, giving N_FOLDS rolling train/test splits
edges = long.upload_date.quantile(np.linspace(0, 1, N_FOLDS + 1)).values
edges = np.unique(edges)

MODELS = ['F1', 'F2', 'F3', 'F4']


def build_X(df, labs_u, lab_idx, model):
    n = len(df)
    nC, nL = len(DIMS), len(labs_u)
    cat_idx = {k: j for j, k in enumerate(DIMS)}
    cols = []
    Ak = np.zeros((n, nC))          # a_k
    for i, k in enumerate(df.cat.values):
        Ak[i, cat_idx[k]] = 1.0
    cols.append(Ak)
    Bk = np.zeros((n, nC))          # b_k * log x  (shared slope, always present as baseline)
    for i, (k, x) in enumerate(zip(df.cat.values, df.lp.values)):
        Bk[i, cat_idx[k]] = x
    cols.append(Bk)

    if model in ('F2', 'F3', 'F4'):
        Ai = np.zeros((n, nL))      # alpha_i
        for i, l in enumerate(df.org.values):
            if l in lab_idx:
                Ai[i, lab_idx[l]] = 1.0
        cols.append(Ai)

    if model in ('F3', 'F4'):
        Dik = np.zeros((n, nL * nC))  # delta_{i,k}
        for i, (l, k) in enumerate(zip(df.org.values, df.cat.values)):
            if l in lab_idx:
                Dik[i, lab_idx[l] * nC + cat_idx[k]] = 1.0
        cols.append(Dik)

    if model == 'F4':
        Sik = np.zeros((n, nL * nC))  # s_{i,k} * log x  (extra lab-specific slope)
        for i, (l, k, x) in enumerate(zip(df.org.values, df.cat.values, df.lp.values)):
            if l in lab_idx:
                Sik[i, lab_idx[l] * nC + cat_idx[k]] = x
        cols.append(Sik)

    return np.hstack(cols)


def fit_predict(model, train, test):
    labs_u = sorted(train.org.unique())
    lab_idx = {l: j for j, l in enumerate(labs_u)}
    Xtr = build_X(train, labs_u, lab_idx, model)
    coef, *_ = np.linalg.lstsq(Xtr, train.y.values, rcond=None)
    Xte = build_X(test, labs_u, lab_idx, model)
    pred = Xte @ coef
    # unseen labs (not in training set) get no alpha_i/delta_ik/s_ik dummy fired ->
    # their prediction collapses to a_k + b_k*log(x) (F1) automatically, which is
    # the right fallback and is scored honestly (not excluded).
    return pred


fold_rows = []
for f in range(len(edges) - 1):
    t_start, t_end = edges[f], edges[f + 1]
    train = long[long.upload_date < t_start]
    test = long[(long.upload_date >= t_start) & (long.upload_date < t_end)]
    if len(train) < MIN_TRAIN or len(test) < 10:
        continue
    for k in DIMS:
        tr_k, te_k = train[train.cat == k], test[test.cat == k]
        if len(te_k) < 3:
            continue
        baseline = tr_k.y.mean()
        ss_tot = np.sum((te_k.y.values - baseline) ** 2)
        row = dict(fold=f, cat=k, n_test=len(te_k), n_train=len(tr_k),
                   t_start=pd.Timestamp(t_start).date(), t_end=pd.Timestamp(t_end).date())
        for m in MODELS:
            pred = fit_predict(m, train, test)
            mask = (test.cat.values == k)
            ss_res = np.sum((te_k.y.values - pred[mask]) ** 2)
            row[m] = 1 - ss_res / ss_tot
        fold_rows.append(row)

res = pd.DataFrame(fold_rows)
pd.set_option('display.width', 140)
print()
print('Per fold x category held-out R^2:')
print(res.round(3).to_string(index=False))

print()
print('Mean held-out R^2 by fold (rolling-origin, weighted by n_test within fold):')
def wmean(g):
    w = g.n_test.values
    return pd.Series({m: np.average(g[m], weights=w) for m in MODELS})
by_fold = res.groupby('fold').apply(wmean, include_groups=False)
by_fold['n_test'] = res.groupby('fold').n_test.sum()
by_fold['t_end'] = res.groupby('fold').t_end.first()
print(by_fold.round(3).to_string())

print()
print('Overall mean held-out R^2 (weighted by n_test, pooled across folds+categories):')
w = res.n_test.values
overall = {m: np.average(res[m], weights=w) for m in MODELS}
for m in MODELS:
    print(f'  {m}: {overall[m]:.3f}')

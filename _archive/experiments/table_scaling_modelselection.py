"""Table 3: which functional form for F best fits the scaling data (empirical
model selection via cross-validated held-out error), and how much does allowing
scaling behavior to vary by firm, then by firm x category, improve prediction?

Candidates (x = params in billions, theta = raw benchmark score):
  Linear                          theta = a x + b
  Log                             theta = a log(x) + b
  Power law                       theta = a x^gamma + b
  Power law + firm                theta = a_i x^gamma + b_i           (gamma shared)
  Power law + firm x category     theta = a_{i,k} x^gamma + b_{i,k}   (gamma shared)

gamma is refit within each training fold (nonlinear, via scipy curve_fit on the
pooled power-law model only); given gamma, every row is LINEAR least squares in
z = x^gamma, so a_i/b_i and a_{i,k}/b_{i,k} are simple, stable per-group fits.
5-fold CV, split by MODEL (not by row) so a model's 6 category-scores never
straddle train/test -- a within-firm/category group unseen in a training fold
falls back to the pooled global (a,b) for that fold's predictions.

Writes experiments/table3_scaling_modelselection.tex
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DIMS = ['ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu_pro']
LABS = ['qwen', 'google', 'meta-llama', 'mistralai', 'deepseek-ai', 'microsoft',
        '01-ai', 'ibm-granite', 'allenai', 'huggingfacetb', 'tiiuae', 'eleutherai',
        'nvidia', 'cohereforai', 'nousresearch', 'bigscience', 'arcee-ai']
MIN_N, MIN_SPAN = 7, 0.6

b = pd.read_parquet('data/benchmarks.parquet')
b['org'] = b.hf_model_id.str.split('/').str[0].str.lower()
b = b[b.params_b.notna() & (b.params_b > 0) & ~b.flagged & ~b.merged]
b = b.dropna(subset=DIMS)
b = b[b.org.isin(LABS)].copy()

keep = []
for org, g in b.groupby('org'):
    span = np.log10(g.params_b.max() / g.params_b.min())
    if len(g) >= MIN_N and span >= MIN_SPAN and g.params_b.nunique() >= 4:
        keep.append(org)
b = b[b.org.isin(keep)].copy()

long = b.melt(id_vars=['hf_model_id', 'org', 'params_b'], value_vars=DIMS,
              var_name='cat', value_name='theta')
long['ik'] = long.org + '|' + long.cat
print(f'models={b.hf_model_id.nunique()}  labs={len(keep)}  long rows={len(long)}')

models = long.hf_model_id.unique()
rng = np.random.default_rng(0)
rng.shuffle(models)
K = 5
folds = np.array_split(models, K)


def group_lstsq(z, y, groups, global_ab):
    """Per-group (a,b) linear fit in feature z; groups unseen fall back to global_ab."""
    out = {}
    for grp in pd.unique(groups):
        m = groups == grp
        if m.sum() >= 2 and len(np.unique(z[m])) >= 2:
            A = np.column_stack([z[m], np.ones(m.sum())])
            coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            out[grp] = coef
        else:
            out[grp] = global_ab
    return out


def predict_grouped(z, groups, fits, global_ab):
    pred = np.empty(len(z))
    for i, (zi, g) in enumerate(zip(z, groups)):
        a, bb = fits.get(g, global_ab)
        pred[i] = a * zi + bb
    return pred


results = {name: [] for name in
           ['Linear', 'Log', 'Power law', 'Power law + firm', 'Power law + firm x category']}
gammas = []

for kf in range(K):
    test_models = set(folds[kf])
    train = long[~long.hf_model_id.isin(test_models)]
    test = long[long.hf_model_id.isin(test_models)]

    x_tr, y_tr = train.params_b.values, train.theta.values
    x_te, y_te = test.params_b.values, test.theta.values

    # Linear
    A = np.column_stack([x_tr, np.ones(len(x_tr))])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    pred = coef[0] * x_te + coef[1]
    results['Linear'].append(y_te - pred)

    # Log
    lx_tr, lx_te = np.log(x_tr), np.log(x_te)
    A = np.column_stack([lx_tr, np.ones(len(lx_tr))])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    pred = coef[0] * lx_te + coef[1]
    results['Log'].append(y_te - pred)

    # Power law: fit gamma via nonlinear LS on training data only
    def powerlaw(x, a, gamma, c):
        return a * np.power(x, gamma) + c
    p0 = [1.0, 0.3, y_tr.mean()]
    try:
        popt, _ = curve_fit(powerlaw, x_tr, y_tr, p0=p0,
                             bounds=([-np.inf, 0.01, -np.inf], [np.inf, 3.0, np.inf]),
                             maxfev=20000)
        a_g, gamma, c_g = popt
    except RuntimeError:
        a_g, gamma, c_g = p0
    gammas.append(gamma)
    z_tr, z_te = np.power(x_tr, gamma), np.power(x_te, gamma)
    pred_pooled = a_g * z_te + c_g
    results['Power law'].append(y_te - pred_pooled)
    global_ab = (a_g, c_g)

    # Power law + firm (same gamma, per-org a_i,b_i)
    fits_firm = group_lstsq(z_tr, y_tr, train.org.values, global_ab)
    pred_firm = predict_grouped(z_te, test.org.values, fits_firm, global_ab)
    results['Power law + firm'].append(y_te - pred_firm)

    # Power law + firm x category
    fits_ik = group_lstsq(z_tr, y_tr, train.ik.values, global_ab)
    pred_ik = predict_grouped(z_te, test.ik.values, fits_ik, global_ab)
    results['Power law + firm x category'].append(y_te - pred_ik)

print(f'\nfitted gamma per fold: {[f"{g:.3f}" for g in gammas]}  (mean {np.mean(gammas):.3f})')

rmse = {name: np.sqrt(np.mean(np.concatenate(res) ** 2)) for name, res in results.items()}
print(f'\n{"model":32s} {"held-out RMSE":>14s}')
for name, e in rmse.items():
    print(f'{name:32s} {e:14.3f}')

firm_improve = (rmse['Power law'] - rmse['Power law + firm']) / rmse['Power law'] * 100
cat_improve = (rmse['Power law + firm'] - rmse['Power law + firm x category']) / rmse['Power law + firm'] * 100
print(f'\nfirm-specific improvement over pooled power law: {firm_improve:+.1f}%')
print(f'category-specific improvement over +firm:          {cat_improve:+.1f}%')

# --------------------------------------------------------------- LaTeX -----
rows = [
    ('Linear', rmse['Linear'], None, None),
    ('Log', rmse['Log'], None, None),
    ('Power law', rmse['Power law'], None, None),
    ('Power law + firm parameters', rmse['Power law + firm'], firm_improve, None),
    ('Power law + firm $\\times$ category parameters', rmse['Power law + firm x category'],
     firm_improve, cat_improve),
]
lines = [r'\begin{tabular}{lccc}', r'\toprule',
         r'Model for $F$ & Held-out error $\downarrow$ & Firm-specific improvement & '
         r'Category-specific improvement \\', r'\midrule']
for name, err, fi, ci in rows:
    fi_s = '--' if fi is None else f'{fi:+.1f}\\%'
    ci_s = '--' if ci is None else f'{ci:+.1f}\\%'
    lines.append(f'{name} & {err:.3f} & {fi_s} & {ci_s} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
with open('experiments/table3_scaling_modelselection.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\nwrote experiments/table3_scaling_modelselection.tex')

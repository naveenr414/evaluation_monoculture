"""RQ2 (eval-chasing): progressively-controlled regressions of

    dtheta_{i,k,t+1} = beta * Gap_{i,k,t} + [controls] + eps

Six columns, each adding one more control on top of the last:
  (A) Basic            : Gap only, pooled, no FE
  (B) Model 1           : + firm FE (a_i) + category FE (g_k) + time FE (l_t)
  (C) Model 2           : + firm x category FE (a_{i,k}) + time FE (l_t)
  (D) Model 3 (main)    : + firm x category FE (a_{i,k}) + category x time FE (l_{k,t})
  (E) + Scaling         : (D) + log(days since the firm's previous flagship release)
                           -- model-size/compute controls aren't usable here: public
                           parameter counts don't exist for the actual closed frontier
                           flagships in this sample (checked against Epoch's tracker),
                           so release cadence is the available "scaling"-adjacent proxy
  (F) Full (+ noise corr): (E) with the errors-in-variables sampling-noise correction
                           (Gap and dtheta both carry -eps_old, the outgoing flagship's
                           own BT sampling noise -> biases beta up; corrected using the
                           per-observation BT variances, as in exp1_evalchasing.py)

"time" = the switch date binned to quarter, so l_t / l_{k,t} are estimable FE.
Writes experiments/table_evalchasing.tex and prints the same table to stdout.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'experiments')
from common import load_panel, build_events, fe_regress, CORE, FRONTIER

d = load_panel(cats=CORE)
ev = build_events(d, cats=CORE)

# time bin (quarter) and firm x category / category x time interaction keys
ev['q'] = pd.PeriodIndex(ev.switch, freq='Q').astype(str)
ev['ik'] = ev.firm + '|' + ev.cat
ev['kt'] = ev.cat + '|' + ev.q

# days since this firm's previous flagship release (NaN for each firm's first event)
uniq = ev.drop_duplicates('event')[['event', 'firm', 'switch']].sort_values(['firm', 'switch'])
uniq['days_since_prev'] = uniq.groupby('firm').switch.diff().dt.days
ev = ev.merge(uniq[['event', 'days_since_prev']], on='event', how='left')
ev['log_days'] = np.log(ev.days_since_prev)

print(f'events={ev.event.nunique()}  obs={len(ev)}  firms={ev.firm.nunique()}  '
      f'quarters={ev.q.nunique()}')


def fit(df, X_cols, fe_groups, noise=False):
    y = df.dtheta.values
    X = df[X_cols].values.astype(float)
    n = len(df)
    n_groups_dof = sum(df[g].nunique() for g in fe_groups) - len(fe_groups) + 1 if fe_groups else 0
    groups = [df[g].values for g in fe_groups]
    noise_cov = noise_var = None
    if noise:
        dfac = (n - n_groups_dof) / n
        s2_old = df.v_old.mean() * dfac
        s2_tot = (df.v_old.mean() + df.v_front.mean()) * dfac
        noise_cov = np.zeros(len(X_cols)); noise_cov[0] = s2_old
        noise_var = np.zeros((len(X_cols), len(X_cols))); noise_var[0, 0] = s2_tot
    beta, se, beta_c, resid, dfac = fe_regress(y, X, groups, noise_cov=noise_cov,
                                                noise_var=noise_var, cluster=df.event.values)
    out = dict(n=n, n_ev=df.event.nunique())
    for j, c in enumerate(X_cols):
        out[f'beta_{c}'] = beta[j]
        out[f'se_{c}'] = se[j]
        out[f't_{c}'] = beta[j] / se[j]
        if beta_c is not None:
            out[f'beta_corr_{c}'] = beta_c[j]
    return out


cols = {}
cols['(A) Basic'] = fit(ev, ['gap'], [])
cols['(B) Model 1: firm+cat+time FE'] = fit(ev, ['gap'], ['firm', 'cat', 'q'])
cols['(C) Model 2: firm×cat FE + time FE'] = fit(ev, ['gap'], ['ik', 'q'])
cols['(D) Model 3: firm×cat FE + cat×time FE'] = fit(ev, ['gap'], ['ik', 'kt'])

evs = ev.dropna(subset=['log_days']).copy()
cols['(E) + scaling (log days since release)'] = fit(evs, ['gap', 'log_days'], ['ik', 'kt'])
cols['(F) Full (+ noise correction)'] = fit(evs, ['gap', 'log_days'], ['ik', 'kt'], noise=True)

# ---------------------------------------------------------------- print ----
print(f'\n{"model":42s} {"n":>5s} {"n_ev":>5s} {"beta(gap)":>10s} {"se":>7s} {"t":>7s} '
      f'{"beta_corr":>10s} {"beta(logdays)":>14s}')
for name, r in cols.items():
    bc = r.get('beta_corr_gap')
    bld = r.get('beta_log_days')
    print(f'{name:42s} {r["n"]:5d} {r["n_ev"]:5d} {r["beta_gap"]:+10.3f} {r["se_gap"]:7.3f} '
          f'{r["beta_gap"]/r["se_gap"]:+7.2f} '
          f'{"" if bc is None else f"{bc:+10.3f}":>10s} '
          f'{"" if bld is None else f"{bld:+14.3f}":>14s}')

# --------------------------------------------------------------- LaTeX -----
def stars(t):
    at = abs(t)
    return '$^{***}$' if at > 2.58 else '$^{**}$' if at > 1.96 else '$^{*}$' if at > 1.645 else ''


latex_names = ['(A) Basic', '(B) Model 1', '(C) Model 2', '(D) Model 3 (main)',
               '(E) + Scaling', '(F) Full']
keys = list(cols.keys())
lines = []
lines.append(r'\begin{tabular}{lcccccc}')
lines.append(r'\toprule')
lines.append(' & ' + ' & '.join(latex_names) + r' \\')
lines.append(r'\midrule')

def fmt_beta(r, key='gap'):
    b, se, t = r[f'beta_{key}'], r[f'se_{key}'], r[f'beta_{key}'] / r[f'se_{key}']
    return f'{b:+.3f}{stars(t)}'

def fmt_se(r, key='gap'):
    return f'({r[f"se_{key}"]:.3f})'

lines.append(r'$\hat\beta$ (Gap) & ' + ' & '.join(fmt_beta(cols[k]) for k in keys) + r' \\')
lines.append(' & ' + ' & '.join(fmt_se(cols[k]) for k in keys) + r' \\')

corr_row = ['' if 'beta_corr_gap' not in cols[k] else f'{cols[k]["beta_corr_gap"]:+.3f}' for k in keys]
lines.append(r'\quad noise-corrected & ' + ' & '.join(corr_row) + r' \\')

ld_row = ['' if 'beta_log_days' not in cols[k] else
          f'{cols[k]["beta_log_days"]:+.3f} ({cols[k]["se_log_days"]:.3f})' for k in keys]
lines.append(r'$\hat\beta$ (log days since release) & ' + ' & '.join(ld_row) + r' \\')

lines.append(r'\midrule')
lines.append(r'Firm FE & -- & \checkmark & -- & -- & -- & -- \\')
lines.append(r'Category FE & -- & \checkmark & -- & -- & -- & -- \\')
lines.append(r'Time (quarter) FE & -- & \checkmark & \checkmark & -- & -- & -- \\')
lines.append(r'Firm$\times$Category FE & -- & -- & \checkmark & \checkmark & \checkmark & \checkmark \\')
lines.append(r'Category$\times$Time FE & -- & -- & -- & \checkmark & \checkmark & \checkmark \\')
lines.append(r'Noise correction & -- & -- & -- & -- & -- & \checkmark \\')
lines.append(r'\midrule')
lines.append(r'$N$ & ' + ' & '.join(str(cols[k]['n']) for k in keys) + r' \\')
lines.append(r'Events & ' + ' & '.join(str(cols[k]['n_ev']) for k in keys) + r' \\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

with open('experiments/table_evalchasing.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\nwrote experiments/table_evalchasing.tex')

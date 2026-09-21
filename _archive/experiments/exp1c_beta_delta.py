"""Exp 1c: single regression with a flagship/non-flagship interaction.

dtheta_{e,k} = alpha_e + gamma_k + beta*gap + delta*(nonflagship)*gap + eps

beta   = eval-chasing strength for flagship releases
delta  = EXTRA eval-chasing strength for non-flagship releases (beta+delta = their total)
Same event set as exp1b (all chained releases, top 8 firms, CORE categories).
"""
import pandas as pd, numpy as np

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']
MATURE_DAYS = 56

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d['org'] = d.organization.str.lower()
d = d[d.org.isin(FIRMS) & d.category.isin(CORE + ['overall'])]

snaps = np.sort(d.dt.unique())
rat = d.pivot_table(index=['dt','model_name'], columns='category', values='rating')
var = d.pivot_table(index=['dt','model_name'], columns='category', values='variance')

ov = d[d.category == 'overall'].dropna(subset=['rating'])
flag = (ov.sort_values('rating').groupby(['dt','org']).tail(1)
          .set_index(['dt','org']).model_name)
flagship_models = set(flag.unique())

first_seen = ov.groupby(['org','model_name'])['dt'].min().reset_index()

events = []
for firm in FIRMS:
    fm = first_seen[first_seen.org == firm].sort_values('dt')
    models = fm.model_name.tolist()
    dates = fm.dt.tolist()
    for i in range(1, len(models)):
        events.append(dict(firm=firm, old=models[i-1], new=models[i],
                           switch=dates[i], last_old=dates[i-1],
                           new_is_flagship=models[i] in flagship_models))

rows = []
for e in events:
    t_old = e['last_old']
    cand = [s for s in snaps if s >= e['switch'] + np.timedelta64(MATURE_DAYS, 'D')]
    if not cand: continue
    t_new = cand[0]
    try:
        th_old = rat.loc[(t_old, e['old'])][CORE]
        th_new = rat.loc[(t_new, e['new'])][CORE]
    except KeyError:
        continue
    if th_old.isna().any() or th_new.isna().any(): continue

    snap = rat.xs(t_old, level='dt')
    rivals = [flag.get((t_old, f)) for f in FIRMS if f != e['firm']]
    rivals = [m for m in rivals if m is not None and m in snap.index]
    if len(rivals) < 3: continue
    riv = snap.loc[rivals][CORE]
    frontier = riv.max(axis=0)

    for k in CORE:
        rows.append(dict(event=f"{e['firm']}:{e['old']}->{e['new']}@{e['switch'].strftime('%Y%m%d')}",
                         cat=k, dtheta=th_new[k]-th_old[k], gap=frontier[k]-th_old[k],
                         nonflag=0.0 if e['new_is_flagship'] else 1.0))

df = pd.DataFrame(rows)
df['gap_x_nonflag'] = df.gap * df.nonflag
print(f'n_events={df.event.nunique()}  obs={len(df)}')

def within(s, ev, ca):
    s = s.copy()
    for _ in range(50):
        s = s - s.groupby(ev).transform('mean')
        s = s - s.groupby(ca).transform('mean')
    return s

ev, ca = df.event.values, df.cat.values
Y  = within(pd.Series(df.dtheta.values), ev, ca)
X1 = within(pd.Series(df.gap.values), ev, ca)
X2 = within(pd.Series(df.gap_x_nonflag.values), ev, ca)

X = np.column_stack([X1.values, X2.values])
coef, res_, rank, sv = np.linalg.lstsq(X, Y.values, rcond=None)
beta, delta = coef
resid = Y.values - X @ coef

# cluster-robust SE by event (standard OLS cluster sandwich on the demeaned model)
XtX_inv = np.linalg.inv(X.T @ X)
meat = np.zeros((2,2))
for e_id in df.event.unique():
    mask = ev == e_id
    Xe = X[mask]; re = resid[mask]
    score = Xe.T @ re
    meat += np.outer(score, score)
cov = XtX_inv @ meat @ XtX_inv
se = np.sqrt(np.diag(cov))

print(f'beta  (flagship gap effect)          = {beta:+.4f}  (SE {se[0]:.4f}, t={beta/se[0]:+.2f})')
print(f'delta (extra non-flagship gap effect) = {delta:+.4f}  (SE {se[1]:.4f}, t={delta/se[1]:+.2f})')
print(f'beta + delta (total non-flagship effect) = {beta+delta:+.4f}')

# joint test-ish: correlation of the two demeaned regressors (collinearity check)
print(f'\ncorr(gap_demeaned, gap_x_nonflag_demeaned) = {np.corrcoef(X1,X2)[0,1]:.3f}')

"""Nested variance decomposition using the EXACT exp1 event definition:
  - 8 major firms, flagship-changeover events only, 56-day maturity window
  - CORE categories (the original 8)
  - gap = rivals' FLAGSHIP frontier (not population max) at t_old
  - noise correction uses BOTH v_old and v_front (as in exp1), at every nested stage

  dtheta_{e,k} = beta*gap_{e,k} + alpha_{firm,k} + lambda_{k, switch_quarter} + eps
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

events = []
for firm in FIRMS:
    ser = flag.xs(firm, level='org') if firm in flag.index.get_level_values('org') else None
    if ser is None: continue
    ser = ser.sort_index()
    prev_model, prev_date = None, None
    for dt_, m in ser.items():
        if prev_model is not None and m != prev_model:
            events.append(dict(firm=firm, old=prev_model, new=m, switch=dt_, last_old=prev_date))
        prev_model, prev_date = m, dt_

rows = []
for e in events:
    t_old = e['last_old']
    cand = [s for s in snaps if s >= e['switch'] + np.timedelta64(MATURE_DAYS, 'D')]
    if not cand: continue
    t_new = cand[0]
    try:
        th_old = rat.loc[(t_old, e['old'])][CORE]
        th_new = rat.loc[(t_new, e['new'])][CORE]
        v_old = var.loc[(t_old, e['old'])][CORE]
        v_new = var.loc[(t_new, e['new'])][CORE]
    except KeyError:
        continue
    if th_old.isna().any() or th_new.isna().any(): continue

    snap = rat.xs(t_old, level='dt')
    rivals = [flag.get((t_old, f)) for f in FIRMS if f != e['firm']]
    rivals = [m for m in rivals if m is not None and m in snap.index]
    if len(rivals) < 3: continue
    riv = snap.loc[rivals][CORE]
    frontier = riv.max(axis=0)
    v_front = np.array([var.loc[(t_old, riv[k].idxmax())][k] for k in CORE])

    switch_q = pd.Period(e['switch'], freq='Q')
    for j, k in enumerate(CORE):
        rows.append(dict(event=f"{e['firm']}:{e['old']}->{e['new']}@{e['switch'].strftime('%Y%m%d')}",
                         firm=e['firm'], cat=k, dtheta=th_new[k]-th_old[k],
                         gap=frontier[k]-th_old[k], v_old=v_old[k], v_front=v_front[j],
                         switch_q=switch_q))

df = pd.DataFrame(rows)
df['firm_cat'] = df.firm + '::' + df.cat
df['cat_q'] = df.cat + '::' + df.switch_q.astype(str)
print(f'events={df.event.nunique()}  obs={len(df)}  firm x cat groups={df.firm_cat.nunique()}  cat x quarter groups={df.cat_q.nunique()}')

Y = df.dtheta.values.astype(float)
G = df.gap.values.astype(float)
TSS = np.sum((Y - Y.mean())**2)
n = len(df)

def fit_corrected(Yin, Gin, dfac):
    cov = np.mean((Gin-Gin.mean())*(Yin-Yin.mean()))
    vg = np.mean((Gin-Gin.mean())**2)
    s2_old = df.v_old.mean()*dfac
    s2_tot = (df.v_old.mean() + df.v_front.mean())*dfac
    b = (cov - s2_old) / (vg - s2_tot)
    a = Yin.mean() - b*Gin.mean()
    resid = Yin - (a + b*Gin)
    return b, np.sum(resid**2)

# Model 1: no FE
b1, SSR1 = fit_corrected(Y, G, dfac=1.0)
R2_1 = 1 - SSR1/TSS

# Model 2: + firm x cat FE
Yd2 = pd.Series(Y) - pd.Series(Y).groupby(df.firm_cat.values).transform('mean')
Gd2 = pd.Series(G) - pd.Series(G).groupby(df.firm_cat.values).transform('mean')
n_fc = df.firm_cat.nunique()
dfac2 = (n - n_fc)/n
b2, SSR2 = fit_corrected(Yd2.values, Gd2.values, dfac=dfac2)
R2_2 = 1 - SSR2/TSS

# Model 3: + cat x quarter FE
Yd3, Gd3 = pd.Series(Y.copy()), pd.Series(G.copy())
for _ in range(100):
    Yd3 = Yd3 - Yd3.groupby(df.firm_cat.values).transform('mean')
    Yd3 = Yd3 - Yd3.groupby(df.cat_q.values).transform('mean')
    Gd3 = Gd3 - Gd3.groupby(df.firm_cat.values).transform('mean')
    Gd3 = Gd3 - Gd3.groupby(df.cat_q.values).transform('mean')
n_cq = df.cat_q.nunique()
dfac3 = max((n - n_fc - n_cq + df.cat.nunique())/n, 0.05)
b3, SSR3 = fit_corrected(Yd3.values, Gd3.values, dfac=dfac3)
R2_3 = 1 - SSR3/TSS

print(f'\nModel 1 (gap only, noise-corrected):     beta={b1:+.4f}   R2={R2_1:.4f}')
print(f'Model 2 (+ firm x cat FE, alpha):         beta={b2:+.4f}   R2={R2_2:.4f}   (+{R2_2-R2_1:.4f})')
print(f'Model 3 (+ cat x quarter FE, lambda):     beta={b3:+.4f}   R2={R2_3:.4f}   (+{R2_3-R2_2:.4f})')
print(f'Residual:                                                 {1-R2_3:.4f}')

print(f'\nShare of total variance explained at each step:')
print(f'  gap (catch-up)         : {R2_1:.1%}')
print(f'  + alpha (persistent)   : {R2_2-R2_1:.1%}')
print(f'  + lambda (common tech) : {R2_3-R2_2:.1%}')
print(f'  residual                : {1-R2_3:.1%}')

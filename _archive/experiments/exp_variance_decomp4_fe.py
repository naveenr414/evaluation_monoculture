"""Extract the actual alpha_{firm,cat} and lambda_{cat,quarter} fixed-effect VALUES
from the model 3 fit (not just R^2), using backfitting on R = dtheta - beta*gap.
"""
import pandas as pd, numpy as np

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']
MATURE_DAYS = 56
BETA = 0.3233  # from model 3

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d['org'] = d.organization.str.lower()
d = d[d.org.isin(FIRMS) & d.category.isin(CORE + ['overall'])]

snaps = np.sort(d.dt.unique())
rat = d.pivot_table(index=['dt','model_name'], columns='category', values='rating')
var = d.pivot_table(index=['dt','model_name'], columns='category', values='variance')
ov = d[d.category == 'overall'].dropna(subset=['rating'])
flag = (ov.sort_values('rating').groupby(['dt','org']).tail(1).set_index(['dt','org']).model_name)

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
    except KeyError:
        continue
    if th_old.isna().any() or th_new.isna().any(): continue
    snap = rat.xs(t_old, level='dt')
    rivals = [flag.get((t_old, f)) for f in FIRMS if f != e['firm']]
    rivals = [m for m in rivals if m is not None and m in snap.index]
    if len(rivals) < 3: continue
    riv = snap.loc[rivals][CORE]
    frontier = riv.max(axis=0)
    switch_q = pd.Period(e['switch'], freq='Q')
    for k in CORE:
        rows.append(dict(firm=e['firm'], cat=k, dtheta=th_new[k]-th_old[k],
                         gap=frontier[k]-th_old[k], switch_q=switch_q))

df = pd.DataFrame(rows)
df['firm_cat'] = df.firm + '::' + df.cat
df['cat_q'] = df.cat + '::' + df.switch_q.astype(str)
df['R'] = df.dtheta - BETA*df.gap

# backfitting for two-way FE (alpha per firm x cat, lambda per cat x quarter)
alpha = pd.Series(0.0, index=df.firm_cat.unique())
lam = pd.Series(0.0, index=df.cat_q.unique())
for _ in range(200):
    resid = df.R.values - lam.loc[df.cat_q].values
    alpha = pd.Series(resid, index=df.index).groupby(df.firm_cat).mean().reindex(alpha.index)
    resid2 = df.R.values - alpha.loc[df.firm_cat].values
    lam = pd.Series(resid2, index=df.index).groupby(df.cat_q).mean().reindex(lam.index)

print(f'alpha: mean={alpha.mean():.2f} std={alpha.std():.2f} min={alpha.min():.2f} max={alpha.max():.2f}')
print(f'lambda: mean={lam.mean():.2f} std={lam.std():.2f} min={lam.min():.2f} max={lam.max():.2f}')
print(f'raw dtheta: mean={df.dtheta.mean():.2f} std={df.dtheta.std():.2f}')
print()

print('=== ALPHA (persistent firm x category improvement-rate advantage), sorted ===')
a_df = alpha.reset_index()
a_df.columns = ['firm_cat','alpha']
a_df[['firm','cat']] = a_df.firm_cat.str.split('::', expand=True)
n_obs = df.groupby('firm_cat').size()
a_df['n_events'] = a_df.firm_cat.map(n_obs)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)
print(a_df.sort_values('alpha', ascending=False)[['firm','cat','alpha','n_events']].to_string(index=False))

print()
print('=== LAMBDA (common tech-progress shock, by category x quarter), sorted, |lambda|>10 ===')
l_df = lam.reset_index()
l_df.columns = ['cat_q','lambda']
l_df[['cat','quarter']] = l_df.cat_q.str.split('::', expand=True)
n_obs2 = df.groupby('cat_q').size()
l_df['n_events'] = l_df.cat_q.map(n_obs2)
big = l_df[l_df['lambda'].abs() > 10].sort_values('lambda', ascending=False)
print(big[['cat','quarter','lambda','n_events']].to_string(index=False))

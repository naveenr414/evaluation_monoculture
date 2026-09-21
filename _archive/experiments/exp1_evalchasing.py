"""Exp 1: eval-chasing vs comparative advantage.

Event = a firm's flagship changeover. For each event e and category k:
    dtheta_{e,k} = theta_new,k - theta_old,k
    gap_{e,k}    = max_{j != i} theta_j,k - theta_old,k    (rivals' frontier at the old date)
Regress dtheta on gap with event FE + category FE.
  beta > 0  => firms move toward where they lag on the shared board (eval-chasing)
  beta < 0  => firms deepen existing strengths (comparative advantage)

Mean-reversion confound: gap and dtheta both carry -eps_old (sampling noise in the
outgoing flagship's BT estimate), inducing spurious beta > 0. Corrected using the
per-observation BT variances shipped with the leaderboard.
"""
import pandas as pd, numpy as np

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']
MATURE_DAYS = 56   # let a new model accumulate votes before measuring its profile

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d['org'] = d.organization.str.lower()
d = d[d.org.isin(FIRMS) & d.category.isin(CORE + ['overall'])]

snaps = np.sort(d.dt.unique())
rat = d.pivot_table(index=['dt','model_name'], columns='category', values='rating')
var = d.pivot_table(index=['dt','model_name'], columns='category', values='variance')
org = d.drop_duplicates('model_name').set_index('model_name').org

# flagship per firm per snapshot = highest 'overall' rating
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
            events.append(dict(firm=firm, old=prev_model, new=m,
                               switch=dt_, last_old=prev_date))
        prev_model, prev_date = m, dt_

rows = []
for e in events:
    t_old = e['last_old']
    # mature measurement date for the new flagship
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

    # rivals' frontier at t_old, using each other firm's flagship
    snap = rat.xs(t_old, level='dt')
    rivals = [flag.get((t_old, f)) for f in FIRMS if f != e['firm']]
    rivals = [m for m in rivals if m is not None and m in snap.index]
    if len(rivals) < 3: continue
    riv = snap.loc[rivals][CORE]
    frontier = riv.max(axis=0)
    v_front = var.xs(t_old, level='dt').loc[riv.idxmax(axis=0).values].values.diagonal() \
        if False else np.array([var.loc[(t_old, riv[k].idxmax())][k] for k in CORE])

    for j, k in enumerate(CORE):
        rows.append(dict(event=f"{e['firm']}:{e['old']}->{e['new']}@{e['switch'].strftime('%Y%m%d')}", firm=e['firm'],
                         cat=k, dtheta=th_new[k]-th_old[k],
                         gap=frontier[k]-th_old[k],
                         v_old=v_old[k], v_new=v_new[k], v_front=v_front[j],
                         switch=e['switch']))

df = pd.DataFrame(rows)
print(f'events={df.event.nunique()}  firms={df.firm.nunique()}  obs={len(df)}')
print(df.groupby('firm').event.nunique().to_string())

def demean(s, by):
    return s - s.groupby(by).transform('mean')

# two-way within transformation (event FE + category FE)
g = df.gap.values.copy(); y = df.dtheta.values.copy()
G = pd.Series(g, index=df.index); Y = pd.Series(y, index=df.index)
for _ in range(50):
    G = demean(demean(G, df.event), df.cat)
    Y = demean(demean(Y, df.event), df.cat)

cov = np.mean(G*Y); vg = np.mean(G*G)
beta_naive = cov/vg

# noise correction: gap carries -eps_old and dtheta carries -eps_old
# within-transform shrinks the noise variance by the FE df factor
n, n_ev, n_cat = len(df), df.event.nunique(), df.cat.nunique()
dfac = (n - n_ev - n_cat + 1)/n
s2_old = df.v_old.mean()*dfac
s2_tot = (df.v_old.mean() + df.v_front.mean())*dfac
beta_corr = (cov - s2_old)/(vg - s2_tot)

# cluster-robust SE by event
res = Y - beta_naive*G
sc = pd.DataFrame({'g': G, 'r': res, 'e': df.event}).groupby('e').apply(
    lambda x: (x.g*x.r).sum(), include_groups=False)
se = np.sqrt((sc**2).sum())/ (vg*n)
print(f'\nnaive     beta = {beta_naive:+.4f}   (cluster SE {se:.4f}, t={beta_naive/se:+.2f})')
print(f'noise-corrected beta = {beta_corr:+.4f}')
print(f'  mean sampling var: old={df.v_old.mean():.1f} frontier={df.v_front.mean():.1f}'
      f' | var(gap) within = {vg:.1f}  -> noise is {100*s2_tot/vg:.0f}% of gap variation')

# split-half sanity: early vs late events
df['half'] = np.where(df.switch < df.switch.median(), 'early', 'late')
for h, sub in df.groupby('half'):
    Gh = pd.Series(sub.gap.values); Yh = pd.Series(sub.dtheta.values)
    ev, ca = sub.event.values, sub.cat.values
    for _ in range(50):
        Gh = Gh - Gh.groupby(ev).transform('mean'); Gh = Gh - Gh.groupby(ca).transform('mean')
        Yh = Yh - Yh.groupby(ev).transform('mean'); Yh = Yh - Yh.groupby(ca).transform('mean')
    b = np.mean(Gh*Yh)/np.mean(Gh*Gh)
    dfac_h = (len(sub)-sub.event.nunique()-sub.cat.nunique()+1)/len(sub)
    bc = (np.mean(Gh*Yh)-sub.v_old.mean()*dfac_h)/(np.mean(Gh*Gh)-(sub.v_old.mean()+sub.v_front.mean())*dfac_h)
    print(f'  {h:5s} n_events={sub.event.nunique():3d}  naive={b:+.3f}  corrected={bc:+.3f}')

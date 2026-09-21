"""Exp 1b: eval-chasing for NON-flagship models at top firms.

Same setup as exp1_evalchasing.py, but instead of restricting events to
flagship-changeovers, we look at ALL model releases by the 8 top firms and
split into flagship vs non-flagship NEW models.

Event = any firm's new model release (any tier), matched against the
immediately preceding model release from that same firm (whatever tier).
  dtheta_{e,k} = theta_new,k - theta_prev,k
  gap_{e,k}    = rivals' flagship frontier at t_old - theta_prev,k
We then split events by whether the NEW model ever became a flagship
(top 'overall' rating for its firm at some snapshot) or not, and fit beta
separately on each subset.
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

# flagship = highest 'overall' rating per firm per snapshot
flag = (ov.sort_values('rating').groupby(['dt','org']).tail(1)
          .set_index(['dt','org']).model_name)
flagship_models = set(flag.unique())

# every model's first-appearance date per firm (any tier, using 'overall' presence)
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

    for j, k in enumerate(CORE):
        rows.append(dict(event=f"{e['firm']}:{e['old']}->{e['new']}@{e['switch'].strftime('%Y%m%d')}",
                         firm=e['firm'], cat=k, dtheta=th_new[k]-th_old[k],
                         gap=frontier[k]-th_old[k], v_old=v_old[k], v_front=v_front[j],
                         switch=e['switch'], new_is_flagship=e['new_is_flagship']))

df = pd.DataFrame(rows)
print(f'total events={df.event.nunique()}  obs={len(df)}')
print(df.groupby('new_is_flagship').event.nunique())
print()

def fit_beta(sub):
    n = len(sub)
    ev, ca = sub.event.values, sub.cat.values
    G = pd.Series(sub.gap.values); Y = pd.Series(sub.dtheta.values)
    for _ in range(50):
        G = G - G.groupby(ev).transform('mean'); G = G - G.groupby(ca).transform('mean')
        Y = Y - Y.groupby(ev).transform('mean'); Y = Y - Y.groupby(ca).transform('mean')
    cov = np.mean(G*Y); vg = np.mean(G*G)
    beta_naive = cov/vg
    n_ev, n_cat = sub.event.nunique(), sub.cat.nunique()
    dfac = (n - n_ev - n_cat + 1)/n
    s2_old = sub.v_old.mean()*dfac
    s2_tot = (sub.v_old.mean() + sub.v_front.mean())*dfac
    beta_corr = (cov - s2_old)/(vg - s2_tot)
    res = Y - beta_naive*G
    sc = pd.DataFrame({'g': G, 'r': res, 'e': ev}).groupby('e').apply(
        lambda x: (x.g*x.r).sum(), include_groups=False)
    se = np.sqrt((sc**2).sum())/(vg*n)
    return beta_naive, beta_corr, se, n_ev, n

for label, sub in [('NON-flagship new models', df[~df.new_is_flagship]),
                    ('flagship new models (replication)', df[df.new_is_flagship])]:
    if len(sub) < 20:
        print(f'{label}: too few obs ({len(sub)})'); continue
    bn, bc, se, n_ev, n = fit_beta(sub)
    print(f'{label}: n_events={n_ev} obs={n}')
    print(f'  naive beta = {bn:+.4f}  (cluster SE {se:.4f}, t={bn/se:+.2f})')
    print(f'  noise-corrected beta = {bc:+.4f}')
    print()

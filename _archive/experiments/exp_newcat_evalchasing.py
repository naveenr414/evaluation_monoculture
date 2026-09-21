"""Eval-chasing beta for NEW categories (launched 2025-11-05) vs mature CORE
categories, matched to the same post-launch time window, same events, same firms.
"""
import pandas as pd, numpy as np

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
NEW = ['expert','industry_life_and_physical_and_social_science','industry_medicine_and_healthcare',
       'industry_writing_and_literature_and_language','industry_business_and_management_and_financial_operations',
       'industry_entertainment_and_sports_and_media','industry_legal_and_government',
       'industry_software_and_it_services','industry_mathematical']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']
MATURE_DAYS = 21
LAUNCH = pd.Timestamp('2025-11-05')

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d['org'] = d.organization.str.lower()
d = d[d.org.isin(FIRMS) & d.category.isin(CORE + NEW + ['overall'])]

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
        if prev_model is not None and m != prev_model and prev_date >= LAUNCH:
            events.append(dict(firm=firm, old=prev_model, new=m, switch=dt_, last_old=prev_date))
        prev_model, prev_date = m, dt_

print(f'post-launch flagship-changeover events: {len(events)}')

rows = []
for e in events:
    t_old = e['last_old']
    cand = [s for s in snaps if s >= e['switch'] + np.timedelta64(MATURE_DAYS, 'D')]
    if not cand: continue
    t_new = cand[0]

    snap = rat.xs(t_old, level='dt')
    rivals = [flag.get((t_old, f)) for f in FIRMS if f != e['firm']]
    rivals = [m for m in rivals if m is not None and m in snap.index]
    if len(rivals) < 3: continue
    riv = snap.loc[rivals]

    for group_name, cats in [('CORE', CORE), ('NEW', NEW)]:
        try:
            th_old = rat.loc[(t_old, e['old'])][cats]
            th_new = rat.loc[(t_new, e['new'])][cats]
            v_old = var.loc[(t_old, e['old'])][cats]
        except KeyError:
            continue
        if th_old.isna().any() or th_new.isna().any(): continue
        riv_c = riv[cats]
        frontier = riv_c.max(axis=0)
        for k in cats:
            rows.append(dict(event=f"{e['firm']}:{e['old']}->{e['new']}@{e['switch'].strftime('%Y%m%d')}",
                             group=group_name, cat=k, dtheta=th_new[k]-th_old[k],
                             gap=frontier[k]-th_old[k], v_old=v_old[k]))

df = pd.DataFrame(rows)
print(f'obs: {len(df)}')
print(df.groupby('group').event.nunique())

def within(s, ev, ca):
    s = s.copy()
    for _ in range(50):
        s = s - s.groupby(ev).transform('mean')
        s = s - s.groupby(ca).transform('mean')
    return s

for grp in ['CORE','NEW']:
    sub = df[df.group == grp]
    if sub.event.nunique() < 3:
        print(f'{grp}: too few events ({sub.event.nunique()})'); continue
    ev, ca = sub.event.values, sub.cat.values
    Y = within(pd.Series(sub.dtheta.values), ev, ca)
    G = within(pd.Series(sub.gap.values), ev, ca)
    cov = np.mean(G*Y); vg = np.mean(G*G)
    beta = cov/vg
    resid = Y.values - beta*G.values
    sc = pd.DataFrame({'g':G.values,'r':resid,'e':ev}).groupby('e').apply(lambda x:(x.g*x.r).sum(), include_groups=False)
    se = np.sqrt((sc**2).sum())/(vg*len(sub))
    print(f'{grp}: n_events={sub.event.nunique()} obs={len(sub)}  beta={beta:+.4f}  SE={se:.4f}  t={beta/se:+.2f}')

"""Heterogeneous gap-chasing estimates for Experiment 1.

Uses the same flagship-event construction as exp1_evalchasing.py, then reports
gap-response slopes by firm, firm group, category, and (where metadata permits)
open-model parameter size.  Firm-level estimates use event and category fixed
effects within the relevant subset; they are descriptive because some firms have
few events.  Noise-corrected estimates use the same sampling-variance adjustment
as Experiment 1.
"""
import numpy as np
import pandas as pd

CORE = ['coding','math','hard_prompts','instruction_following',
        'creative_writing','longer_query','multi_turn','english']
FIRMS = ['openai','anthropic','google','meta','mistral','xai','deepseek','alibaba']
OPEN = {'meta', 'mistral', 'deepseek', 'alibaba'}
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
    if firm not in flag.index.get_level_values('org'):
        continue
    ser = flag.xs(firm, level='org').sort_index()
    prev_model = prev_date = None
    for dt_, m in ser.items():
        if prev_model is not None and m != prev_model:
            events.append(dict(firm=firm, old=prev_model, new=m,
                               switch=dt_, last_old=prev_date))
        prev_model, prev_date = m, dt_

rows = []
for e in events:
    cand = [s for s in snaps if s >= e['switch'] + np.timedelta64(MATURE_DAYS, 'D')]
    if not cand:
        continue
    t_new = cand[0]
    try:
        old = rat.loc[(e['last_old'], e['old'])][CORE]
        new = rat.loc[(t_new, e['new'])][CORE]
        vold = var.loc[(e['last_old'], e['old'])][CORE]
    except KeyError:
        continue
    if old.isna().any() or new.isna().any():
        continue
    try:
        snap = rat.xs(e['last_old'], level='dt')
        rivals = [flag.get((e['last_old'], f)) for f in FIRMS if f != e['firm']]
        rivals = [m for m in rivals if m is not None and m in snap.index]
        if len(rivals) < 3:
            continue
        riv = snap.loc[rivals][CORE]
        frontier = riv.max(axis=0)
        vfront = np.array([var.loc[(e['last_old'], riv[k].idxmax())][k] for k in CORE])
    except (KeyError, TypeError):
        continue
    for j, k in enumerate(CORE):
        rows.append(dict(event=f"{e['firm']}:{e['old']}->{e['new']}@{e['switch'].strftime('%Y%m%d')}",
                         firm=e['firm'], cat=k, dtheta=new[k]-old[k],
                         gap=frontier[k]-old[k], v_old=vold[k],
                         v_front=vfront[j], switch=e['switch'],
                         old_model=e['old'], new_model=e['new']))
df = pd.DataFrame(rows)
df['group'] = np.where(df.firm.isin(OPEN), 'open', 'closed')

def demean(x, by):
    return x - x.groupby(by).transform('mean')

def estimate(z):
    """FE slope plus same noise correction; return NaN if underidentified."""
    if z.event.nunique() < 2 or z.cat.nunique() < 2 or len(z) < 2:
        return np.nan, np.nan, np.nan
    G, Y = z.gap.copy(), z.dtheta.copy()
    for _ in range(60):
        G = demean(demean(G, z.event), z.cat)
        Y = demean(demean(Y, z.event), z.cat)
    vg, cov = np.mean(G*G), np.mean(G*Y)
    if vg <= 1e-10:
        return np.nan, np.nan, np.nan
    fac = (len(z)-z.event.nunique()-z.cat.nunique()+1)/len(z)
    corr = (cov-z.v_old.mean()*fac) / (vg-(z.v_old.mean()+z.v_front.mean())*fac)
    return cov/vg, corr, len(z)

print(f'events={df.event.nunique()} firms={df.firm.nunique()} observations={len(df)}')
print('\n=== by firm (descriptive; corrected uses sampling variances) ===')
print(f'{"firm":14s} {"events":>7s} {"naive":>9s} {"corrected":>11s}')
for firm, z in df.groupby('firm'):
    b, bc, _ = estimate(z)
    n = z.event.nunique()
    print(f'{firm:14s} {n:7d} {b:9.3f} {bc:11.3f}')

print('\n=== open vs closed ===')
for group, z in df.groupby('group'):
    b, bc, n = estimate(z)
    print(f'{group:8s} events={z.event.nunique():3d} obs={len(z):4d} '
          f'naive={b:+.3f} corrected={bc:+.3f}')

print('\n=== category responses (joint event/category FE interactions) ===')
# Estimate all gap x category slopes jointly after removing event and category
# fixed effects.  This avoids the invalid operation of fitting a separate
# one-category regression with event FE (one observation per event in that case).
G0 = pd.Series(df.gap.values, index=df.index)
Y0 = pd.Series(df.dtheta.values, index=df.index)
def within(s):
    q = s.copy()
    for _ in range(60):
        q = demean(demean(q, df.event), df.cat)
    return q.values
yw = within(Y0)
X = np.column_stack([within(pd.Series(df.gap.values * (df.cat.values == c), index=df.index))
                      for c in CORE])
coefs = np.linalg.lstsq(X, yw, rcond=None)[0]
for c, b in zip(CORE, coefs):
    print(f'{c:24s} {df[df.cat == c].event.nunique():7d} {b:+.3f} {len(df[df.cat == c]):5d}')

print('\n=== firm x category raw slopes (descriptive; small cells) ===')
print('Each cell is the within-cell slope of improvement on the old gap; cells with '
      'few events are not reliable firm-specific estimates.')
mat = pd.DataFrame(index=sorted(df.firm.unique()), columns=CORE, dtype=float)
for (firm, cat), z in df.groupby(['firm', 'cat']):
    mat.loc[firm, cat] = np.cov(z.gap, z.dtheta, bias=True)[0, 1] / np.var(z.gap) \
        if np.var(z.gap) > 1e-10 else np.nan
print(mat.round(2).to_string())

# Closed-lab parameter metadata are not available.  For open models, use a
# deliberately rough name-based size parser (e.g. 7B, 70B, 8x22B); report it
# as exploratory rather than as a measured parameter count.
import re
def name_size(name):
    q = str(name).lower().replace('-', ' ')
    m = re.search(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*b', q)
    if m:
        return float(m.group(1)) * float(m.group(2))
    vals = re.findall(r'(\d+(?:\.\d+)?)\s*b\b', q)
    return max(map(float, vals)) if vals else np.nan

df['new_params_b_name'] = df.new_model.map(name_size)
op = df[(df.group == 'open') & df.new_params_b_name.notna()].copy()
if len(op):
    cut = op.new_params_b_name.median()
    print(f'\n=== open events split by name-derived new-model size (median={cut:.1f}B; exploratory) ===')
    for label, z in op.assign(size=np.where(op.new_params_b_name <= cut, 'small', 'large')).groupby('size'):
        b, bc, _ = estimate(z)
        print(f'{label:6s} events={z.event.nunique():3d} models={z.new_model.nunique():3d} '
              f'naive={b:+.3f} corrected={bc:+.3f}')
else:
    print('\nNo name-derived open-model parameter sizes were available.')

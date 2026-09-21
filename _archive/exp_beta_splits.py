"""Is gap-chasing the same in every period and at every model scale?

Part 1 splits the draft's event panel by calendar time.
Part 2 rebuilds the event panel at the level of a firm's model TIER rather than
its overall flagship, so that small and mid-size releases enter the sample.  A
tier flagship is the highest-'overall'-rated model of that tier for that firm at
a snapshot; a changeover is an event exactly as before.  The gap is always the
per-category frontier over rival firms' overall flagships, so the benchmark does
not move with the tier.

Specification throughout: firm x category and category x quarter fixed effects,
standard errors clustered on the release event.
"""
import re
import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (load_panel, build_events, flagships, fe_regress, CORE,
                    FRONTIER, MATURE_DAYS)

# separator-anchored so that 'gemini' does not match 'mini'
SMALL = re.compile(r'(?:^|[-_])(mini|flash|lite|nano|haiku|small|air)(?:$|[-_0-9])'
                   r'|[-_](1\.5|3|4|7|8|9|12)b(?:$|[-_])')
LARGE = re.compile(r'(?:^|[-_])(opus|pro|ultra|large|reasoner|thinking)(?:$|[-_0-9])'
                   r'|[-_](405|235|70|72)b(?:$|[-_])')


def tier(name):
    n = name.lower()
    if SMALL.search(n):
        return 'small'
    if LARGE.search(n):
        return 'large'
    return 'mid'


def fit(df, gcol='gap'):
    if df.event.nunique() < 6:
        return None
    y, X = df.dtheta.values, df[[gcol]].values.astype(float)
    groups = [df.ik.values, df.kt.values]
    n = len(df)
    dfac = (n - (df.ik.nunique() + df.kt.nunique() - 1)) / n
    nc = np.array([df.v_old.mean() * dfac])
    nv = np.array([[(df.v_old.mean() + df.v_front.mean()) * dfac]])
    try:
        beta, se, beta_c, resid, _, _ = fe_regress(y, X, groups, noise_cov=nc,
                                                noise_var=nv,
                                                cluster=df.event.values)
    except np.linalg.LinAlgError:
        return None
    return dict(beta=beta[0], se=se[0], t=beta[0] / se[0], corr=beta_c[0],
                n=n, events=df.event.nunique(), firms=df.firm.nunique())


def show(title, parts):
    print('\n' + '=' * 76)
    print(title)
    print('=' * 76)
    rows = []
    for lab, df in parts:
        r = fit(df)
        if r is None:
            print(f'  {lab:<26} too few events ({df.event.nunique()})')
            continue
        rows.append(dict(split=lab, **r))
    if rows:
        print(pd.DataFrame(rows).to_string(
            index=False, float_format=lambda x: f'{x:7.3f}'))


d = load_panel(cats=CORE)
ev = build_events(d, cats=CORE)
ev['q'] = pd.PeriodIndex(ev.switch, freq='Q').astype(str)
ev['ik'] = ev.firm + '|' + ev.cat
ev['kt'] = ev.cat + '|' + ev.q
ev['yr'] = pd.to_datetime(ev.switch).dt.year
print(f'baseline panel: {ev.event.nunique()} events, {len(ev)} obs')
print('  events per year:',
      ev.drop_duplicates('event').groupby('yr').size().to_dict())
show('PART 1.  BY PERIOD (flagship events, the draft panel)', [
    ('all', ev),
    ('2024-2025', ev[ev.yr <= 2025]),
    ('2025 only', ev[ev.yr == 2025]),
    ('2026 only', ev[ev.yr == 2026]),
])

# median split on the event date, which balances the two halves
uniq = ev.drop_duplicates('event')[['event', 'switch']].sort_values('switch')
cut = uniq.switch.median()
print(f'\n  median event date = {pd.Timestamp(cut).date()}')
show('PART 1b.  EARLY vs LATE HALF', [
    ('early half', ev[ev.switch <= cut]),
    ('late half', ev[ev.switch > cut]),
])

# --------------------------------------------------------------- part 2 ----
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
var = d.pivot_table(index=['dt', 'model_name'], columns='category', values='variance')
ov = d[d.category == 'overall'].dropna(subset=['rating'])
flag = flagships(d)
snaps = np.sort(d.dt.unique())

ov = ov.copy()
ov['tier'] = ov.model_name.map(tier)
# tier flagship: best-overall model of that tier for that firm at each snapshot
tf = (ov.sort_values('rating').groupby(['dt', 'org', 'tier']).tail(1)
        .set_index(['dt', 'org', 'tier']).model_name)

events = []
for firm in FRONTIER:
    for tr in ('small', 'mid', 'large'):
        try:
            ser = tf.xs((firm, tr), level=('org', 'tier')).sort_index()
        except KeyError:
            continue
        prev_m, prev_t = None, None
        for t, m in ser.items():
            if prev_m is not None and m != prev_m:
                events.append((firm, tr, prev_m, m, t, prev_t))
            prev_m, prev_t = m, t

rows = []
for firm, tr, old, new, switch, t_old in events:
    cand = [s for s in snaps if s >= switch + np.timedelta64(MATURE_DAYS, 'D')]
    if not cand:
        continue
    t_new = cand[0]
    try:
        th_old, th_new = rat.loc[(t_old, old)][CORE], rat.loc[(t_new, new)][CORE]
        v_old = var.loc[(t_old, old)][CORE]
    except KeyError:
        continue
    if th_old.isna().any() or th_new.isna().any():
        continue
    snap = rat.xs(t_old, level='dt')
    rivals = [flag.get((t_old, f)) for f in FRONTIER if f != firm]
    rivals = [m for m in rivals if m is not None and m in snap.index]
    riv = snap.loc[rivals][CORE].dropna()
    if len(riv) < 3:
        continue
    front = riv.max(axis=0)
    vfr = np.array([var.loc[(t_old, riv[k].idxmax())][k] for k in CORE])
    for j, k in enumerate(CORE):
        rows.append(dict(
            event=f'{firm}:{tr}:{old}->{new}@{pd.Timestamp(switch).strftime("%Y%m%d")}',
            firm=firm, tier=tr, cat=k, switch=switch,
            dtheta=th_new[k] - th_old[k], gap=front[k] - th_old[k],
            v_old=v_old[k], v_front=vfr[j]))

et = pd.DataFrame(rows)
et['q'] = pd.PeriodIndex(et.switch, freq='Q').astype(str)
et['ik'] = et.firm + '|' + et.cat
et['kt'] = et.cat + '|' + et.q
print('\n' + '-' * 76)
print('tier-level event panel')
u = et.drop_duplicates('event')
print('  events per tier:', u.groupby('tier').size().to_dict())
print('  firms per tier :', u.groupby('tier').firm.nunique().to_dict())
print('  mean gap by tier:',
      {k: round(v, 1) for k, v in et.groupby('tier').gap.mean().items()})
for tr in ('small', 'mid', 'large'):
    ms = sorted(set(u[u.tier == tr].event.str.split(':').str[2].str.split('->').str[1]))
    print(f'  {tr:<6} incoming models (n={len(ms)}): {ms[:8]}')

show('PART 2.  BY MODEL TIER (tier-level events)', [
    ('all tiers', et),
    ('small', et[et.tier == 'small']),
    ('mid', et[et.tier == 'mid']),
    ('large', et[et.tier == 'large']),
])

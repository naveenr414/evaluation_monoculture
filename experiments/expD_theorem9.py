"""Experiment D: Theorem 9 test on the staggered 2024 category launches.

Theorem 9: firms with parallel F respond IDENTICALLY to the same (c,w) perturbation.
So divergent responses certify divergent technology.

A category entering the board is a discrete change in w. We cannot observe the
treated dimension pre-launch (no rating exists before the category does), so we
measure the response on the CONTROL categories, which have full pre-periods.

In two-way residual space the common component of the response is removed by
construction, so "all firms respond identically" == "all residual responses are
zero". The test is whether cross-firm dispersion of the residual response around a
shock exceeds what is seen in placebo windows with no launch.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, two_way_resid, firm_profiles, FRONTIER

SHOCKS = {'coding': '2024-04-03', 'multi_turn': '2024-06-29',
          'math+instruction_following': '2024-07-08',
          'creative_writing': '2024-10-23',
          'industry+expert (8 cats)': '2025-11-05'}

ALWAYS = ['english', 'longer_query', 'chinese', 'french']   # live from 2024-03/04
PRE_W, GAP_W, POST_W = 8, 4, 16    # weeks


def response(d, cats, t0):
    """Cross-firm dispersion of the residual profile CHANGE around date t0."""
    t0 = pd.Timestamp(t0)
    pre = [s for s in sorted(d.dt.unique())
           if t0 - pd.Timedelta(weeks=PRE_W) <= s < t0]
    post = [s for s in sorted(d.dt.unique())
            if t0 + pd.Timedelta(weeks=GAP_W) <= s < t0 + pd.Timedelta(weeks=POST_W)]
    if len(pre) < 2 or len(post) < 2:
        return None

    def avg(snaps):
        acc, vac, cnt = {}, {}, {}
        for s in snaps:
            P, V, names = firm_profiles(d, cats, s)
            if len(P) < 4:
                continue
            R = two_way_resid(P)
            for i, f in enumerate(names):
                acc[f] = acc.get(f, 0) + R[i]; vac[f] = vac.get(f, 0) + V[i]
                cnt[f] = cnt.get(f, 0) + 1
        return ({f: acc[f] / cnt[f] for f in acc},
                {f: vac[f] / cnt[f] ** 2 for f in vac})   # averaging shrinks noise

    a_pre, v_pre = avg(pre)
    a_post, v_post = avg(post)
    firms = sorted(set(a_pre) & set(a_post))
    if len(firms) < 4:
        return None
    D = np.array([a_post[f] - a_pre[f] for f in firms])
    Vn = np.array([v_post[f] + v_pre[f] for f in firms])
    stat = (D ** 2).mean() - np.nanmean(Vn)      # noise-corrected response magnitude
    return dict(stat=stat, raw=(D ** 2).mean(), noise=np.nanmean(Vn),
                n_firms=len(firms), firms=firms)


d = load_panel(cats=ALWAYS, since='2024-03-01')

# placebo distribution: dates with no category launch
launch_dates = pd.to_datetime(list(SHOCKS.values()))
all_dates = pd.to_datetime(sorted(d.dt.unique()))
placebo_dates = [t for t in all_dates
                 if (abs(t - launch_dates).min() > pd.Timedelta(weeks=10))
                 and t > all_dates[0] + pd.Timedelta(weeks=PRE_W)
                 and t < all_dates[-1] - pd.Timedelta(weeks=POST_W)]

pl = [response(d, ALWAYS, t) for t in placebo_dates]
pl = [p['stat'] for p in pl if p is not None]
print(f'placebo windows: n={len(pl)}  mean={np.mean(pl):.1f}  sd={np.std(pl):.1f}'
      f'  p90={np.percentile(pl,90):.1f}\n')

print(f'{"shock":30s} {"date":12s} {"firms":>5s} {"response":>9s} {"noise":>7s} {"z":>6s} {"p":>6s}')
res = []
for nm, dt in SHOCKS.items():
    r = response(d, ALWAYS, dt)
    if r is None:
        print(f'{nm:30s} {dt:12s}   -- insufficient coverage --'); continue
    z = (r['stat'] - np.mean(pl)) / np.std(pl)
    p = (np.sum(np.array(pl) >= r['stat']) + 1) / (len(pl) + 1)
    print(f'{nm:30s} {dt:12s} {r["n_firms"]:5d} {r["stat"]:9.1f} {r["noise"]:7.1f}'
          f' {z:+6.2f} {p:6.3f}')
    res.append((nm, r, z, p))

print('\nInterpretation:')
print('  response >> placebo  => firms respond DIFFERENTLY to the same w-shock')
print('                       => F_i are NOT parallel (Theorem 9 certifies divergent tech)')
print('  response ~ placebo   => no detectable heterogeneity; consistent with parallel F,')
print('                          or the test is underpowered at this firm count.')

# per-firm decomposition for the strongest shock
if res:
    nm, r, z, p = max(res, key=lambda x: x[2])
    print(f'\nper-firm response magnitudes for the strongest shock ({nm}):')
    t0 = pd.Timestamp(SHOCKS[nm])
    print(f'  firms present: {", ".join(r["firms"])}')

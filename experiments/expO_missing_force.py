"""Priority 1: find the missing homogenizing force behind the 2.56x steady-state
over-prediction (§3.3/§3.8).

Candidates tested here:
  (a) Rating-scale / voter-population drift: does an UNCHANGED model's category profile
      deviation shrink toward the pack over calendar time, even with zero real capability
      change? If yes, part of "homogenization" is a measurement artifact, not real
      convergence -- directly testable using long static-flagship spells in the panel.
  (b) Mechanical/ceiling compression: does cross-firm dispersion fall simply because mean
      capability rose (a bounded/near-ceiling rating scale compresses differences at the
      top), independent of any behavioral mechanism?
  (c) Continuous vs. at-event convergence: does a firm's profile deviation keep shrinking
      BETWEEN its own releases (i.e. while its flagship name is literally unchanged), at a
      rate comparable to the at-event decay from §3.3/§3.8? If so, whatever is happening
      is not really about the release/gap-chasing DECISION -- it is a background process.

(Imitation via teacher-following, tested in §3.2, is already known to explain only a small
slice of the residual variance and is not re-tested here.)
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, flagships, firm_profiles, two_way_resid, CORE, FRONTIER

d = load_panel(cats=CORE)
flag = flagships(d)
snaps = np.sort(d.dt.unique())
rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
var = d.pivot_table(index=['dt', 'model_name'], columns='category', values='variance')

# ---- (a) static-model drift test ----
print('=== (a) does an UNCHANGED flagship model drift toward the pack over calendar time? ===\n')

# per-snapshot two-way residual profile for every firm present that day
snap_resid = {}
for t in snaps:
    P, V, names = firm_profiles(d, CORE, t)
    if len(P) < 5:
        continue
    R = two_way_resid(P)
    snap_resid[pd.Timestamp(t)] = dict(zip(names, zip(R, V)))

# build static spells: ALL pairs of snapshot dates within a spell where the firm's
# flagship name is unchanged (not just adjacent), to get a spread of lags for an
# exponential-decay fit rather than being stuck at the ~3-day modal snapshot spacing
pairs = []
for firm in FRONTIER:
    if firm not in flag.index.get_level_values('org'):
        continue
    ser = flag.xs(firm, level='org').sort_index()
    dates, models = list(ser.index), list(ser.values)
    i = 0
    while i < len(dates):
        j = i
        while j + 1 < len(dates) and models[j + 1] == models[i]:
            j += 1
        spell = [pd.Timestamp(x) for x in dates[i:j + 1]]
        for a in range(len(spell)):
            for b in range(a + 1, len(spell)):
                t0, t1 = spell[a], spell[b]
                if t0 in snap_resid and t1 in snap_resid and firm in snap_resid[t0] and firm in snap_resid[t1]:
                    r0, v0 = snap_resid[t0][firm]
                    r1, v1 = snap_resid[t1][firm]
                    pairs.append(dict(firm=firm, model=models[i], t0=t0, t1=t1,
                                       lag=(t1 - t0).days, r0=r0, r1=r1, v0=v0, v1=v1))
        i = j + 1

rows = []
for p in pairs:
    for j, k in enumerate(CORE):
        rows.append(dict(firm=p['firm'], model=p['model'], lag=p['lag'],
                          r0=p['r0'][j], r1=p['r1'][j], v0=p['v0'][j], v1=p['v1'][j]))
sp = pd.DataFrame(rows)
print(f'static (unchanged-model) snapshot pairs: {len(pairs)} spell-pairs, '
      f'{len(sp)} (firm,category) obs, lag range [{sp.lag.min():.0f}, {sp.lag.max():.0f}] days')

# naive (uncorrected) rho by lag bin -- noise correction is unstable when var0 is small
# relative to sampling noise at short lags, so report naive and note the correction is
# conservative (attenuates rho toward 0, i.e. UNDER-states persistence if anything)
bins = [0, 7, 14, 30, 60, 120, 10000]
labels = ['0-7d', '7-14d', '14-30d', '30-60d', '60-120d', '120d+']
sp['bin'] = pd.cut(sp.lag, bins=bins, labels=labels, right=False)
print('\nnaive rho by lag bin (unchanged model -- any decay here is NOT explained by real updates):')
for b in labels:
    sub = sp[sp.bin == b]
    if len(sub) < 20:
        continue
    rho_b = np.mean(sub.r0 * sub.r1) / np.mean(sub.r0 ** 2)
    print(f'  {b:8s} n={len(sub):5d}  median lag={sub.lag.median():5.0f}d  rho={rho_b:+.3f}')

# exponential-decay fit: log(rho) roughly linear in lag if this is a real decay process
fit_rows = []
for b in labels:
    sub = sp[sp.bin == b]
    if len(sub) < 20:
        continue
    rho_b = np.mean(sub.r0 * sub.r1) / np.mean(sub.r0 ** 2)
    if rho_b > 0:
        fit_rows.append((sub.lag.median(), np.log(rho_b)))
if len(fit_rows) >= 3:
    lags_, logrho_ = zip(*fit_rows)
    slope, intercept = np.polyfit(lags_, logrho_, 1)
    per_day = np.exp(slope)
    med_release_spacing = 28
    implied_28d = per_day ** med_release_spacing
    print(f'\nfitted per-day decay = {per_day:.4f}  =>  implied rho over a {med_release_spacing}-day '
          f'window = {implied_28d:+.3f}')
    print(f'cf. at-release-event decay (§3.3 direct test): measured 0.390, '
          f'gap-chasing-alone-predicted 0.451')
else:
    implied_28d = np.nan
    print('\nnot enough lag bins with data for a decay fit')

print()
if np.isfinite(implied_28d) and implied_28d > 0.80:
    verdict_a = ('UNCHANGED models do NOT drift meaningfully even over a 28-day window -- '
                 'rating-scale/voter-population drift is NOT the missing force; convergence is '
                 'tied to real model updates, not measurement artifact.')
elif np.isfinite(implied_28d) and implied_28d < 0.60:
    verdict_a = ('UNCHANGED models DO drift toward the pack at a rate comparable to the '
                 'at-release decay -- part of "homogenization" may be a rating-scale/measurement '
                 'artifact, not real capability convergence. Serious candidate for the missing force.')
else:
    verdict_a = 'Ambiguous -- decay rate is intermediate or the fit is noisy; not a clean answer.'
print('>>> ' + verdict_a)

# ---- (b) mechanical/ceiling compression ----
print('\n=== (b) does dispersion fall simply because mean capability rose (ceiling effect)? ===\n')
obs = []
for t, snap in d.groupby('dt'):
    P, V, names = firm_profiles(d, CORE, t)
    if len(P) < 5:
        continue
    R = two_way_resid(P)
    nfac = (len(P) - 1) / len(P)
    obs.append(dict(date=t, level=P.mean(), raw=R.var(axis=0, ddof=1).mean(),
                     noise=np.nanmean(V) * nfac))
o = pd.DataFrame(obs)
o['D'] = o.raw - o.noise
o['t_idx'] = (o.date - o.date.min()).dt.days
corr_level = np.corrcoef(o.level, o.D)[0, 1]
# partial correlation controlling for calendar time (both D and level trend over time)
from numpy.polynomial import polynomial as P_
def resid_on_time(y, t):
    b = np.polyfit(t, y, 1)
    return y - np.polyval(b, t)
D_res = resid_on_time(o.D.values, o.t_idx.values)
lvl_res = resid_on_time(o.level.values, o.t_idx.values)
corr_partial = np.corrcoef(D_res, lvl_res)[0, 1]
print(f'corr(dispersion D_t, mean capability level)          = {corr_level:+.3f}')
print(f'partial corr, controlling for calendar time           = {corr_partial:+.3f}')
print(f'mean level rose {o.level.iloc[-5:].mean()-o.level.iloc[:5].mean():.0f} pts over the panel; '
      f'dispersion moved {o.D.iloc[-5:].mean()-o.D.iloc[:5].mean():+.0f} pts (first 5 vs last 5 snapshots)')
if corr_partial < -0.3:
    verdict_b = 'Dispersion falls with rising capability net of time trend -- consistent with a ceiling/compression effect contributing.'
elif corr_partial > 0.3:
    verdict_b = 'Dispersion RISES with capability net of time trend -- argues against a simple ceiling story.'
else:
    verdict_b = 'No clear level-dispersion relationship net of time trend -- ceiling/compression is not obviously the answer.'
print('>>> ' + verdict_b)

# ---- (c) continuous (between-event) vs at-event convergence ----
print('\n=== (c) does convergence continue BETWEEN a firm\'s own releases (no name change)? ===\n')
print(f'ambient (between-release, unchanged-model) decay over a {med_release_spacing}-day window: '
      f'{implied_28d:+.3f}' if np.isfinite(implied_28d) else 'ambient decay: not enough data to fit')
print(f'at-release-event decay from §3.3: measured 0.390, gap-chasing-alone-predicted 0.451')
if np.isfinite(implied_28d):
    if implied_28d < 0.55:
        print('>>> Convergence keeps happening even without a flagship change, at a rate in the '
              'same ballpark as the at-event decay -- consistent with a CONTINUOUS background '
              'process (ongoing fine-tuning of deployed models, continuous imitation via API '
              'access, or common technique diffusion) rather than something tied specifically '
              'to the release/gap-chasing decision.')
    else:
        print('>>> Ambient (between-release) decay is much slower than at-event decay -- the '
              'convergence really is concentrated AT releases, consistent with the '
              'gap-chasing/imitation story already in the model, not a continuous process.')

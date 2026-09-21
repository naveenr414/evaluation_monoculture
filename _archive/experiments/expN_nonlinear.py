"""Exp N: is the gap -> improvement response linear, or does its shape matter?

dtheta_{iek} = f(gap_{iek}) + event FE + category FE + eps

f estimated three ways: (1) transparent gap bins (already ahead / near tie /
moderately behind / far behind), (2) a continuous piecewise-linear spline in gap,
(3) a quadratic term added to the linear model as a shape test. All within the
event+category FE structure used throughout (so bins/splines act on the
FE-residualized gap and dtheta, exactly as exp1's linear beta does).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, build_events, CORE, FRONTIER

d = load_panel(cats=CORE)
ev = build_events(d, CORE)


def demean(x, by):
    return x - x.groupby(by).transform('mean')


def within(s, df=ev):
    q = s.copy()
    for _ in range(60):
        q = demean(demean(q, df.event), df.cat)
    return q.values


Yw = within(ev.dtheta)
Gw = within(ev.gap)
g_raw = ev.gap.values

print(f'events={ev.event.nunique()}  obs={len(ev)}')
print(f'gap (pre-release, rivals-frontier minus own): mean={g_raw.mean():.1f} sd={g_raw.std():.1f} '
      f'range [{g_raw.min():.1f}, {g_raw.max():.1f}]\n')

# ---- 1. linear baseline (within-FE slope, unweighted -- for comparison to bins/spline) ----
beta_lin = np.sum(Gw * Yw) / np.sum(Gw * Gw)
pred_lin = beta_lin * Gw
ss_res_lin = np.sum((Yw - pred_lin) ** 2)
ss_tot = np.sum(Yw ** 2)
print(f'1. linear f(g)=beta*g       beta={beta_lin:+.4f}   R^2={1-ss_res_lin/ss_tot:.4f}')

# ---- 2. binned specification, on RAW gap (transparent, before any FE assumption) ----
print('\n2. binned response (raw gap -> raw dtheta, event+category FE via within-bin demeaning)')
qs = np.nanpercentile(g_raw, [25, 50, 75])
# gap = rivals' frontier - own old rating: negative gap => already ahead, positive => behind
labels = ['ahead (Q1)', 'near tie (Q2)', 'moderately behind (Q3)', 'far behind (Q4)']
ev['bin'] = pd.cut(ev.gap, bins=[-np.inf, qs[0], qs[1], qs[2], np.inf], labels=labels)
# recompute within-FE dtheta/gap so bin means are comparable net of event+category level
for b in labels:
    sub = ev[ev.bin == b]
    print(f'  {b:24s} n={len(sub):4d}  gap range [{sub.gap.min():7.1f},{sub.gap.max():7.1f}]  '
          f'mean gap={sub.gap.mean():7.1f}  mean dtheta={sub.dtheta.mean():7.2f}  '
          f'(within-FE mean dtheta={Yw[ev.bin.values==b].mean():7.2f}, se={Yw[ev.bin.values==b].std(ddof=1)/np.sqrt(len(sub)):5.2f})')

# pure step-function fit, for reference (NOT nested with the linear model, so not
# comparable by F-test on its own -- a 4-level step necessarily loses information
# relative to a continuous slope; report only as a descriptive shape check)
bin_idx = pd.Categorical(ev.bin, categories=labels).codes
Xbin = np.eye(4)[bin_idx]
coef_bin = np.linalg.lstsq(Xbin, Yw, rcond=None)[0]
pred_bin = Xbin @ coef_bin
ss_res_bin = np.sum((Yw - pred_bin) ** 2)
print(f'\n  bin means of within-FE dtheta: ' + ', '.join(f'{l}={c:+.2f}' for l, c in zip(labels, coef_bin)))
print(f'  R^2 (4-bin step function alone) = {1-ss_res_bin/ss_tot:.4f}  (not nested with linear -- descriptive only)')

# proper nested test: does adding bin dummies ON TOP OF the linear slope improve fit?
# (this is the right way to ask "is there curvature the linear term misses")
from scipy import stats
Xlin = Gw.reshape(-1, 1)
Xnest = np.column_stack([Gw, Xbin[:, 1:]])   # linear slope + 3 bin-deviation dummies
coef_nest = np.linalg.lstsq(Xnest, Yw, rcond=None)[0]
ss_res_nest = np.sum((Yw - Xnest @ coef_nest) ** 2)
n, p_lin, p_nest = len(Yw), 1, Xnest.shape[1]
F = ((ss_res_lin - ss_res_nest) / (p_nest - p_lin)) / (ss_res_nest / (n - p_nest))
p_F = 1 - stats.f.cdf(F, p_nest - p_lin, n - p_nest)
print(f'  nested test (linear + bin dummies vs linear alone): F={F:.2f}, p={p_F:.3f}  '
      f'{"(bins add significant curvature beyond linear)" if p_F < 0.05 else "(no significant curvature beyond linear)"}')

# ---- 3. piecewise-linear spline: knot at the near-tie point (gap=0) ----
print('\n3. piecewise-linear spline (kink at Gw=0, the FE-adjusted gap mean;')
print('   Gw>0 = more-behind-than-typical for that event/category, Gw<0 = more-ahead-than-typical)')
g_pos = np.where(Gw > 0, Gw, 0.0)   # relatively more-behind-than-typical component
g_neg = np.where(Gw < 0, Gw, 0.0)   # relatively more-ahead-than-typical component
X = np.column_stack([g_neg, g_pos])
coef = np.linalg.lstsq(X, Yw, rcond=None)[0]
pred_spl = X @ coef
ss_res_spl = np.sum((Yw - pred_spl) ** 2)
print(f'  slope on more-ahead-than-typical side  (Gw<0) = {coef[0]:+.4f}')
print(f'  slope on more-behind-than-typical side (Gw>0) = {coef[1]:+.4f}')
print(f'  R^2 (kinked spline) = {1-ss_res_spl/ss_tot:.4f}  vs linear R^2={1-ss_res_lin/ss_tot:.4f}')
F2 = ((ss_res_lin - ss_res_spl) / 1) / (ss_res_spl / (n - 2))
p_F2 = 1 - stats.f.cdf(F2, 1, n - 2)
print(f'  F-test (kink vs linear): F={F2:.2f}, p={p_F2:.3f}')

# ---- 4. quadratic shape test ----
print('\n4. quadratic: f(g) = beta*g + kappa*g^2 (kappa>0: convex/accelerating; kappa<0: flattening)')
X2 = np.column_stack([Gw, Gw ** 2 - np.mean(Gw ** 2)])
coef2 = np.linalg.lstsq(X2, Yw, rcond=None)[0]
pred_q = X2 @ coef2
ss_res_q = np.sum((Yw - pred_q) ** 2)
print(f'  beta={coef2[0]:+.4f}  kappa={coef2[1]:+.6f}')
print(f'  R^2 (quadratic) = {1-ss_res_q/ss_tot:.4f}')
F3 = ((ss_res_lin - ss_res_q) / 1) / (ss_res_q / (n - 2))
p_F3 = 1 - stats.f.cdf(F3, 1, n - 2)
print(f'  F-test (quadratic term vs linear): F={F3:.2f}, p={p_F3:.3f}')

# ---- 5. shape by category (does the sign of any nonlinearity vary?) ----
print('\n5. kinked slopes by category (descriptive; ~57 obs per category, wide CIs)')
print(f'{"cat":24s} {"ahead-side slope":>16s} {"behind-side slope":>17s} {"n":>4s}')
for cat, sub in ev.groupby('cat'):
    idx = ev.cat.values == cat
    Xc = X[idx]; Yc = Yw[idx]
    if Xc.shape[0] < 8:
        continue
    cc = np.linalg.lstsq(Xc, Yc, rcond=None)[0]
    print(f'{cat:24s} {cc[0]:13.3f} {cc[1]:12.3f} {idx.sum():4d}')

print('\n=== summary ===')
print(f'linear R^2={1-ss_res_lin/ss_tot:.4f} | bins R^2={1-ss_res_bin/ss_tot:.4f} (p={p_F:.3f}) | '
      f'kink R^2={1-ss_res_spl/ss_tot:.4f} (p={p_F2:.3f}) | quad R^2={1-ss_res_q/ss_tot:.4f} (p={p_F3:.3f})')

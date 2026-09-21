"""Same eval-chasing regression as expA, but over ALL orgs with a flagship history
(not just the 8 FRONTIER majors) -- broader firm universe."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, flagships, fe_regress, CORE, MATURE_DAYS

d = load_panel(cats=CORE, firms_only=False)
print(f"orgs in panel: {d.org.nunique()}")

def build_events_all(d, cats=CORE, mature_days=MATURE_DAYS, min_rivals=3):
    rat = d.pivot_table(index=['dt', 'model_name'], columns='category', values='rating')
    var = d.pivot_table(index=['dt', 'model_name'], columns='category', values='variance')
    flag = flagships(d)
    snaps = np.sort(d.dt.unique())
    all_firms = flag.index.get_level_values('org').unique()

    events = []
    for firm in all_firms:
        ser = flag.xs(firm, level='org').sort_index()
        prev_m, prev_t = None, None
        for t, m in ser.items():
            if prev_m is not None and m != prev_m:
                events.append((firm, prev_m, m, t, prev_t))
            prev_m, prev_t = m, t

    rows = []
    for firm, old, new, switch, t_old in events:
        cand = [s for s in snaps if s >= switch + np.timedelta64(mature_days, 'D')]
        if not cand:
            continue
        t_new = cand[0]
        try:
            th_old, th_new = rat.loc[(t_old, old)][cats], rat.loc[(t_new, new)][cats]
            v_old, v_new = var.loc[(t_old, old)][cats], var.loc[(t_new, new)][cats]
        except KeyError:
            continue
        if th_old.isna().any() or th_new.isna().any():
            continue

        snap = rat.xs(t_old, level='dt')
        rivals = [flag.get((t_old, f)) for f in all_firms if f != firm]
        rivals = [m for m in rivals if m is not None and m in snap.index]
        riv = snap.loc[rivals][cats].dropna()
        if len(riv) < min_rivals:
            continue
        frontier = riv.max(axis=0)
        v_front = np.array([var.loc[(t_old, riv[k].idxmax())][k] for k in cats])

        for j, k in enumerate(cats):
            rows.append(dict(
                event=f'{firm}:{old}->{new}@{pd.Timestamp(switch).strftime("%Y%m%d")}',
                firm=firm, cat=k, switch=switch,
                dtheta=th_new[k] - th_old[k],
                gap=frontier[k] - th_old[k],
                v_old=v_old[k], v_new=v_new[k], v_front=v_front[j]))
    return pd.DataFrame(rows)

ev = build_events_all(d)
print(f"events={ev.event.nunique()}  firms={ev.firm.nunique()}  obs={len(ev)}")

X = ev[['gap']].values
groups = [ev.event.values, ev.cat.values]
ncov = np.array([ev.v_old.mean()])
nvar = np.array([[ev.v_old.mean() + ev.v_front.mean()]])
beta, se, beta_c, resid, dfac = fe_regress(
    ev.dtheta.values, X, groups, ncov, nvar, cluster=ev.event.values)

print(f"beta naive = {beta[0]:+.4f} (SE {se[0]:.4f}, t={beta[0]/se[0]:+.2f})")
print(f"beta noise-corrected = {beta_c[0]:+.4f}")
print("\nfirms included:", sorted(ev.firm.unique()))

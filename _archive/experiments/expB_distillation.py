"""Experiment B: is beta measuring evaluation response, or imitation of a teacher?

Eval-chasing  -> firms move toward the PER-CATEGORY leader (a different rival in
                 each category, wherever the board says they lag).
Distillation  -> firms move toward ONE model's whole profile, including in
                 categories where they already lead it.

Three tests:
  B1 horse race: dtheta on gap-to-category-leader vs gap-to-best-overall-rival.
  B2 orthogonalised: gap to the per-category frontier EXCLUDING the teacher,
     which isolates board-following from teacher-following.
  B3 sign test: where a firm is ALREADY AHEAD of the teacher, distillation predicts
     relative decline; eval-chasing predicts no pull.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'experiments')
from common import load_panel, build_events, fe_regress, CORE

d = load_panel(cats=CORE)
ev = build_events(d, CORE).dropna(subset=['gap_teacher', 'gap_excl_teacher'])
groups = [ev.event.values, ev.cat.values]
print(f'events={ev.event.nunique()}  obs={len(ev)}')
print('teacher (best-overall rival) frequency:')
print(ev.groupby('event').teacher.first().value_counts().head(8).to_string())

rho = np.corrcoef(ev.gap.values, ev.gap_teacher.values)[0, 1]
print(f'\ncorr(gap_to_category_leader, gap_to_teacher) = {rho:.3f}')
share = (ev.gap - ev.gap_teacher).abs().lt(1e-9).mean()
print(f'share of (event,category) cells where teacher IS the category leader = {share:.2%}')
print('  -> identification comes from the other {:.0%} of cells\n'.format(1 - share))

vo, vf = ev.v_old.mean(), ev.v_front.mean()

def show(name, cols, ncov, nvar):
    X = ev[cols].values
    b, se, bc, _, _ = fe_regress(ev.dtheta.values, X, groups, ncov, nvar,
                                 cluster=ev.event.values)
    print(f'  {name}')
    for j, c in enumerate(cols):
        star = '***' if abs(b[j]/se[j]) > 2.58 else '**' if abs(b[j]/se[j]) > 1.96 else \
               '*' if abs(b[j]/se[j]) > 1.64 else ''
        print(f'    {c:20s} beta={b[j]:+.4f} (SE {se[j]:.4f}, t={b[j]/se[j]:+.2f}){star}'
              f'   corrected={bc[j]:+.4f}')

# --- B1: horse race ---
print('B1  horse race')
# both regressors carry -eps_old, so both covary spuriously with dtheta by var(eps_old)
ncov = np.array([vo, vo])
nvar = np.array([[vo + vf, vo], [vo, vo + vf]])
show('', ['gap', 'gap_teacher'], ncov, nvar)

# --- B2: orthogonalised (frontier excluding teacher) ---
print('\nB2  board-following isolated from teacher-following')
show('', ['gap_excl_teacher', 'gap_teacher'], ncov, nvar)

# --- B3: sign test where already ahead of the teacher ---
print('\nB3  cells where the firm ALREADY LEADS the teacher (gap_teacher < 0)')
ahead = ev[ev.gap_teacher < 0]
behind = ev[ev.gap_teacher > 0]
print(f'    n ahead = {len(ahead)}, n behind = {len(behind)}')
for nm, sub in [('ahead of teacher', ahead), ('behind teacher', behind)]:
    if len(sub) < 40:
        print(f'    {nm}: too few obs'); continue
    g = [sub.event.values, sub.cat.values]
    b, se, bc, _, _ = fe_regress(sub.dtheta.values, sub[['gap_teacher']].values, g,
                                 np.array([vo]), np.array([[vo + vf]]),
                                 cluster=sub.event.values)
    print(f'    {nm:18s} beta_teacher={b[0]:+.4f} (SE {se[0]:.4f}, t={b[0]/se[0]:+.2f})'
          f'  corrected={bc[0]:+.4f}')
print('    distillation predicts a POSITIVE teacher pull in both halves;')
print('    eval-chasing predicts the pull vanishes once you lead the teacher.')

# --- B4: does the teacher-following channel explain the missing homogenising force? ---
print('\nB4  variance accounting')
for cols in (['gap'], ['gap', 'gap_teacher']):
    X = ev[cols].values
    b, se, bc, resid, _ = fe_regress(ev.dtheta.values, X, groups, cluster=ev.event.values)
    print(f'    regressors {str(cols):32s} residual var = {resid.var(ddof=1):7.1f}')

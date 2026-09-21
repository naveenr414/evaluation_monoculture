"""Residual dispersion over time since a new category's inception (2025-11-05).
Two-way (firm x category) demeaning computed WITHIN each snapshot, across the
9 new categories, then cross-sectional std of residuals at each snapshot,
plotted against days/weeks since launch.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEW = ['expert','industry_life_and_physical_and_social_science','industry_medicine_and_healthcare',
       'industry_writing_and_literature_and_language','industry_business_and_management_and_financial_operations',
       'industry_entertainment_and_sports_and_media','industry_legal_and_government',
       'industry_software_and_it_services','industry_mathematical']
LAUNCH = pd.Timestamp('2025-11-05')

d = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
d['dt'] = pd.to_datetime(d.leaderboard_publish_date)
d['org'] = d.organization.str.strip().str.lower()
d = d[(d.org != '') & d.category.isin(NEW)]

# frontier (max rating) per org x category x snapshot
g = d.groupby(['org','category','dt'])['rating'].max().reset_index()

# only keep orgs with reasonable coverage across snapshots (avoid 1-shot noise)
org_counts = g.groupby('org')['dt'].nunique()
keep_orgs = org_counts[org_counts >= 15].index
g = g[g.org.isin(keep_orgs)]
print(f'orgs kept: {len(keep_orgs)} -> {sorted(keep_orgs)}')

def demean(group):
    grand = group['rating'].mean()
    fm = group.groupby('org')['rating'].transform('mean')
    cm = group.groupby('category')['rating'].transform('mean')
    group = group.copy()
    group['resid'] = group['rating'] - fm - cm + grand
    return group

g = g.groupby('dt', group_keys=False).apply(demean)
g['days_since_launch'] = (g['dt'] - LAUNCH).dt.days

disp = g.groupby('dt').agg(std_resid=('resid','std'), n_obs=('resid','size'),
                            days=('days_since_launch','first')).reset_index()
disp = disp[disp.n_obs >= 10].sort_values('days')
print(disp.to_string(index=False))

# linear trend
from numpy.polynomial import polynomial as P
coef = np.polyfit(disp.days, disp.std_resid, 1)
print(f'\nlinear trend: std_resid = {coef[1]:.2f} + {coef[0]:.4f} * days_since_launch')
print(f'  ({coef[0]*7:.3f} per week, {coef[0]*30:.2f} per month)')

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(disp.days, disp.std_resid, s=25, alpha=0.7, color="steelblue")
xs = np.linspace(disp.days.min(), disp.days.max(), 100)
ax.plot(xs, coef[1] + coef[0]*xs, color="crimson", lw=2, label=f"trend: {coef[0]*7:+.2f}/week")
ax.set_xlabel("days since category launch (2025-11-05)")
ax.set_ylabel("cross-firm std of residual\n(after removing firm strength & category difficulty)")
ax.set_title("Dispersion of firm advantage in NEW categories, since inception")
ax.legend()
plt.tight_layout()
plt.savefig("experiments/scratch_persistence/newcat_dispersion.png", dpi=130, bbox_inches="tight")
print("\nSaved experiments/scratch_persistence/newcat_dispersion.png")

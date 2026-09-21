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

g = d.groupby(['org','category','dt'])['rating'].max().reset_index()
n_snaps = g.dt.nunique()
cov = g.groupby('org')['dt'].nunique()
balanced_orgs = cov[cov == n_snaps].index
g = g[g.org.isin(balanced_orgs)]
print(f'balanced panel: {len(balanced_orgs)} orgs x {n_snaps} snapshots x 9 categories')

def demean(group):
    grand = group['rating'].mean()
    fm = group.groupby('org')['rating'].transform('mean')
    cm = group.groupby('category')['rating'].transform('mean')
    group = group.copy()
    group['resid'] = group['rating'] - fm - cm + grand
    return group

g = g.groupby('dt', group_keys=False).apply(demean)
g['days'] = (g['dt'] - LAUNCH).dt.days

disp = g.groupby('dt').agg(std_resid=('resid','std'), n_obs=('resid','size'), days=('days','first')).reset_index()
disp = disp.sort_values('days')
print(disp.to_string(index=False))

coef = np.polyfit(disp.days, disp.std_resid, 1)
print(f'\nlinear trend: std = {coef[1]:.2f} + {coef[0]:.5f}*days  ({coef[0]*7:+.3f}/week, {coef[0]*30:+.2f}/month)')

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(disp.days, disp.std_resid, s=25, alpha=0.7, color="steelblue")
xs = np.linspace(disp.days.min(), disp.days.max(), 100)
ax.plot(xs, coef[1] + coef[0]*xs, color="crimson", lw=2, label=f"trend: {coef[0]*7:+.3f}/week")
ax.set_xlabel("days since category launch (2025-11-05)")
ax.set_ylabel("cross-firm std of residual\n(after removing firm strength & category difficulty)")
ax.set_title(f"Dispersion in NEW categories since inception (balanced panel, n={len(balanced_orgs)} firms)")
ax.legend()
plt.tight_layout()
plt.savefig("experiments/scratch_persistence/newcat_dispersion_balanced.png", dpi=130, bbox_inches="tight")
print("Saved experiments/scratch_persistence/newcat_dispersion_balanced.png")

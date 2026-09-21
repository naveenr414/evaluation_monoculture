"""Nested variance decomposition for predicting quarter-over-quarter change in
firm-category performance:

  dtheta_{i,k,t} = beta * gap_{i,k,t-1} + alpha_{i,k} + lambda_{k,t} + eps

  gap_{i,k,t-1}  = rivals' frontier in category k at t-1  minus  firm i's own theta at t-1
                   (pure catch-up predictor)
  alpha_{i,k}    = firm x category fixed effect (persistent comparative advantage in
                   IMPROVEMENT RATE for that category)
  lambda_{k,t}   = category x quarter fixed effect (common technology progress: a shock
                   shared by every firm in category k that quarter)

Nested R^2:
  Model 1: dtheta ~ gap
  Model 2: dtheta ~ gap + alpha_{i,k}
  Model 3: dtheta ~ gap + alpha_{i,k} + lambda_{k,t}
  Residual = 1 - R^2 of model 3
"""
import pandas as pd, numpy as np

df = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
df = df[df['organization'].str.strip() != '']
df = df[df['category'] != 'overall']
df = df[~df['category'].isin(['exclude_ties', 'hard_prompts', 'hard_prompts_english', 'longer_query'])]
df['date'] = pd.to_datetime(df['leaderboard_publish_date'])
df['quarter'] = df['date'].dt.to_period('Q')

g = df.groupby(['organization', 'category', 'quarter'])['rating'].max().reset_index()
firm_counts = g.groupby('organization')['quarter'].nunique()
keep_firms = firm_counts[firm_counts >= 6].index
g = g[g['organization'].isin(keep_firms)]

quarters = sorted(g['quarter'].unique())
qidx = {q: i for i, q in enumerate(quarters)}
g['qi'] = g['quarter'].map(qidx)

# rivals' frontier per category x quarter-index (max across ALL orgs present)
frontier = g.groupby(['category', 'qi'])['rating'].max().rename('cat_frontier_all')
# for computing "rivals" frontier excluding self, we need max EXCLUDING firm i;
# use top-2 trick: if firm i IS the max, rivals' frontier = 2nd highest, else = max
def top2(s):
    s = s.sort_values(ascending=False)
    top = s.iloc[0]
    second = s.iloc[1] if len(s) > 1 else np.nan
    return pd.Series({'top': top, 'second': second})

t2 = g.groupby(['category', 'qi'])['rating'].apply(lambda s: pd.Series(top2(s))).unstack()

piv = g.pivot_table(index=['organization', 'category'], columns='qi', values='rating')

rows = []
for (org, cat), row in piv.iterrows():
    present = row.dropna()
    qis = sorted(present.index)
    for a, b_ in zip(qis[:-1], qis[1:]):
        if b_ != a + 1:
            continue  # only true consecutive quarters
        theta_old = present[a]
        theta_new = present[b_]
        try:
            top, second = t2.loc[(cat, a), 'top'], t2.loc[(cat, a), 'second']
        except KeyError:
            continue
        rival_frontier = second if theta_old >= top else top
        if pd.isna(rival_frontier):
            continue
        rows.append(dict(org=org, cat=cat, qi=b_, dtheta=theta_new - theta_old,
                         gap=rival_frontier - theta_old))

d = pd.DataFrame(rows)
d['firm_cat'] = d.org + '::' + d.cat
d['cat_q'] = d.cat + '::' + d.qi.astype(str)
print(f'n obs = {len(d)}, firm x cat groups = {d.firm_cat.nunique()}, cat x quarter groups = {d.cat_q.nunique()}')

Y = d.dtheta.values.astype(float)
G = d.gap.values.astype(float)
TSS = np.sum((Y - Y.mean())**2)

# ---- Model 1: gap only ----
b1 = np.sum((G - G.mean())*(Y - Y.mean())) / np.sum((G-G.mean())**2)
a1 = Y.mean() - b1*G.mean()
resid1 = Y - (a1 + b1*G)
SSR1 = np.sum(resid1**2)
R2_1 = 1 - SSR1/TSS

# ---- Model 2: gap + firm x cat FE (within-transform, FWL) ----
def within(x, keys):
    x = pd.Series(x.copy())
    for _ in range(1):  # single grouping, exact in one pass
        x = x - x.groupby(keys).transform('mean')
    return x.values

Yd2 = within(Y, d.firm_cat)
Gd2 = within(G, d.firm_cat)
b2 = np.sum(Gd2*Yd2) / np.sum(Gd2**2)
resid2 = Yd2 - b2*Gd2
SSR2 = np.sum(resid2**2)
R2_2 = 1 - SSR2/TSS

# ---- Model 3: gap + firm x cat FE + cat x quarter FE (iterative alternating demeaning) ----
Yd3 = Y.copy(); Gd3 = G.copy()
Yd3 = pd.Series(Yd3); Gd3 = pd.Series(Gd3)
for _ in range(100):
    Yd3 = Yd3 - Yd3.groupby(d.firm_cat.values).transform('mean')
    Yd3 = Yd3 - Yd3.groupby(d.cat_q.values).transform('mean')
    Gd3 = Gd3 - Gd3.groupby(d.firm_cat.values).transform('mean')
    Gd3 = Gd3 - Gd3.groupby(d.cat_q.values).transform('mean')
b3 = np.sum(Gd3.values*Yd3.values) / np.sum(Gd3.values**2)
resid3 = Yd3.values - b3*Gd3.values
SSR3 = np.sum(resid3**2)
R2_3 = 1 - SSR3/TSS

print(f'\nModel 1 (gap only):                       beta={b1:+.4f}   R2={R2_1:.4f}')
print(f'Model 2 (+ firm x cat FE, alpha):          beta={b2:+.4f}   R2={R2_2:.4f}   (+{R2_2-R2_1:.4f} from alpha)')
print(f'Model 3 (+ cat x quarter FE, lambda):      beta={b3:+.4f}   R2={R2_3:.4f}   (+{R2_3-R2_2:.4f} from lambda)')
print(f'Residual (unexplained):                                    {1-R2_3:.4f}')

print(f'\nShare of TOTAL R2 (={R2_3:.4f}) attributable to each piece:')
print(f'  gap (catch-up)         : {R2_1/R2_3:.1%}')
print(f'  alpha (persistent adv) : {(R2_2-R2_1)/R2_3:.1%}')
print(f'  lambda (common tech)   : {(R2_3-R2_2)/R2_3:.1%}')

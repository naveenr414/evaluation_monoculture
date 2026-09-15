"""Same nested decomposition, with sampling-noise correction on beta at each stage
(gap and dtheta both carry -eps_old from the outgoing quarter's BT sampling noise,
which biases beta toward positive... but the demeaning by many small/noisy firms can
also flip sign; correcting properly matters here).
"""
import pandas as pd, numpy as np

df = pd.read_parquet('data/leaderboard/text/full-00000-of-00001.parquet')
df = df[df['organization'].str.strip() != '']
df = df[df['category'] != 'overall']
df = df[~df['category'].isin(['exclude_ties', 'hard_prompts', 'hard_prompts_english', 'longer_query'])]
df['date'] = pd.to_datetime(df['leaderboard_publish_date'])
df['quarter'] = df['date'].dt.to_period('Q')

df = df.dropna(subset=['rating'])
# frontier MODEL (not just rating) per org x category x quarter, so we can pull its variance too
idx = df.groupby(['organization','category','quarter'])['rating'].idxmax()
g = df.loc[idx, ['organization','category','quarter','rating','variance']].reset_index(drop=True)

firm_counts = g.groupby('organization')['quarter'].nunique()
keep_firms = firm_counts[firm_counts >= 6].index
g = g[g['organization'].isin(keep_firms)]

quarters = sorted(g['quarter'].unique())
qidx = {q: i for i, q in enumerate(quarters)}
g['qi'] = g['quarter'].map(qidx)

def top2(s):
    s = s.sort_values(ascending=False)
    return pd.Series({'top': s.iloc[0], 'second': s.iloc[1] if len(s) > 1 else np.nan})

t2 = g.groupby(['category','qi'])['rating'].apply(lambda s: pd.Series(top2(s))).unstack()

piv_r = g.pivot_table(index=['organization','category'], columns='qi', values='rating')
piv_v = g.pivot_table(index=['organization','category'], columns='qi', values='variance')

rows = []
for (org, cat), row in piv_r.iterrows():
    present = row.dropna()
    qis = sorted(present.index)
    for a, b_ in zip(qis[:-1], qis[1:]):
        if b_ != a + 1: continue
        theta_old, theta_new = present[a], present[b_]
        try:
            top, second = t2.loc[(cat,a),'top'], t2.loc[(cat,a),'second']
        except KeyError:
            continue
        rival = second if theta_old >= top else top
        if pd.isna(rival): continue
        v_old = piv_v.loc[(org,cat), a] if a in piv_v.columns else np.nan
        rows.append(dict(org=org, cat=cat, qi=b_, dtheta=theta_new-theta_old,
                         gap=rival-theta_old, v_old=v_old))

d = pd.DataFrame(rows).dropna(subset=['v_old'])
d['firm_cat'] = d.org + '::' + d.cat
d['cat_q'] = d.cat + '::' + d.qi.astype(str)
print(f'n obs = {len(d)}')

Y = d.dtheta.values.astype(float)
G = d.gap.values.astype(float)
TSS = np.sum((Y - Y.mean())**2)
mean_v_old = d.v_old.mean()

def fit_corrected(Yin, Gin, dfac):
    cov = np.mean((Gin-Gin.mean())*(Yin-Yin.mean()))
    vg = np.mean((Gin-Gin.mean())**2)
    s2 = mean_v_old * dfac  # noise in theta_old enters BOTH gap (as -eps_old) and dtheta (as -eps_old)
    b = (cov - s2) / (vg - s2)
    a = Yin.mean() - b*Gin.mean()
    resid = Yin - (a + b*Gin)
    return b, np.sum(resid**2)

# Model 1: no FE, dfac=1 (no demeaning shrinkage)
b1, SSR1 = fit_corrected(Y, G, dfac=1.0)
R2_1 = 1 - SSR1/TSS

# Model 2: within firm x cat
Yd2 = pd.Series(Y) - pd.Series(Y).groupby(d.firm_cat.values).transform('mean')
Gd2 = pd.Series(G) - pd.Series(G).groupby(d.firm_cat.values).transform('mean')
n_fc = d.firm_cat.nunique()
dfac2 = (len(d) - n_fc) / len(d)
b2, SSR2 = fit_corrected(Yd2.values, Gd2.values, dfac=dfac2)
R2_2 = 1 - SSR2/TSS

# Model 3: within firm x cat AND cat x quarter (iterative)
Yd3, Gd3 = pd.Series(Y.copy()), pd.Series(G.copy())
for _ in range(100):
    Yd3 = Yd3 - Yd3.groupby(d.firm_cat.values).transform('mean')
    Yd3 = Yd3 - Yd3.groupby(d.cat_q.values).transform('mean')
    Gd3 = Gd3 - Gd3.groupby(d.firm_cat.values).transform('mean')
    Gd3 = Gd3 - Gd3.groupby(d.cat_q.values).transform('mean')
n_cq = d.cat_q.nunique()
dfac3 = (len(d) - n_fc - n_cq + d.cat.nunique()) / len(d)  # rough df correction
b3, SSR3 = fit_corrected(Yd3.values, Gd3.values, dfac=max(dfac3, 0.05))
R2_3 = 1 - SSR3/TSS

print(f'mean BT sampling variance (v_old) = {mean_v_old:.1f}')
print(f'\nModel 1 (gap only, noise-corrected):       beta={b1:+.4f}   R2={R2_1:.4f}')
print(f'Model 2 (+ firm x cat FE, alpha):           beta={b2:+.4f}   R2={R2_2:.4f}   (+{R2_2-R2_1:.4f} from alpha)')
print(f'Model 3 (+ cat x quarter FE, lambda):       beta={b3:+.4f}   R2={R2_3:.4f}   (+{R2_3-R2_2:.4f} from lambda)')
print(f'Residual (unexplained):                                     {1-R2_3:.4f}')

print(f'\nShare of TOTAL R2 (={R2_3:.4f}) attributable to each piece:')
print(f'  gap (catch-up)         : {R2_1/R2_3:.1%}')
print(f'  alpha (persistent adv) : {(R2_2-R2_1)/R2_3:.1%}')
print(f'  lambda (common tech)   : {(R2_3-R2_2)/R2_3:.1%}')

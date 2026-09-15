import sys, numpy as np, pandas as pd
sys.path.insert(0,'experiments')
from exp_sign_persistence import build, nw_tstat, bh, DO, DC
from common import CORE
from scipy import stats

qp = build('2025-01-01')
bal = qp.pivot_table(index=['org','cat'], columns='q', values='resid').dropna()
recs=[]
for (o,c), r in zip(bal.index, bal.values):
    mu,t = nw_tstat(r, lags=1)
    recs.append((o, c, mu, t, 2*stats.t.sf(abs(t), df=len(r)-1),
                 bool((r>0).all() or (r<0).all()), np.sign(mu)))
tab=pd.DataFrame(recs, columns=['org','cat','mean','t','p','same','sgn'])
rej,q=bh(tab.p.values,0.05); tab['q']=q; tab['FDR']=rej

nf = tab.org.nunique()
rows=[]
for c in CORE:
    s=tab[tab.cat==c]
    rows.append((DC[c], len(s),
        s.same.sum(), 100*s.same.mean(),
        int(((s.same)&(s.sgn>0)).sum()), int(((s.same)&(s.sgn<0)).sum()),
        int(s.FDR.sum()), 100*s.FDR.mean(),
        s['mean'].std(), s['mean'].abs().mean()))
out=pd.DataFrame(rows, columns=['category','firms','same_sign','same_%',
     'pos','neg','FDR_sig','FDR_%','sd_of_means','mean_|resid|'])
print(f'{nf} firms, 6 quarters (2025Q1-2026Q2)\n')
print(out.sort_values('same_%',ascending=False)
         .to_string(index=False, float_format=lambda x: f'{x:6.1f}'))
print('\nwhich firms are specialized in each category (same sign all 6 quarters):')
for c in CORE:
    s=tab[(tab.cat==c)&tab.same].sort_values('mean',ascending=False)
    if len(s)==0: continue
    txt=', '.join(f'{DO.get(o,o)} {m:+.0f}' for o,m in zip(s.org,s['mean']))
    print(f'  {DC[c]:<12} ({len(s)}/{nf}): {txt}')

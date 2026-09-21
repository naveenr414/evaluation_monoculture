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
    p = 2*stats.t.sf(abs(t), df=len(r)-1)
    same = bool((r>0).all() or (r<0).all())
    recs.append((DO.get(o,o), DC[c], mu, r.min(), r.max(), t, p, same))
tab = pd.DataFrame(recs, columns=['firm','category','mean','min_q','max_q','t_HAC','p','all_same'])
rej,q = bh(tab.p.values,0.05); tab['q_BH']=q; tab['FDR']=rej

ss = tab[tab.all_same].copy()
ss['abs']=ss['mean'].abs()
ss=ss.sort_values('abs',ascending=False)
print(f'{len(ss)} of {len(tab)} pairs same-signed in all 6 quarters, by |mean|:\n')
print(ss[['firm','category','mean','min_q','max_q','t_HAC','q_BH','FDR']]
      .to_string(index=False, float_format=lambda x: f'{x:7.2f}'))
print('\nby firm: count of same-signed pairs')
print(ss.groupby('firm').size().sort_values(ascending=False).to_string())
print('\nby category: count of same-signed pairs')
print(ss.groupby('category').size().sort_values(ascending=False).to_string())

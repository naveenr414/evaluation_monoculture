"""Is the quarter-to-quarter slope below 1 (catch-up), and by how much after
correcting for BT sampling noise in the regressor?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'experiments')
from common import load_panel, two_way_resid, flagships, CORE

d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
flag = flagships(d)
rat = d.pivot_table(index=['dt','model_name'], columns='category', values='rating')
var = d.pivot_table(index=['dt','model_name'], columns='category', values='variance')
dates = sorted(d.dt.unique()); last = d.dt.max()
snap = d[(d.dt==last)&(d.category=='overall')].dropna(subset=['rating'])
ranked = snap.sort_values('rating',ascending=False).groupby('org').head(1).sort_values('rating',ascending=False)
ORGS=[o for o in ranked.org if o and o not in ('inception-ai',)][:23]

rows=[]
for t in dates:
    prof, vs = {}, {}
    for o in ORGS:
        m=flag.get((t,o))
        if m is None: continue
        try: r=rat.loc[(t,m)][CORE]; v=var.loc[(t,m)][CORE]
        except KeyError: continue
        if r.isna().any(): continue
        prof[o]=r.values.astype(float); vs[o]=v.values.astype(float)
    if len(prof)<6: continue
    nm=sorted(prof); P=np.array([prof[n] for n in nm]); V=np.array([vs[n] for n in nm])
    G=P.max(0,keepdims=True)-P; R=two_way_resid(P)
    for i,o in enumerate(nm):
        for j,c in enumerate(CORE):
            rows.append((t,o,c,G[i,j],R[i,j],V[i,j]))
p=pd.DataFrame(rows,columns=['dt','org','cat','gap','resid','v'])
p['q']=pd.PeriodIndex(p.dt,freq='Q')
q=p.groupby(['q','org','cat'])[['gap','resid','v']].mean().reset_index()
# within-quarter averaging over ~20 snapshots of the SAME flagship does not
# average away BT noise (same model, same battles), so keep v as-is.
qs=sorted(q.q.unique())

def pairs(col,lag,extra=('v',)):
    w={c:q.pivot_table(index=['org','cat'],columns='q',values=c) for c in (col,)+extra}
    out=[]
    for i in range(len(qs)-lag):
        a,b=w[col].get(qs[i]),w[col].get(qs[i+lag])
        va=w['v'].get(qs[i])
        m=a.notna()&b.notna()&va.notna()
        out.append(pd.DataFrame({'x':a[m],'y':b[m],'vx':va[m],
                                 'org':[k[0] for k in a[m].index],
                                 'cat':[k[1] for k in a[m].index]}))
    return pd.concat(out,ignore_index=True)

def ols(x,y):
    b1,b0=np.polyfit(x,y,1); e=y-(b0+b1*x); n=len(x)
    se=np.sqrt((e**2).sum()/(n-2)/((x-x.mean())**2).sum())
    r2=1-(e**2).sum()/((y-y.mean())**2).sum()
    return b1,se,r2

print('quarterly AR(1): slope, SE, test vs 1 (slope<1 == catch-up)\n')
for col in ('gap','resid'):
    for lag in (1,4):
        D=pairs(col,lag)
        b1,se,r2=ols(D.x.values,D.y.values)
        # errors-in-variables: OLS slope attenuated by vx/(var(x))
        lam=1-D.vx.mean()/D.x.var(ddof=1)
        print(f'  {col:6s} lag={lag}  slope={b1:+.3f} (se {se:.3f})  '
              f't vs 1 = {(b1-1)/se:+.2f}   R2={r2:.3f}  '
              f'reliability lambda={lam:.3f}  EIV-corrected slope={b1/lam:+.3f}')
    # within firm (firm FE) and within category
    D=pairs(col,1)
    for grp in ('org','cat'):
        Dg=D.copy()
        Dg['x']=Dg.x-Dg.groupby(grp).x.transform('mean')
        Dg['y']=Dg.y-Dg.groupby(grp).y.transform('mean')
        b1,se,r2=ols(Dg.x.values,Dg.y.values)
        print(f'         lag=1 within-{grp} FE: slope={b1:+.3f} (se {se:.3f})  R2={r2:.3f}')
    # both FE
    Dg=D.copy()
    for g in ('org','cat'):
        Dg['x']=Dg.x-Dg.groupby(g).x.transform('mean')
        Dg['y']=Dg.y-Dg.groupby(g).y.transform('mean')
    b1,se,r2=ols(Dg.x.values,Dg.y.values)
    print(f'         lag=1 within firm+cat FE: slope={b1:+.3f} (se {se:.3f})  R2={r2:.3f}')
    print()

print('AR(1) consistency check: does lag-4 slope equal (lag-1 slope)^4?')
for col in ('gap','resid'):
    b1,_,_=ols(*[pairs(col,1)[c].values for c in ('x','y')])
    b4,_,_=ols(*[pairs(col,4)[c].values for c in ('x','y')])
    print(f'  {col:6s} rho_1={b1:.3f}  rho_1^4={b1**4:.3f}  observed rho_4={b4:.3f}')

print('\nimplied half-life and annual closure')
for col in ('gap','resid'):
    b1,_,_=ols(*[pairs(col,1)[c].values for c in ('x','y')])
    print(f'  {col:6s} {100*(1-b1):.1f}% closed per quarter, '
          f'{100*(1-b1**4):.1f}% per year, half-life {np.log(.5)/np.log(b1):.2f} quarters')

print('\nis the frontier moving? mean category leader rating by quarter')
lead=p.groupby(['q','cat']).apply(lambda g: g.gap.min()).groupby('q').mean()
thet=p.groupby('q').gap.mean()
raw=d[d.category!='overall'].copy(); raw['q']=pd.PeriodIndex(raw.dt,freq='Q')
top=raw.groupby(['q','category']).rating.max().groupby('q').mean()
print('  mean across categories of the max rating:')
print(top.to_string())
print('\n  mean gap to leader by quarter:')
print(p.groupby('q').gap.mean().to_string())

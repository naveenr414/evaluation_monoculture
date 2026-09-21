"""Is the long-horizon plateau real, or a composition artifact?  Recompute the
slope-vs-horizon curve on a BALANCED panel of firms present in every month."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'experiments')
from common import load_panel, two_way_resid, flagships, CORE

d = load_panel(cats=CORE, firms_only=False, since='2025-01-01')
flag = flagships(d)
rat = d.pivot_table(index=['dt','model_name'], columns='category', values='rating')
dates = sorted(d.dt.unique()); last = d.dt.max()
snap = d[(d.dt==last)&(d.category=='overall')].dropna(subset=['rating'])
ranked = snap.sort_values('rating',ascending=False).groupby('org').head(1).sort_values('rating',ascending=False)
ORGS=[o for o in ranked.org if o and o not in ('inception-ai',)][:23]

rows=[]
for t in dates:
    prof={}
    for o in ORGS:
        m=flag.get((t,o))
        if m is None: continue
        try: r=rat.loc[(t,m)][CORE]
        except KeyError: continue
        if r.isna().any(): continue
        prof[o]=r.values.astype(float)
    if len(prof)<6: continue
    nm=sorted(prof); P=np.array([prof[n] for n in nm])
    G=P.max(0,keepdims=True)-P; R=two_way_resid(P)
    for i,o in enumerate(nm):
        for j,c in enumerate(CORE):
            rows.append((t,o,c,G[i,j],R[i,j]))
p=pd.DataFrame(rows,columns=['dt','org','cat','gap','resid'])
p['m']=pd.PeriodIndex(p.dt,freq='M')
mo=p.groupby(['m','org','cat'])[['gap','resid']].mean().reset_index()
months=sorted(mo.m.unique())

cov=mo.groupby('org').m.nunique()
BAL=cov[cov==len(months)].index.tolist()
print(f'{len(months)} months; balanced firms present in all: {len(BAL)}')
print(' ', BAL)

def curve(sub, label):
    W={c:sub.pivot_table(index=['org','cat'],columns='m',values=c) for c in ('gap','resid')}
    def lf(col,h):
        out=[]
        for i in range(len(months)-h):
            a,b=W[col].get(months[i]),W[col].get(months[i+h])
            if a is None or b is None: continue
            k=a.notna()&b.notna()
            if k.sum()==0: continue
            out.append(pd.DataFrame({'x':a[k].values,'y':b[k].values,
                'org':[z[0] for z in a[k].index],'cat':[z[1] for z in a[k].index]}))
        return pd.concat(out,ignore_index=True) if out else None
    def ols_cl(x,y,cl):
        X=np.column_stack([np.ones(len(x)),x])
        bt=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@bt
        A=np.linalg.inv(X.T@X); meat=np.zeros((2,2))
        for g in np.unique(cl):
            s=X[cl==g].T@e[cl==g]; meat+=np.outer(s,s)
        G=len(np.unique(cl)); V=A@meat@A*(G/max(G-1,1))
        return bt[1], np.sqrt(V[1,1])
    print(f'\n--- {label} ---')
    print(f'{"h":>3} {"gap":>17} {"gap|FE":>17} {"resid":>17} {"n":>6} {"starts":>7}')
    for h in range(1,16):
        D=lf('gap',h)
        if D is None or len(D)<40: continue
        b,se=ols_cl(D.x.values,D.y.values,D.org.values)
        Df=D.copy()
        for g in ('org','cat'):
            Df['x']=Df.x-Df.groupby(g).x.transform('mean')
            Df['y']=Df.y-Df.groupby(g).y.transform('mean')
        bf,sef=ols_cl(Df.x.values,Df.y.values,Df.org.values)
        Dr=lf('resid',h); br,ser=ols_cl(Dr.x.values,Dr.y.values,Dr.org.values)
        print(f'{h:>3} {b:>9.3f} ({se:.3f}) {bf:>9.3f} ({sef:.3f}) {br:>9.3f} ({ser:.3f}) '
              f'{len(D):>6} {len(months)-h:>7}')

curve(mo, 'ALL firms (unbalanced; the published curve)')
curve(mo[mo.org.isin(BAL)], f'BALANCED panel, {len(BAL)} firms in all 17 months')

# fixed-start check: only pairs starting in the first 5 months, all h
print('\n--- fixed START window (t in 2025-01..2025-05), all firms ---')
sub=mo
W={c:sub.pivot_table(index=['org','cat'],columns='m',values=c) for c in ('gap','resid')}
for h in range(1,13):
    xs,ys,og=[],[],[]
    for i in range(0,5):
        if i+h>=len(months): continue
        a,b=W['gap'].get(months[i]),W['gap'].get(months[i+h])
        if a is None or b is None: continue
        k=a.notna()&b.notna()
        xs.append(a[k].values); ys.append(b[k].values); og+= [z[0] for z in a[k].index]
    if not xs: continue
    x=np.concatenate(xs); y=np.concatenate(ys)
    b1=np.polyfit(x,y,1)[0]
    print(f'  h={h:>2}  gap slope={b1:+.3f}  n={len(x)}')

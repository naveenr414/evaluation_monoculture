"""Spread among the three frontier labs' flagships, and variants."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'experiments')
from common import load_panel, flagships, CORE
DC={'coding':'Coding','math':'Math','hard_prompts':'Hard Prompts',
    'instruction_following':'Instr. Following','creative_writing':'Creative Writing',
    'longer_query':'Longer Query','multi_turn':'Multi-Turn','english':'English'}
END='2026-03-19'
d=load_panel(cats=CORE, firms_only=False, since=None)
d=d[(d.dt>=pd.Timestamp('2024-06-01'))&(d.dt<=pd.Timestamp(END))]
flag=flagships(d)
rat=d.pivot_table(index=['dt','model_name'],columns='category',values='rating')
dates=sorted(d.dt.unique())

SETS={'big3':['anthropic','openai','google'],
      'big5':['anthropic','openai','google','xai','deepseek'],
      'big3+alibaba':['anthropic','openai','google','alibaba']}

rows=[]
for t in dates:
    for lab,orgs in SETS.items():
        for cat in CORE:
            v=[]
            for o in orgs:
                m=flag.get((t,o))
                if m is None: continue
                try: x=rat.loc[(t,m)][cat]
                except KeyError: continue
                if pd.notna(x): v.append(float(x))
            if len(v)<len(orgs): continue
            v=np.array(v)
            rows.append(dict(dt=t,set=lab,cat=cat,sd=v.std(ddof=1),
                             rng=v.max()-v.min(),n=len(v)))
S=pd.DataFrame(rows); S['q']=pd.PeriodIndex(S.dt,freq='Q')
print('quarterly mean, pooled over the 8 categories\n')
print(S.groupby(['set','q'])[['sd','rng']].mean().unstack(0).round(1).to_string())
print('\ntrend per year')
for lab in SETS:
    sub=S[S.set==lab]
    x=(sub.dt-sub.dt.min()).dt.days.values/365.25
    out=[]
    for c in ('sd','rng'):
        b1=np.polyfit(x,sub[c].values,1)[0]; r=np.corrcoef(x,sub[c].values)[0,1]
        out.append(f'{c} {b1:+6.1f} (r {r:+.2f})')
    print(f'  {lab:<13} '+'   '.join(out)+f'   n={len(sub)}')

print('\nbig3 only: range (max-min) by category, first 20 vs last 20 snapshots')
for cat in CORE:
    s=S[(S.set=='big3')&(S.cat==cat)].sort_values('dt')
    x=(s.dt-s.dt.min()).dt.days.values/365.25
    b1=np.polyfit(x,s.rng.values,1)[0]; r=np.corrcoef(x,s.rng.values)[0,1]
    print(f'  {DC[cat]:<18} {s.rng.iloc[:20].mean():6.1f} -> {s.rng.iloc[-20:].mean():6.1f}'
          f'   {b1:+6.1f}/yr  r {r:+.2f}')

print('\nbig3: who leads each category, share of snapshots')
lead=[]
for t in dates:
    for cat in CORE:
        best,bv=None,-np.inf
        for o in SETS['big3']:
            m=flag.get((t,o))
            if m is None: continue
            try: x=rat.loc[(t,m)][cat]
            except KeyError: continue
            if pd.notna(x) and x>bv: best,bv=o,float(x)
        if best: lead.append((cat,best))
L=pd.DataFrame(lead,columns=['cat','firm'])
print((100*L.groupby('cat').firm.value_counts(normalize=True)).unstack()
      .round(1).to_string())

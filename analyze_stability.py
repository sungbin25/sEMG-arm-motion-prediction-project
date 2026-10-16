from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from week2 import read_record, windows, td_features

OUT=Path('outputs/stability'); OUT.mkdir(parents=True,exist_ok=True)
trials=pd.read_csv('outputs/week2/trials.csv')
trials=trials[trials.split.eq('train')].copy()
assert len(trials)==1536
names=['RMS','MAV','WL','ZC','SSC']
rows=[]; envelopes=[]
for filename, group in trials.groupby('file'):
    x,_=read_record(Path('data/source/data')/filename)
    env=[]
    for r in group.itertuples():
        trial=x[r.start:r.stop]
        # 49 disjoint 100 ms bins fit even a 4999-sample trial.
        rms=np.sqrt(np.mean(trial[:4900].reshape(49,100,5)**2,axis=1))
        norm=np.sqrt(np.mean(trial[:4900]**2,axis=0))
        env.append(np.median(rms/np.maximum(norm,1e-12),axis=1))
        ordinal=int(r.trial_id.rsplit('#',1)[1])+1
        phase=('early' if ordinal<=10 else 'middle' if ordinal<=20 else 'late')
        for start in [500,1000,1500]:
            features=np.stack([td_features(w) for w in windows(trial[start:start+2000])]).mean(axis=0)
            rows.append(dict(file=filename,trial_id=r.trial_id,ordinal=ordinal,phase=phase,start=start,
                             **{f'{n}_ch{ch+1}':float(features[i*5+ch]) for i,n in enumerate(names) for ch in range(5)}))
    envelopes.append(np.median(env,axis=0))
data=pd.DataFrame(rows);data.to_csv(OUT/'trial_features.csv',index=False)
features=[f'{n}_ch{ch}' for n in names for ch in range(1,6)]
def symmetric(a,b):
    denom=np.abs(a)+np.abs(b)
    return np.divide(200*(a-b),denom,out=np.zeros_like(denom),where=denom>0)
base=data[data.start.eq(1500)].set_index('trial_id')
segment=[]
for start in [500,1000]:
    other=data[data.start.eq(start)].set_index('trial_id').loc[base.index]
    change=np.abs(symmetric(other[features].to_numpy(),base[features].to_numpy()))
    change=pd.DataFrame(change,columns=features,index=base.index)
    change['file']=base.file
    rec=change.groupby('file')[features].median()
    for name in names:
        vals=rec[[f'{name}_ch{i}' for i in range(1,6)]].median(axis=1)
        segment.append(dict(start=start,feature=name,median=vals.median(),q25=vals.quantile(.25),q75=vals.quantile(.75)))
pd.DataFrame(segment).to_csv(OUT/'segment_summary.csv',index=False)
phase=data[data.start.eq(1500)].groupby(['file','phase'])[features].median()
early=phase.xs('early',level='phase'); late=phase.xs('late',level='phase').loc[early.index]
change=pd.DataFrame(symmetric(late.to_numpy(),early.to_numpy()),index=early.index,columns=features)
record_changes=pd.DataFrame({n:change[[f'{n}_ch{i}' for i in range(1,6)]].median(axis=1) for n in names})
record_changes.to_csv(OUT/'record_time_changes.csv')
counts=data[data.start.eq(1500)].groupby(['file','phase']).size().unstack()
counts.to_csv(OUT/'phase_counts.csv')
env=np.stack(envelopes); t=np.arange(49)*.1+.05
fig,ax=plt.subplots(figsize=(9,3.4))
ax.fill_between(t,np.quantile(env,.25,axis=0),np.quantile(env,.75,axis=0),alpha=.25,label='IQR across 64 records')
ax.plot(t,np.median(env,axis=0),label='Median across records')
ax.axvspan(1.5,3.5,color='orange',alpha=.15,label='Current analysis segment')
ax.set(xlabel='Time relative to mark (s)',ylabel='Normalized RMS',title='Training trials only | record-weighted RMS envelope')
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'envelope.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.4));positions=np.arange(5)
for offset,start,label in [(-.18,500,'0.5-2.5 s vs 1.5-3.5 s'),(.18,1000,'1.0-3.0 s vs 1.5-3.5 s')]:
    v=[next(r for r in segment if r['start']==start and r['feature']==n) for n in names]
    y=np.array([r['median'] for r in v]); lo=y-np.array([r['q25'] for r in v]); hi=np.array([r['q75'] for r in v])-y
    ax.bar(positions+offset,y,.35,yerr=[lo,hi],capsize=3,label=label)
ax.set_xticks(positions,names);ax.set(ylabel='Absolute symmetric change (%)',title='Segment sensitivity | median and IQR across records');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'segment_sensitivity.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.4));ax.boxplot([record_changes[n] for n in names],tick_labels=names,showfliers=True)
ax.axhline(0,color='gray',ls='--');ax.set(ylabel='Signed symmetric change (%)',title='Late trials 21-30 vs early trials 1-10 | 64 records')
fig.tight_layout();fig.savefig(OUT/'time_change.png',dpi=180);plt.close(fig)
result=dict(n_train_trials=len(trials),n_records=len(envelopes),phase_totals=counts.sum().to_dict(),
            phase_min=counts.min().to_dict(),phase_max=counts.max().to_dict(),segment=segment,
            time={n:dict(median=record_changes[n].median(),q25=record_changes[n].quantile(.25),q75=record_changes[n].quantile(.75),positive=int((record_changes[n]>0).sum()),negative=int((record_changes[n]<0).sum())) for n in names},
            envelope_peak_bin_center=float(t[np.argmax(np.median(env,axis=0))]))
(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

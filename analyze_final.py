"""Project saved predictions into semantic matrices and fixed trial voting."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from experiment_loso import decompose

OUT=Path('outputs/final'); OUT.mkdir(parents=True,exist_ok=True)
classes=np.load('outputs/week2/dataset.npz')['class_names'].tolist()
base=pd.read_csv('outputs/week3/holdout_predictions.csv')
loso=pd.read_csv('outputs/loso/predictions.csv')
protocol={'vote':'Joint 8-class plurality of exactly 19 windows; tie: smallest fixed class index',
          'class_order':classes,'selection':'No tuning; all existing predictions included',
          'sources':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['outputs/week3/holdout_predictions.csv','outputs/loso/predictions.csv']}}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

def vote(frame):
    result=[]
    for (model,trial),g in frame.groupby(['model','trial_id']):
        assert len(g)==19 and g.actual.nunique()==1
        count=np.bincount(g.predicted,minlength=8)
        winners=np.flatnonzero(count==count.max())
        result.append(dict(model=model,trial_id=trial,subject=trial.split('_')[0],actual=int(g.actual.iloc[0]),predicted=int(winners[0]),tie=len(winners)>1,votes=int(count.max())))
    return pd.DataFrame(result)

rows=[];subrows=[]; votes=[]
sets={'baseline':base,'loso_matched':loso[loso.original_test],'loso_all':loso}
for name,frame in sets.items():
    v=vote(frame);v['protocol']=name;votes.append(v)
    for unit,f in [('window',frame),('trial',v)]:
        for model,g in f.groupby('model'):
            rows.append(dict(protocol=name,unit=unit,model=model,ties=int(g.tie.sum()) if unit=='trial' else 0,**decompose(g.actual.to_numpy(),g.predicted.to_numpy(),classes)))
            for subject,h in g.groupby(g.trial_id.str.split('_').str[0]):
                subrows.append(dict(protocol=name,unit=unit,model=model,subject=subject,**decompose(h.actual.to_numpy(),h.predicted.to_numpy(),classes)))
pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
pd.DataFrame(subrows).to_csv(OUT/'by_subject.csv',index=False)
pd.concat(votes).to_csv(OUT/'trial_predictions.csv',index=False)
angle=[]
for model in ['SVM','RF','XGB']:
    g=loso[loso.model.eq(model)]
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    for ax,kind,pos,labels in [(axes[0],'gesture',1,['fist','open','pronation','supination']),(axes[1],'angle',0,['30','60'])]:
        lookup=np.array([c.split('_')[pos] for c in classes])
        cm=confusion_matrix(lookup[g.actual],lookup[g.predicted],labels=labels)
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(OUT/f'{kind}_{model}.csv')
        norm=cm/cm.sum(axis=1,keepdims=True)*100
        ax.imshow(norm,vmin=0,vmax=100,cmap='Blues')
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j,i,f'{cm[i,j]:,}\n{norm[i,j]:.1f}%',ha='center',va='center',fontsize=10,color='white' if norm[i,j]>55 else 'black')
        ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,xlabel='Predicted',ylabel='Actual',title=f'{model} LOSO {kind}')
        if kind=='angle':
            angle.append(dict(model=model,predicted_30_share=float(cm[:,0].sum()/cm.sum()),recall_30=float(norm[0,0]/100),recall_60=float(norm[1,1]/100),actual30_pred60=int(cm[0,1]),actual60_pred30=int(cm[1,0])))
    fig.tight_layout();fig.savefig(OUT/f'semantic_{model}.png',dpi=180);plt.close(fig)
pd.DataFrame(angle).to_csv(OUT/'angle_bias.csv',index=False)
print(pd.DataFrame(rows)[['protocol','unit','model','n','accuracy','f1_macro','ties']].to_string(index=False))
print(pd.DataFrame(angle).to_string(index=False))

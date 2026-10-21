"""Independently check saved voting decisions and combined metrics."""
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
base=pd.read_csv('outputs/week3/holdout_predictions.csv')
loso=pd.read_csv('outputs/loso/predictions.csv')
votes=pd.read_csv('outputs/final/trial_predictions.csv')
metrics=pd.read_csv('outputs/final/comparison.csv')
sources={'baseline':base,'loso_matched':loso[loso.original_test],'loso_all':loso}
for protocol,f in sources.items():
    v=votes[votes.protocol.eq(protocol)]
    for r in v.itertuples():
        g=f[f.model.eq(r.model)&f.trial_id.eq(r.trial_id)]
        counts=Counter(g.predicted.tolist());top=max(counts.values())
        winners=sorted(k for k,n in counts.items() if n==top)
        assert len(g)==19 and g.actual.nunique()==1
        assert r.predicted==winners[0] and r.tie==(len(winners)>1)
        assert r.actual==g.actual.iloc[0]
    for unit,frame in [('window',f),('trial',v)]:
        for model,g in frame.groupby('model'):
            m=metrics[(metrics.protocol==protocol)&(metrics.unit==unit)&(metrics.model==model)].iloc[0]
            p,r,f1,_=precision_recall_fscore_support(g.actual,g.predicted,labels=range(8),average='macro',zero_division=0)
            np.testing.assert_allclose([m.accuracy,m.precision_macro,m.recall_macro,m.f1_macro],[accuracy_score(g.actual,g.predicted),p,r,f1])
            assert m.n==len(g)
for model in ['SVM','RF','XGB']:
    a=pd.read_csv(f'outputs/final/angle_{model}.csv',index_col=0).to_numpy()
    assert a.sum()==36480 and (a.sum(axis=1)==18240).all()
print('Verified all 18 comparison rows, 8064 trial decisions, tie rules, and angle supports.')

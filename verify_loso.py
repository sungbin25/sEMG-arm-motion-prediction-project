"""Recalculate saved LOSO and matched-holdout metrics without training."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from experiment_loso import decompose

out=Path('outputs/loso')
config=json.loads((out/'protocol.json').read_text())
assert hashlib.sha256(Path('outputs/week2/dataset.npz').read_bytes()).hexdigest()==config['dataset_sha256']
d=np.load('outputs/week2/dataset.npz',allow_pickle=False)
y=np.r_[d['ytr'],d['yte']]
subjects=np.r_[d['subjects_tr'],d['subjects_te']]
groups=np.r_[d['groups_tr'],d['groups_te']]
classes=d['class_names'].tolist()
pred=pd.read_csv(out/'predictions.csv')
folds=pd.read_csv(out/'fold_metrics.csv')
assert len(pred)==36480*3 and len(folds)==24
assert not pred.duplicated(['model','row_id']).any()
assert folds.groupby('model').size().eq(8).all()
summary=json.loads((out/'summary.json').read_text())
for name in ['SVM','RF','XGB']:
    rows=pred[pred.model.eq(name)].sort_values('row_id')
    np.testing.assert_array_equal(rows.row_id,np.arange(len(y)))
    np.testing.assert_array_equal(rows.subject,subjects)
    np.testing.assert_array_equal(rows.actual,y)
    np.testing.assert_array_equal(rows.trial_id,groups)
    assert rows.groupby('trial_id').size().eq(19).all()
    for subject in sorted(set(subjects)):
        tr=subjects!=subject;te=~tr
        assert set(subjects[tr]).isdisjoint(subjects[te])
        assert set(groups[tr]).isdisjoint(groups[te])
        r=folds[(folds.model==name)&(folds.subject==subject)].iloc[0]
        recalc=decompose(y[te],rows.predicted.to_numpy()[te],classes)
        assert r.train_windows==31920 and r.test_windows==4560
        for k,v in recalc.items():np.testing.assert_allclose(r[k],v)
    for key,mask in [('loso',np.ones(len(y),dtype=bool)),
                     ('matched_test',np.arange(len(y))>=len(d['ytr']))]:
        actual=next(r for r in summary[key] if r['model']==name)
        calc=decompose(y[mask],rows.predicted.to_numpy()[mask],classes)
        for k,v in calc.items():np.testing.assert_allclose(actual[k],v)
        assert actual['n']==sum(actual[k] for k in ['both_correct','gesture_correct_angle_wrong','angle_correct_gesture_wrong','both_wrong'])
        assert actual['gesture_accuracy']>=actual['accuracy']
        assert actual['angle_accuracy']>=actual['accuracy']
print('PASS: 24 folds, subject/trial separation, exact prediction alignment, decomposition and matched-test metrics')

"""Validate saved week3 results without retraining."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from week3 import metrics, error_summary, MODEL_NAMES, holm

out=Path('outputs/week3')
config=json.loads((out/'config.json').read_text())
assert config['dataset_sha256']==hashlib.sha256(Path('outputs/week2/dataset.npz').read_bytes()).hexdigest()
d=np.load('outputs/week2/dataset.npz',allow_pickle=False)
classes=d['class_names'].tolist()
motions={'extor':'supination','pro':'pronation','open':'open','fist':'fist'}
for split in ('tr','te'):
    for group,y in zip(d['groups_'+split],d['y'+split]):
        subject_angle,motion=group.rsplit('#',1)[0].split(' ')
        assert classes[y]==subject_angle.split('_')[1]+'_'+motions[motion]
assign=pd.read_csv(out/'cv_trial_assignments.csv')
assert len(assign)==1536 and assign.trial_id.is_unique
assert set(assign.trial_id)==set(d['groups_tr'])
assert set(assign.trial_id).isdisjoint(d['groups_te'])
cv=pd.read_csv(out/'cv_scores.csv')
assert len(cv)==15 and cv.groupby('model').size().eq(5).all()
for fold,rows in cv.groupby('fold'):
    n=int((assign.validation_fold==fold).sum())
    assert rows.validation_trials.eq(n).all() and rows.validation_windows.eq(n*19).all()
    assert rows.train_trials.eq(1536-n).all()
scores=pd.read_csv(out/'holdout_metrics.csv')
predictions=pd.read_csv(out/'holdout_predictions.csv')
result=json.loads((out/'results.json').read_text(encoding='utf-8'))
for name in MODEL_NAMES:
    rows=predictions[predictions.model.eq(name)].sort_values('test_row')
    assert len(rows)==7296
    np.testing.assert_array_equal(rows.actual,d['yte'])
    np.testing.assert_array_equal(rows.trial_id,d['groups_te'])
    recalc=metrics(rows.actual,rows.predicted)
    saved=scores[scores.model.eq(name)].iloc[0]
    for key,val in recalc.items():np.testing.assert_allclose(val,saved[key])
    errors=error_summary(rows.actual.to_numpy(),rows.predicted.to_numpy(),classes)
    assert errors==result['errors'][name]
    assert sum(errors['error_types'].values())==errors['errors']
tests=pd.read_csv(out/'paired_tests.csv')
for row in tests.itertuples():
    a=cv[cv.model.eq(row.model_a)].sort_values('fold').accuracy
    b=cv[cv.model.eq(row.model_b)].sort_values('fold').accuracy
    t,p=ttest_rel(a,b)
    np.testing.assert_allclose([t,p],[row.t,row.p_raw],rtol=1e-10)
np.testing.assert_allclose(holm(tests.p_raw),tests.p_holm,rtol=1e-10)
print('PASS: dataset hash, filename labels, trial separation, all metrics, confusion matrices, error counts and paired tests')

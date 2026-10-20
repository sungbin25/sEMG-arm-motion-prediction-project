"""Reproduce Step2 week3 with trial-grouped CV and a sealed holdout."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
MODEL_NAMES = ['SVM', 'RF', 'XGB']


def metrics(y, p):
    return dict(accuracy=float(accuracy_score(y, p)),
                precision_macro=float(precision_score(y, p, average='macro', zero_division=0)),
                recall_macro=float(recall_score(y, p, average='macro', zero_division=0)),
                f1_macro=float(f1_score(y, p, average='macro', zero_division=0)))


def models(jobs):
    return {
        'SVM': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1., gamma='scale',
                            decision_function_shape='ovr', cache_size=1024, random_state=SEED)),
        'RF': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100,
                            random_state=SEED, n_jobs=jobs)),
        'XGB': make_pipeline(StandardScaler(), XGBClassifier(n_estimators=400, max_depth=6,
                            learning_rate=.05, subsample=.9, colsample_bytree=.8,
                            random_state=SEED, n_jobs=jobs, tree_method='hist', device='cpu',
                            objective='multi:softprob', eval_metric='mlogloss')),
    }


def trial_folds(y, groups):
    frame = pd.DataFrame({'group': groups, 'y': y})
    if not frame.groupby('group').y.nunique().eq(1).all():
        raise ValueError('Each trial must have one class')
    trials = frame.drop_duplicates('group').reset_index(drop=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (a, b) in enumerate(cv.split(trials.group, trials.y), 1):
        tr = np.flatnonzero(np.isin(groups, trials.group.iloc[a]))
        va = np.flatnonzero(np.isin(groups, trials.group.iloc[b]))
        assert set(groups[tr]).isdisjoint(groups[va])
        yield fold, tr, va


def holm(pvalues):
    """Holm familywise adjustment for finite p-values in the supplied family."""
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError('Holm requires finite p-values')
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def error_summary(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    recall = np.diag(cm) / cm.sum(axis=1)
    counts = dict(same_gesture_wrong_angle=0, same_angle_wrong_gesture=0, both_wrong=0)
    pairs = []
    for a in range(len(classes)):
        for b in range(len(classes)):
            if a == b or not cm[a,b]:
                continue
            aa, ga = classes[a].split('_',1)
            ab, gb = classes[b].split('_',1)
            kind = 'same_gesture_wrong_angle' if ga == gb else 'same_angle_wrong_gesture' if aa == ab else 'both_wrong'
            counts[kind] += int(cm[a,b])
            pairs.append(dict(actual=classes[a], predicted=classes[b], count=int(cm[a,b])))
    pairs.sort(key=lambda r: (-r['count'],r['actual'],r['predicted']))
    return dict(confusion=cm.tolist(), class_recall=recall.tolist(), error_types=counts,
                errors=int((y != pred).sum()), top_errors=pairs[:5],
                best_classes=[classes[i] for i in np.flatnonzero(recall == recall.max())],
                worst_classes=[classes[i] for i in np.flatnonzero(recall == recall.min())])


def main(dataset, output, jobs):
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    d = np.load(dataset,allow_pickle=False)
    X, y, groups = d['Xtr_raw'],d['ytr'],d['groups_tr']
    Xt, yt, gt = d['Xte_raw'],d['yte'],d['groups_te']
    classes = d['class_names'].tolist()
    assert X.shape == (29184,25) and Xt.shape == (7296,25)
    assert np.isfinite(X).all() and np.isfinite(Xt).all()
    assert set(groups).isdisjoint(gt)
    assert len(np.unique(groups)) == 1536 and len(np.unique(gt)) == 384
    assert set(y) == set(yt) == set(range(8))
    prototypes = models(jobs)
    config = dict(dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(), seed=SEED,
                  jobs=jobs, versions={n:importlib.metadata.version(n) for n in
                      ['numpy','pandas','scipy','scikit-learn','xgboost','matplotlib']},
                  python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
                  logical_cpus=os.cpu_count(),
                  parameters={k:{key:val for key,val in m.steps[-1][1].get_params().items()
                                 if isinstance(val,(str,int,float,bool,type(None)))} for k,m in prototypes.items()},
                  cv='5-fold StratifiedKFold over unique training trial IDs; expand IDs to windows',
                  standardization='Pipeline fitted within each training fold; refit on full train for holdout',
                  cv_std_ddof=0, timing='perf_counter; full pipeline; 10 warmups then 100 single-row predict calls',
                  tuning='No hyperparameter or segment search; fixed lecture parameters')
    cfg_path=out/'config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise ValueError('Existing output configuration differs; choose a new output directory')
    cfg_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    fold_rows=[]; assignment=[]
    for fold,tr,va in trial_folds(y,groups):
        for group in np.unique(groups[va]):assignment.append(dict(trial_id=group,validation_fold=fold))
        assert len(np.unique(y[tr]))==len(np.unique(y[va]))==8
        for name,proto in prototypes.items():
            saved=out/f'cv_{name}_fold{fold}.json'
            if saved.exists():
                row=json.loads(saved.read_text())
            else:
                print(f'CV {fold}/5 {name}: fitting {len(tr)} windows',flush=True)
                model=clone(proto); t=time.perf_counter();model.fit(X[tr],y[tr]);elapsed=time.perf_counter()-t
                np.testing.assert_allclose(model.steps[0][1].mean_,X[tr].mean(axis=0),rtol=1e-10,atol=1e-12)
                row=dict(model=name,fold=fold,train_trials=len(np.unique(groups[tr])),
                         validation_trials=len(np.unique(groups[va])),train_windows=len(tr),validation_windows=len(va),
                         fit_seconds=elapsed,**metrics(y[va],model.predict(X[va])))
                saved.write_text(json.dumps(row,indent=2),encoding='utf-8')
                print(f'CV {fold}/5 {name}: acc={row["accuracy"]:.6f}, fit={elapsed:.2f}s',flush=True)
            fold_rows.append(row)
    pd.DataFrame(assignment).to_csv(out/'cv_trial_assignments.csv',index=False)
    folds=pd.DataFrame(fold_rows);folds.to_csv(out/'cv_scores.csv',index=False)
    cv_summary=[]
    for name in MODEL_NAMES:
        rows=folds[folds.model==name]
        cv_summary.append(dict(model=name,accuracy_mean=rows.accuracy.mean(),accuracy_std=rows.accuracy.std(ddof=0),
                               f1_macro_mean=rows.f1_macro.mean(),f1_macro_std=rows.f1_macro.std(ddof=0)))
    pd.DataFrame(cv_summary).to_csv(out/'cv_summary.csv',index=False)
    tests=[]
    for a,b in itertools.combinations(MODEL_NAMES,2):
        av=folds[folds.model==a].sort_values('fold').accuracy.to_numpy()
        bv=folds[folds.model==b].sort_values('fold').accuracy.to_numpy()
        t,p=ttest_rel(av,bv)
        tests.append(dict(model_a=a,model_b=b,n_pairs=5,mean_accuracy_difference=float(np.mean(av-bv)),
                          t=float(t) if np.isfinite(t) else None,p_raw=float(p) if np.isfinite(p) else None))
    if all(r['p_raw'] is not None for r in tests):
        for r,adj in zip(tests,holm([r['p_raw'] for r in tests])):r['p_holm']=float(adj)
    for r in tests:r.setdefault('p_holm',None)
    pd.DataFrame(tests).to_csv(out/'paired_tests.csv',index=False)
    holdout=[]; errors={}; predictions=[]
    for name,proto in prototypes.items():
        print(f'Final holdout {name}: fitting full training data',flush=True)
        model=clone(proto);t=time.perf_counter();model.fit(X,y);fit_time=time.perf_counter()-t
        np.testing.assert_allclose(model.steps[0][1].mean_,X.mean(0),rtol=1e-10,atol=1e-12)
        t=time.perf_counter();pred=model.predict(Xt);batch_time=time.perf_counter()-t
        for _ in range(10):model.predict(Xt[:1])
        timings=[]
        for _ in range(100):
            t=time.perf_counter();model.predict(Xt[:1]);timings.append((time.perf_counter()-t)*1000)
        row=dict(model=name,**metrics(yt,pred),fit_seconds=fit_time,
                 single_sample_mean_ms=float(np.mean(timings)),single_sample_median_ms=float(np.median(timings)),
                 single_sample_p95_ms=float(np.quantile(timings,.95)),batch_seconds=batch_time,
                 batch_ms_per_window=batch_time/len(yt)*1000)
        holdout.append(row); errors[name]=error_summary(yt,pred,classes)
        for i,(actual,predicted,group) in enumerate(zip(yt,pred,gt)):
            predictions.append(dict(model=name,test_row=i,trial_id=group,actual=int(actual),predicted=int(predicted)))
        pd.DataFrame({'milliseconds':timings}).to_csv(out/f'inference_{name}.csv',index=False)
        pd.DataFrame(errors[name]['confusion'],index=classes,columns=classes).to_csv(out/f'confusion_{name}.csv')
        print(f'Holdout {name}: {row}',flush=True)
    pd.DataFrame(holdout).to_csv(out/'holdout_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'holdout_predictions.csv',index=False)
    result=dict(holdout=holdout,cv=cv_summary,paired_tests=tests,errors=errors,class_names=classes,
                cv_train_only=True,overlapping_holdout_trial_ids=0,cv_validation_trial_count=len(assignment),
                report_note='Fold scores have dependent training sets; paired t-tests are exploratory, not definitive inference.')
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    plot_results(out,result,folds)
    print('COMPLETE',out,flush=True)


def plot_results(out,result,folds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,e in result['errors'].items():
        cm=np.array(e['confusion']);fig,ax=plt.subplots(figsize=(8.8,7))
        im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=912)
        ax.set_xticks(range(8),result['class_names'],rotation=40,ha='right')
        ax.set_yticks(range(8),result['class_names'])
        ax.set(xlabel='Predicted class',ylabel='True class',title=f'{name} | holdout windows | support 912/class')
        for i in range(8):
            for j in range(8):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,color='white' if cm[i,j]>456 else 'black')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/f'confusion_{name}.png',dpi=180);plt.close(fig)
    # Lecture's submission filename refers to the RF matrix.
    import shutil
    shutil.copyfile(out/'confusion_RF.png',out/'confusion8.png')
    fig,ax=plt.subplots(figsize=(8.5,3.6))
    for name in MODEL_NAMES:
        rows=folds[folds.model==name].sort_values('fold');ax.plot(rows.fold,rows.accuracy*100,marker='o',label=name)
    ax.set(xlabel='Validation fold (same trial IDs for all models)',ylabel='Accuracy (%)',xticks=range(1,6),title='Training-only trial-grouped 5-fold cross-validation')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'cv_accuracy.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('outputs/week2/dataset.npz'))
    parser.add_argument('--output',type=Path,default=Path('outputs/week3'))
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args();main(args.dataset,args.output,args.jobs)

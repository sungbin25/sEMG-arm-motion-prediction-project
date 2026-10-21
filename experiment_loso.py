"""Fixed-protocol exploratory LOSO, plus decomposition of existing predictions."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from week3 import models, metrics

OUT = Path('outputs/loso')
NAMES = ['SVM', 'RF', 'XGB']


def decompose(y, pred, classes):
    labels = [c.split('_', 1) for c in classes]
    angles = np.array([c[0] for c in labels])
    gestures = np.array([c[1] for c in labels])
    a = angles[y] == angles[pred]
    g = gestures[y] == gestures[pred]
    return dict(n=len(y), both_correct=int((a & g).sum()),
                gesture_correct_angle_wrong=int((g & ~a).sum()),
                angle_correct_gesture_wrong=int((a & ~g).sum()),
                both_wrong=int((~a & ~g).sum()),
                gesture_accuracy=float(g.mean()), angle_accuracy=float(a.mean()),
                **metrics(y, pred))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path('outputs/week2/dataset.npz')
    d = np.load(path, allow_pickle=False)
    X = np.concatenate([d['Xtr_raw'], d['Xte_raw']])
    y = np.concatenate([d['ytr'], d['yte']])
    groups = np.concatenate([d['groups_tr'], d['groups_te']])
    subjects = np.concatenate([d['subjects_tr'], d['subjects_te']])
    classes = d['class_names'].tolist()
    old_test = np.arange(len(X)) >= len(d['Xtr_raw'])
    assert X.shape == (36480, 25) and len(np.unique(groups)) == 1920
    assert np.isfinite(X).all()
    assert set(subjects) == set('abcdefgh')
    assert np.all(np.unique(groups, return_counts=True)[1] == 19)
    assert all(g.split('_')[0] == s for g, s in zip(groups, subjects))
    protos = models(4)
    protocol = dict(date='2026-10-21', dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    input='Unscaled week2 features; original train and test combined for a distinct exploratory protocol',
                    evaluation='8 LOSO folds; 7 subjects train, 1 held out; scaler fit only on training subjects',
                    n_windows=36480, n_trials=1920, windows_per_subject=4560,
                    parameters='Identical to week3.models(4); no tuning or segment changes',
                    week3_code_sha256=hashlib.sha256(Path('week3.py').read_bytes()).hexdigest(),
                    original_test_comparison='Also compare LOSO vs existing holdout on the exact same 7296 windows',
                    limitations='Not independent confirmation after previous data exploration; training size differs. Subject and recording effects are confounded. No new sensor measurements.',
                    versions=json.loads(Path('outputs/week3/config.json').read_text())['versions'])
    f = OUT/'protocol.json'
    if f.exists():
        assert json.loads(f.read_text()) == protocol, 'Protocol changed; use a different output folder.'
    f.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    baseline = pd.read_csv('outputs/week3/holdout_predictions.csv')
    base_rows = []
    base_subject = []
    for name in NAMES:
        rows = baseline[baseline.model.eq(name)].sort_values('test_row')
        np.testing.assert_array_equal(rows.actual, d['yte'])
        np.testing.assert_array_equal(rows.trial_id, d['groups_te'])
        base_rows.append(dict(model=name, **decompose(d['yte'], rows.predicted.to_numpy(), classes)))
        for subject in sorted(set(subjects)):
            mask = d['subjects_te'] == subject
            base_subject.append(dict(model=name, subject=subject,
                **decompose(d['yte'][mask], rows.predicted.to_numpy()[mask], classes)))
    pd.DataFrame(base_rows).to_csv(OUT/'baseline_decomposition.csv', index=False)
    pd.DataFrame(base_subject).to_csv(OUT/'baseline_by_subject.csv', index=False)
    all_predictions = []
    folds = []
    for subject in sorted(set(subjects)):
        tr = np.flatnonzero(subjects != subject)
        te = np.flatnonzero(subjects == subject)
        assert len(te) == 4560 and len(tr) == 31920
        assert set(subjects[tr]).isdisjoint(subjects[te])
        assert set(groups[tr]).isdisjoint(groups[te])
        assert np.all(np.bincount(y[te], minlength=8) == 570)
        for name in NAMES:
            cache = OUT/f'pred_{name}_{subject}.csv'
            info = OUT/f'fold_{name}_{subject}.json'
            if cache.exists() and info.exists():
                predictions = pd.read_csv(cache)
                row = json.loads(info.read_text())
                np.testing.assert_array_equal(predictions.row_id, te)
                np.testing.assert_array_equal(predictions.actual, y[te])
            else:
                print(f'LOSO {subject} / {name}: train 7 subjects, test {subject}', flush=True)
                model = clone(protos[name])
                start = time.perf_counter()
                model.fit(X[tr], y[tr])
                elapsed = time.perf_counter()-start
                np.testing.assert_allclose(model.steps[0][1].mean_, X[tr].mean(0), rtol=1e-10, atol=1e-12)
                pred = model.predict(X[te])
                predictions = pd.DataFrame(dict(row_id=te, subject=subject, model=name,
                    trial_id=groups[te], actual=y[te], predicted=pred, original_test=old_test[te]))
                row = dict(model=name, subject=subject, train_windows=len(tr), test_windows=len(te),
                           fit_seconds=elapsed, **decompose(y[te], pred, classes))
                predictions.to_csv(cache, index=False)
                info.write_text(json.dumps(row, indent=2), encoding='utf-8')
                print(f'  completed accuracy={row["accuracy"]:.4f} gesture={row["gesture_accuracy"]:.4f} angle={row["angle_accuracy"]:.4f} fit={elapsed:.1f}s', flush=True)
            all_predictions.append(predictions)
            folds.append(row)
    frame = pd.concat(all_predictions, ignore_index=True)
    frame.to_csv(OUT/'predictions.csv', index=False)
    pd.DataFrame(folds).to_csv(OUT/'fold_metrics.csv', index=False)
    totals = []
    matched = []
    for name in NAMES:
        rows = frame[frame.model.eq(name)].sort_values('row_id')
        np.testing.assert_array_equal(rows.row_id, np.arange(len(y)))
        np.testing.assert_array_equal(rows.actual, y)
        pred = rows.predicted.to_numpy()
        fs = [r for r in folds if r['model'] == name]
        totals.append(dict(model=name, **decompose(y, pred, classes),
                           subject_accuracy_mean=float(np.mean([r['accuracy'] for r in fs])),
                           subject_accuracy_sd=float(np.std([r['accuracy'] for r in fs])),
                           subject_f1_mean=float(np.mean([r['f1_macro'] for r in fs])),
                           subject_f1_sd=float(np.std([r['f1_macro'] for r in fs]))))
        matched.append(dict(model=name, **decompose(y[old_test], pred[old_test], classes)))
        pd.DataFrame(confusion_matrix(y, pred), index=classes, columns=classes).to_csv(OUT/f'confusion8_{name}.csv')
        # Semantic projections of the existing joint-class predictions, not new classifiers.
        for kind, mapping in [('gesture', [c.split('_')[1] for c in classes]),
                              ('angle', [c.split('_')[0] for c in classes])]:
            mapping = np.array(mapping)
            labels = sorted(set(mapping))
            pd.DataFrame(confusion_matrix(mapping[y], mapping[pred], labels=labels),
                         index=labels, columns=labels).to_csv(OUT/f'confusion_{kind}_{name}.csv')
    pd.DataFrame(totals).to_csv(OUT/'loso_summary.csv', index=False)
    pd.DataFrame(matched).to_csv(OUT/'matched_original_test.csv', index=False)
    summary = dict(baseline=base_rows, loso=totals, matched_test=matched)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    make_plots(folds, summary)
    print('ALL 24 FITS COMPLETE', flush=True)


def make_plots(folds, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(folds)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ['accuracy', 'gesture_accuracy', 'angle_accuracy']):
        for name in NAMES:
            rows = frame[frame.model.eq(name)].sort_values('subject')
            ax.plot(rows.subject, rows[metric]*100, marker='o', label=name)
        ax.set(title=metric.replace('_', ' '), xlabel='Held-out subject', ylabel='Accuracy (%)', ylim=(0, 100))
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.tight_layout();fig.savefig(OUT/'subject_performance.png', dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(3)
    for offset, key, label in [(-.25,'baseline','Original holdout'),(0,'matched_test','LOSO on same holdout windows'),(.25,'loso','LOSO all windows')]:
        ax.bar(x+offset,[r['accuracy']*100 for r in summary[key]],width=.25,label=label)
    ax.set_xticks(x,NAMES);ax.set(ylabel='8-class accuracy (%)',ylim=(0,100))
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'protocol_comparison.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    run()

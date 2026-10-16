"""Step2 week2: trial-grouped, 25-dimensional time-domain feature dataset."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FS, WIN, HOP = 1000, 200, 100
CHANNELS = [f"bandpass {i}" for i in range(1, 6)]
MOTIONS = {"fist": "fist", "open": "open", "pro": "pronation", "extor": "supination"}
CLASSES = [f"{angle}_{motion}" for angle in (30, 60)
           for motion in ("fist", "open", "pronation", "supination")]
FEATURE_NAMES = [f"{name}_ch{ch}" for name in ("RMS", "MAV", "WL", "ZC", "SSC")
                 for ch in range(1, 6)]


def filter_emg(x):
    """Lecture demonstration; input CSV is already filtered by its publisher."""
    bn, an = iirnotch(60, 30, fs=FS)
    b, a = butter(4, [20, 450], btype="bandpass", fs=FS)
    return filtfilt(b, a, filtfilt(bn, an, x, axis=0), axis=0)


def td_features(w, th=0.01):
    """Feature-major order; SSC has no amplitude threshold, as in lecture."""
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (WIN, 5) or not np.isfinite(w).all():
        raise ValueError("Expected finite (200, 5) window")
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum(((w[:-1] * w[1:]) < 0) & (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1] * d[1:]) < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])


def windows(segment):
    if segment.shape != (2000, 5):
        raise ValueError(f"Expected (2000, 5), got {segment.shape}")
    return np.stack([segment[i:i + WIN] for i in range(0, 2000 - WIN + 1, HOP)])


def read_record(path):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    values = frame[CHANNELS + ["time", "mark"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite data: {path.name}")
    if not np.allclose(np.diff(frame['time']), 1 / FS, atol=1e-6, rtol=0):
        raise ValueError(f"Sampling interval is not 1 ms: {path.name}")
    mark = frame['mark'].to_numpy()
    if not np.isin(mark, [0, 1]).all():
        raise ValueError(f"Nonbinary marker: {path.name}")
    starts = np.flatnonzero((mark == 1) & np.r_[True, mark[:-1] == 0])
    return frame[CHANNELS].to_numpy(dtype=float), starts


def inventory(data_dir):
    rows, files = [], []
    for path in sorted(Path(data_dir).glob('*.csv')):
        match = re.fullmatch(r'([a-h])_(30|60) (fist|open|pro|extor)', path.stem)
        if not match:
            raise ValueError(f"Unexpected filename: {path.name}")
        subject, angle, motion = match.groups()
        x, starts = read_record(path)
        if len(starts) != 30:
            raise ValueError(f"Expected 30 trials: {path.name}: {len(starts)}")
        label = f'{angle}_{MOTIONS[motion]}'
        files.append(dict(file=path.name, samples=len(x), markers=len(starts),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for trial, start in enumerate(starts):
            # Stop at the next marker to prevent even a 1-sample overlap.
            stop = min(int(start) + 5000, len(x), int(starts[trial + 1])
                       if trial + 1 < len(starts) else len(x))
            rows.append(dict(trial_id=f'{path.stem}#{trial:02d}', file=path.name,
                             subject=subject, label=label, y=CLASSES.index(label),
                             start=int(start), stop=stop))
    frame = pd.DataFrame(rows)
    if len(files) != 64 or len(frame) != 1920:
        raise ValueError(f'Expected complete 64-record/1920-trial dataset; got {len(files)}/{len(frame)}')
    return frame, files


def split_trials(trials, seed=42):
    train, test = train_test_split(np.arange(len(trials)), test_size=0.2,
                                   stratify=trials.y, random_state=seed)
    result = trials.copy()
    result['split'] = 'train'
    result.loc[test, 'split'] = 'test'
    assert set(result.loc[train, 'trial_id']).isdisjoint(result.loc[test, 'trial_id'])
    return result


def build(data_dir, output_dir, segment_start=1500, filter_mode='already-filtered', seed=42):
    if not 0 <= segment_start <= 3000:
        raise ValueError('segment_start must be between 0 and 3000 samples')
    if filter_mode not in ('already-filtered', 'refilter-demo'):
        raise ValueError('Unknown filter mode')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trials, source_files = inventory(data_dir)
    trials = split_trials(trials, seed)  # BEFORE windowing and feature extraction.
    features, meta, example = [], [], None
    for filename, record_trials in trials.groupby('file', sort=False):
        signal, _ = read_record(Path(data_dir) / filename)
        for row in record_trials.itertuples(index=False):
            trial = signal[row.start:row.stop]
            if segment_start + 2000 > len(trial):
                raise ValueError(f'Incomplete analysis segment: {row.trial_id}')
            processed = filter_emg(trial) if filter_mode == 'refilter-demo' else trial
            segment = processed[segment_start:segment_start + 2000]
            trial_windows = windows(segment)
            if example is None and row.split == 'train':
                example = (trial.copy(), row.trial_id)
            for number, w in enumerate(trial_windows):
                features.append(td_features(w))
                meta.append(dict(trial_id=row.trial_id, subject=row.subject, file=filename,
                                 label=row.label, y=row.y, split=row.split, window=number,
                                 sample_start=row.start + segment_start + number * HOP,
                                 zc_zero=float(td_features(w, 0)[15:20].mean())))
    X = np.stack(features)
    metadata = pd.DataFrame(meta)
    train_mask = (metadata.split == 'train').to_numpy()
    Xtr_raw, Xte_raw = X[train_mask], X[~train_mask]
    scaler = StandardScaler().fit(Xtr_raw)  # Test statistics never used here.
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)
    active = scaler.var_ > 0
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
    assert np.allclose(Xtr.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(Xtr[:, active].std(axis=0), 1, atol=1e-10)
    assert np.all(Xtr[:, ~active] == 0)
    tr_meta, te_meta = metadata[train_mask], metadata[~train_mask]
    assert set(tr_meta.trial_id).isdisjoint(te_meta.trial_id)
    np.savez_compressed(out / 'dataset.npz', Xtr=Xtr, Xte=Xte,
                        ytr=tr_meta.y.to_numpy(), yte=te_meta.y.to_numpy(),
                        Xtr_raw=Xtr_raw, Xte_raw=Xte_raw,
                        groups_tr=tr_meta.trial_id.to_numpy(dtype=str),
                        groups_te=te_meta.trial_id.to_numpy(dtype=str),
                        subjects_tr=tr_meta.subject.to_numpy(dtype=str),
                        subjects_te=te_meta.subject.to_numpy(dtype=str),
                        feature_names=np.array(FEATURE_NAMES), class_names=np.array(CLASSES),
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_)
    trials.to_csv(out / 'trials.csv', index=False)
    metadata.drop(columns='zc_zero').to_csv(out / 'windows.csv', index=False)
    distribution = metadata.groupby(['label', 'split']).size().unstack(fill_value=0).reindex(CLASSES)
    distribution.to_csv(out / 'class_distribution.csv')
    diagnostics(example, segment_start, out)
    source_commit = None
    try:
        source_commit = subprocess.check_output(['git', '-C', str(data_dir), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    summary = dict(source_url='https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset',
                   source_commit=source_commit, source_files=source_files,
                   seed=seed, fs=FS, filter_mode=filter_mode, segment_start_samples=segment_start,
                   segment_note='Lecture assumption relative to mark; movement alignment unverified.',
                   split='Stratified marker-trial holdout 80:20; subjects and recordings shared.',
                   n_records=64, n_trials=len(trials), n_train_trials=int((trials.split == 'train').sum()),
                   n_test_trials=int((trials.split == 'test').sum()),
                   segment_shape=[2000, 5], windows_per_trial=19, window_shape=[19, 200, 5],
                   Xtr_shape=list(Xtr.shape), Xte_shape=list(Xte.shape),
                   zc_threshold=0.01, ssc_threshold=None,
                   mean_zc_threshold_zero=float(metadata.zc_zero.mean()),
                   mean_zc_threshold_001=float(X[:, 15:20].mean()),
                   max_abs_train_mean=float(np.abs(Xtr.mean(axis=0)).max()),
                   max_abs_train_std_error=float(np.abs(Xtr[:, active].std(axis=0) - 1).max()),
                   constant_features=np.array(FEATURE_NAMES)[~active].tolist(),
                   overlapping_trial_ids=0,
                   class_distribution=distribution.to_dict(orient='index'))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    import importlib.metadata
    versions = {p: importlib.metadata.version(p) for p in
                ('numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'nbformat', 'nbclient', 'ipykernel')}
    (out / 'environment.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    return summary


def diagnostics(example, segment_start, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x, trial_id = example
    extra = filter_emg(x)
    pd.DataFrame({'channel': CHANNELS, 'published_std': x.std(axis=0),
                  'additional_filter_std': extra.std(axis=0)}).to_csv(out / 'filter_std.csv', index=False)
    fig, axes = plt.subplots(5, 2, figsize=(12, 12))
    for ch in range(5):
        axes[ch, 0].plot(np.arange(len(x)) / FS, x[:, ch], lw=.5)
        axes[ch, 0].axvspan(segment_start / FS, (segment_start + 2000) / FS, alpha=.2, color='orange')
        axes[ch, 0].set(ylabel=f'Ch {ch + 1} (source units)', xlabel='Time relative to mark (s)')
        for data, label in ((x, 'Published (already filtered)'), (extra, 'Additional lecture filter')):
            freq, power = welch(data[:, ch], fs=FS, nperseg=1000)
            axes[ch, 1].semilogy(freq, power + np.finfo(float).tiny, label=label)
        axes[ch, 1].set(xlabel='Frequency (Hz)', ylabel='PSD')
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f'Train example: {trial_id}; shaded segment = provisional lecture assumption')
    fig.tight_layout()
    fig.savefig(out / 'filter_and_segment.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/source/data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/week2'))
    parser.add_argument('--segment-start', type=int, default=1500)
    parser.add_argument('--filter-mode', choices=['already-filtered', 'refilter-demo'], default='already-filtered')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = build(args.data_dir, args.output_dir, args.segment_start, args.filter_mode, args.seed)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_files'}, ensure_ascii=False, indent=2))

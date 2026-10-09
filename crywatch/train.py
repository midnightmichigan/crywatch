"""Train CryWatch: stratified split, class-weighted MLP, honest test metrics.

Usage: .venv/bin/python crywatch/train.py
Outputs: weights.npz, labels.json, metrics.json, features.npz (cache)
"""
import json
import os
import numpy as np

from features import extract_features, feature_dim
from model import TinyMLP, train

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'data', 'baby-cry', 'baby_crying_sound')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'features.npz')
OUT = os.path.dirname(os.path.abspath(__file__))


def load_dataset():
    classes = sorted(d for d in os.listdir(DATA_ROOT)
                     if os.path.isdir(os.path.join(DATA_ROOT, d)))
    label_of = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        d = os.path.join(DATA_ROOT, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith('.wav'):
                paths.append(os.path.join(d, f))
                labels.append(label_of[c])
    return classes, np.array(paths), np.array(labels)


def main():
    classes, paths, labels = load_dataset()
    print(f'{len(paths)} clips, {len(classes)} classes: {classes}', flush=True)

    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        X = z['X']
        paths = z['paths']
        labels = z['labels']
        print('loaded cached features', flush=True)
    else:
        feats = []
        skipped = 0
        good_paths, good_labels = [], []
        for p, lb in zip(paths, labels):
            try:
                feats.append(extract_features(p))
                good_paths.append(p)
                good_labels.append(lb)
            except Exception as e:
                skipped += 1
        if skipped:
            print(f'skipped {skipped} unreadable files', flush=True)
        paths = np.array(good_paths)
        labels = np.array(good_labels, dtype=int)
        X = np.stack(feats)
        np.savez_compressed(CACHE, X=X, paths=np.array(paths),
                            labels=np.array(labels))
        print('features cached', flush=True)
    print('feature dim:', X.shape[1], 'expected:', feature_dim(), flush=True)

    # stratified 70/15/15 split, fixed seed
    rng = np.random.default_rng(7)
    tr_idx, va_idx, te_idx = [], [], []
    for c in range(len(classes)):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_te = max(1, int(0.15 * n))
        n_va = max(1, int(0.15 * n))
        te_idx += idx[:n_te].tolist()
        va_idx += idx[n_te:n_te + n_va].tolist()
        tr_idx += idx[n_te + n_va:].tolist()
    tr_idx, va_idx, te_idx = map(np.array, (tr_idx, va_idx, te_idx))
    Xtr, ytr = X[tr_idx], labels[tr_idx]
    Xva, yva = X[va_idx], labels[va_idx]
    Xte, yte = X[te_idx], labels[te_idx]
    print(f'split: train {len(Xtr)} / val {len(Xva)} / test {len(Xte)}',
          flush=True)

    # class weights (inverse frequency) for the imbalanced set
    counts = np.bincount(ytr, minlength=len(classes)).astype(float)
    cw = counts.sum() / (len(classes) * np.maximum(counts, 1))

    model = TinyMLP(X.shape[1], len(classes), hidden=(96, 48))
    print('params:', model.params(), flush=True)
    train(model, Xtr, ytr, Xva, yva, epochs=150, lr=3e-3,
          batch=64, class_weight=cw, verbose=True)

    # ---- honest test metrics ----
    proba = model.predict_proba(Xte)
    pred = proba.argmax(axis=1)
    acc = (pred == yte).mean()
    per_class = {}
    for i, c in enumerate(classes):
        m = yte == i
        per_class[c] = float((pred[m] == i).mean()) if m.sum() else 0.0
    # cry-vs-non-cry detection view
    non_cry = {'laugh', 'noise', 'silence'}
    cry_true = np.array([c not in non_cry for c in np.array(classes)[yte]])
    cry_pred = np.array([c not in non_cry for c in np.array(classes)[pred]])
    det_acc = float((cry_true == cry_pred).mean())

    metrics = {
        'test_accuracy': float(acc),
        'cry_detection_accuracy': det_acc,
        'per_class_accuracy': per_class,
        'n_test': int(len(Xte)),
        'classes': classes,
        'params': int(model.params()),
    }
    print(json.dumps(metrics, indent=2), flush=True)

    model.save(os.path.join(OUT, '..', 'weights.npz'))
    with open(os.path.join(OUT, '..', 'labels.json'), 'w') as f:
        json.dump(classes, f)
    with open(os.path.join(OUT, '..', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    # save test split file list for the demo
    with open(os.path.join(OUT, '..', 'test_files.json'), 'w') as f:
        json.dump([paths[i] for i in te_idx], f)
    print('saved weights.npz, labels.json, metrics.json', flush=True)


if __name__ == '__main__':
    main()

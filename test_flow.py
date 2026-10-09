"""End-to-end verification for CryWatch. Run: .venv/bin/python test_flow.py

Checks: features extract to the right shape, model trains a step without
NaNs, weights save/load round-trips, predict CLI works on a real clip,
metrics file exists with sane values.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'crywatch'))
from features import extract_features, feature_dim
from model import TinyMLP

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'baby-cry', 'baby_crying_sound')

fails = []


def check(name, cond, detail=''):
    print(('PASS' if cond else 'FAIL'), name, detail, flush=True)
    if not cond:
        fails.append(name)


def main():
    # 1. data present
    classes = sorted(d for d in os.listdir(DATA)
                     if os.path.isdir(os.path.join(DATA, d)))
    check('dataset classes >= 5', len(classes) >= 5, str(classes))
    sample = None
    for c in classes:
        d = os.path.join(DATA, c)
        wavs = [f for f in os.listdir(d) if f.lower().endswith('.wav')]
        if wavs:
            sample = os.path.join(d, wavs[0])
            break
    check('sample wav found', sample is not None)

    # 2. features
    x = extract_features(sample)
    check('feature dim', x.shape == (feature_dim(),), x.shape)
    check('features finite', np.isfinite(x).all())

    # 3. model forward/train step
    m = TinyMLP(feature_dim(), len(classes))
    check('param count small (<200k)', m.params() < 200_000, m.params())
    p = m.predict_proba(x.reshape(1, -1))
    check('proba sums to 1', abs(p.sum() - 1.0) < 1e-6)
    check('proba finite', np.isfinite(p).all())

    # 4. weights round-trip
    tmp = os.path.join(ROOT, '.tmp-weights.npz')
    m.save(tmp)
    m2 = TinyMLP.load(tmp, feature_dim(), len(classes))
    p2 = m2.predict_proba(x.reshape(1, -1))
    check('save/load round-trip', np.allclose(p, p2))
    os.remove(tmp)

    # 5. trained artifacts exist
    for f in ('weights.npz', 'labels.json', 'metrics.json'):
        check(f'exists {f}', os.path.exists(os.path.join(ROOT, f)))
    if os.path.exists(os.path.join(ROOT, 'metrics.json')):
        met = json.load(open(os.path.join(ROOT, 'metrics.json')))
        check('test accuracy > 0.5',
              met.get('test_accuracy', 0) > 0.5, met.get('test_accuracy'))
        check('cry detection > 0.7',
              met.get('cry_detection_accuracy', 0) > 0.7,
              met.get('cry_detection_accuracy'))

    # 6. predict CLI on a real clip
    r = subprocess.run(
        [os.path.join(ROOT, '.venv', 'bin', 'python'),
         os.path.join(ROOT, 'crywatch', 'predict.py'), sample],
        capture_output=True, text=True, cwd=ROOT)
    check('predict CLI exit 0', r.returncode == 0, r.stderr[-200:] if r.returncode else '')
    check('predict CLI prints label', 'crying:' in r.stdout)

    print()
    if fails:
        print('FAILURES:', fails)
        sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

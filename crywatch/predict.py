"""CryWatch inference CLI — what the demo video shows.

Usage: .venv/bin/python crywatch/predict.py <audio.wav> [--top 3]
Prints the predicted cry reason + confidence, and a cry/non-cry verdict.
Runs on CPU in milliseconds; the same forward pass targets the
UGen300 (Hailo-10H) via ONNX in Stage II.
"""
import json
import os
import sys
import time

import numpy as np

from features import extract_features, feature_dim
from model import TinyMLP

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

NON_CRY = {'laugh', 'noise', 'silence'}

# parent-friendly guidance per predicted reason
GUIDANCE = {
    'hungry': 'Likely hunger — check when the last feed was.',
    'belly pain': 'Sounds like belly pain — try burping, bicycle legs, or a warm bath.',
    'discomfort': 'Sounds like discomfort — check diaper, clothing, and room temperature.',
    'tired': 'Likely tired — dim lights, white noise, swaddle.',
    'burping': 'May need burping — hold upright and pat gently.',
    'cold_hot': 'Possibly too cold or too hot — check neck/chest temperature.',
    'laugh': 'No cry detected — happy sounds.',
    'noise': 'No cry detected — background noise.',
    'silence': 'No cry detected — quiet.',
}


def load_model():
    with open(os.path.join(ROOT, 'labels.json')) as f:
        labels = json.load(f)
    model = TinyMLP.load(os.path.join(ROOT, 'weights.npz'),
                         in_dim=feature_dim(), n_classes=len(labels),
                         hidden=(96, 48))
    return model, labels


def classify(path, model=None, labels=None, top=3):
    if model is None:
        model, labels = load_model()
    t0 = time.perf_counter()
    x = extract_features(path)
    proba = model.predict_proba(x)[0]
    dt = (time.perf_counter() - t0) * 1000
    order = np.argsort(proba)[::-1][:top]
    results = [(labels[i], float(proba[i])) for i in order]
    crying = results[0][0] not in NON_CRY
    return {
        'file': os.path.basename(path),
        'crying': crying,
        'top': results,
        'guidance': GUIDANCE.get(results[0][0], ''),
        'latency_ms': dt,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    model, labels = load_model()
    for path in sys.argv[1:]:
        if path == '--top':
            continue
        r = classify(path, model, labels)
        print(f"\n{r['file']}")
        print(f"  crying: {r['crying']}  ({r['latency_ms']:.1f} ms on CPU)")
        for lab, p in r['top']:
            print(f"  {lab:12s} {p * 100:5.1f}%")
        print(f"  -> {r['guidance']}")


if __name__ == '__main__':
    main()

"""Tiny MLP classifier in pure numpy — the whole learned model.

Architecture: 480 -> 128 -> 64 -> 9 (ReLU, softmax). ~70k params,
~280 KB fp32 / ~70 KB int8. Trains on CPU in minutes; the forward pass
is a few matrix multiplies — ideal for the UGen300's Hailo-10H NPU
(INT8, 20 TOPS) in Stage II via ONNX export.
"""
import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class TinyMLP:
    def __init__(self, in_dim, n_classes, hidden=(128, 64), seed=0):
        rng = np.random.default_rng(seed)
        dims = [in_dim, *hidden, n_classes]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            # He init
            self.W.append(rng.normal(0, np.sqrt(2.0 / dims[i]),
                                     (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))
        self.n_classes = n_classes

    def forward(self, X):
        a = X
        self._cache = [a]
        self._pre = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            self._pre.append(z)
            a = _relu(z) if i < len(self.W) - 1 else z
            self._cache.append(a)
        return _softmax(a)

    def params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def predict_proba(self, X):
        return self.forward(np.atleast_2d(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path):
        data = {f'W{i}': w for i, w in enumerate(self.W)}
        data.update({f'b{i}': b for i, b in enumerate(self.b)})
        np.savez(path, **data)

    @classmethod
    def load(cls, path, in_dim, n_classes, hidden=(128, 64)):
        m = cls(in_dim, n_classes, hidden)
        z = np.load(path)
        m.W = [z[f'W{i}'] for i in range(len(hidden) + 1)]
        m.b = [z[f'b{i}'] for i in range(len(hidden) + 1)]
        return m


def train(model, Xtr, ytr, Xva, yva, epochs=120, lr=3e-3,
          batch=64, class_weight=None, seed=1, verbose=True, l2=1e-4):
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    # Adam state
    mW = [np.zeros_like(w) for w in model.W]
    vW = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]
    vb = [np.zeros_like(b) for b in model.b]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    cw = None
    if class_weight is not None:
        cw = np.asarray(class_weight, dtype=float)

    def loss_acc(X, y):
        p = model.forward(X)
        oh = np.zeros_like(p)
        oh[np.arange(len(y)), y] = 1.0
        w = cw[y][:, None] if cw is not None else 1.0
        ce = -(oh * np.log(p + 1e-12) * w).sum(axis=1).mean()
        return ce, (p.argmax(axis=1) == y).mean()

    best_va, best = -1.0, None
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            t += 1
            bi = idx[s:s + batch]
            Xb, yb = Xtr[bi], ytr[bi]
            p = model.forward(Xb)
            oh = np.zeros_like(p)
            oh[np.arange(len(yb)), yb] = 1.0
            w = cw[yb][:, None] if cw is not None else 1.0
            delta = (p - oh) * w / len(yb)          # dL/dz at logits
            grads_W, grads_b = [], []
            a_prev = model._cache[-2]
            # walk layers backwards
            d = delta
            for i in reversed(range(len(model.W))):
                a_prev = model._cache[i]
                grads_W.append(a_prev.T @ d)
                grads_b.append(d.sum(axis=0))
                if i > 0:
                    d = (d @ model.W[i].T) * (model._pre[i - 1] > 0)
            grads_W.reverse()
            grads_b.reverse()
            for i in range(len(model.W)):
                # L2 weight decay (not on biases)
                grads_W[i] = grads_W[i] + l2 * model.W[i]
                for P, G, M, V in ((model.W, grads_W, mW, vW),
                                   (model.b, grads_b, mb, vb)):
                    M[i] = b1 * M[i] + (1 - b1) * G[i]
                    V[i] = b2 * V[i] + (1 - b2) * G[i] ** 2
                    mh = M[i] / (1 - b1 ** t)
                    vh = V[i] / (1 - b2 ** t)
                    P[i] -= lr * mh / (np.sqrt(vh) + eps)
        tr_loss, tr_acc = loss_acc(Xtr, ytr)
        va_loss, va_acc = loss_acc(Xva, yva)
        if va_acc > best_va:
            best_va = va_acc
            best = ([w.copy() for w in model.W], [b.copy() for b in model.b])
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f'ep {ep:3d}  train loss {tr_loss:.4f} acc {tr_acc:.3f} | '
                  f'val loss {va_loss:.4f} acc {va_acc:.3f}', flush=True)
    model.W, model.b = best
    return model

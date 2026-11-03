"""
A small Convolutional Neural Network written from scratch in NumPy.

Why NumPy?  The web app must run on any laptop for the demo / viva, without
installing TensorFlow.  Every layer below implements both the forward pass
(used for prediction) and the backward pass (used for training and Grad-CAM),
so the whole model can be explained line by line.

Architecture (input 48 x 48 x 3, H&E patch scaled to [0, 1]):
    Conv3x3(16) -> ReLU -> MaxPool2   -> 24 x 24 x 16
    Conv3x3(32) -> ReLU -> MaxPool2   -> 12 x 12 x 32
    Conv3x3(64) -> ReLU -> MaxPool2   ->  6 x  6 x 64
    Flatten -> Dense(64) -> ReLU -> Dropout(0.5)
    Dense(1) -> Sigmoid  = P(malignant)
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

IMG = 48


# ---------------------------------------------------------------- layers
class Conv2D:
    """3x3 'same' convolution implemented with im2col + one matrix multiply."""

    def __init__(self, cin, cout, rng):
        self.W = rng.normal(0, np.sqrt(2 / (9 * cin)), (3, 3, cin, cout)).astype(np.float32)  # He init
        self.b = np.zeros(cout, np.float32)

    def forward(self, x):                                    # x: (N,H,W,C)
        self.x_shape = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        cols = sliding_window_view(xp, (3, 3), axis=(1, 2))  # (N,H,W,C,3,3)
        N, H, W, C = x.shape
        self.cols = cols.transpose(0, 1, 2, 4, 5, 3).reshape(N * H * W, 9 * C)
        out = self.cols @ self.W.reshape(9 * C, -1) + self.b
        return out.reshape(N, H, W, -1)

    def backward(self, g):                                   # g: (N,H,W,Cout)
        N, H, W, C = self.x_shape
        g2 = g.reshape(-1, g.shape[-1])
        self.dW = (self.cols.T @ g2).reshape(self.W.shape)
        self.db = g2.sum(0)
        dcols = (g2 @ self.W.reshape(9 * C, -1).T).reshape(N, H, W, 3, 3, C)
        dxp = np.zeros((N, H + 2, W + 2, C), np.float32)
        for i in range(3):                                   # col2im
            for j in range(3):
                dxp[:, i:i + H, j:j + W] += dcols[:, :, :, i, j]
        return dxp[:, 1:-1, 1:-1]

    def params(self):
        return [("W", self.W, "dW"), ("b", self.b, "db")]


class ReLU:
    def forward(self, x):
        self.m = x > 0
        return x * self.m

    def backward(self, g):
        return g * self.m

    def params(self):
        return []


class MaxPool2:
    def forward(self, x):
        N, H, W, C = x.shape
        r = x.reshape(N, H // 2, 2, W // 2, 2, C)
        out = r.max(axis=(2, 4))
        self.mask = r == out[:, :, None, :, None]
        return out

    def backward(self, g):
        N, h, _, w, _, C = self.mask.shape
        return (self.mask * g[:, :, None, :, None]).reshape(N, h * 2, w * 2, C)

    def params(self):
        return []


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, g):
        return g.reshape(self.shape)

    def params(self):
        return []


class Dense:
    def __init__(self, nin, nout, rng):
        self.W = rng.normal(0, np.sqrt(2 / nin), (nin, nout)).astype(np.float32)
        self.b = np.zeros(nout, np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW = self.x.T @ g
        self.db = g.sum(0)
        return g @ self.W.T

    def params(self):
        return [("W", self.W, "dW"), ("b", self.b, "db")]


class Dropout:
    def __init__(self, p, rng):
        self.p, self.rng, self.training = p, rng, False

    def forward(self, x):
        if not self.training:
            return x
        self.m = (self.rng.random(x.shape) > self.p) / (1 - self.p)
        return x * self.m

    def backward(self, g):
        return g * self.m if self.training else g

    def params(self):
        return []


# ---------------------------------------------------------------- model
class BreastCNN:
    LAST_CONV = 6  # index of the last Conv2D layer (used by Grad-CAM)

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.layers = [
            Conv2D(3, 16, rng), ReLU(), MaxPool2(),
            Conv2D(16, 32, rng), ReLU(), MaxPool2(),
            Conv2D(32, 64, rng), ReLU(), MaxPool2(),
            Flatten(), Dense(6 * 6 * 64, 64, rng), ReLU(), Dropout(0.5, rng),
            Dense(64, 1, rng),
        ]

    def train_mode(self, on):
        for l in self.layers:
            if isinstance(l, Dropout):
                l.training = on

    def logits(self, x, keep=None):
        """Forward pass. If keep is an index, also return that layer's output."""
        kept = None
        for i, l in enumerate(self.layers):
            x = l.forward(x)
            if i == keep:
                kept = x
        return (x[:, 0], kept) if keep is not None else x[:, 0]

    def predict_proba(self, x, batch=256):
        self.train_mode(False)
        out = [1 / (1 + np.exp(-self.logits(x[i:i + batch]))) for i in range(0, len(x), batch)]
        return np.concatenate(out)

    def backward(self, g):
        g = g.reshape(-1, 1)
        for l in reversed(self.layers):
            g = l.backward(g)
        return g

    def backward_to(self, g, stop):
        """Back-propagate only down to (and excluding) layer `stop`."""
        for l in reversed(self.layers[stop + 1:]):
            g = l.backward(g)
        return g

    # --- save / load
    def state(self):
        return {f"{i}_{n}": a for i, l in enumerate(self.layers) for n, a, _ in l.params()}

    def save(self, path, **meta):
        np.savez_compressed(path, **self.state(), **{f"meta_{k}": np.array(v) for k, v in meta.items()})

    @classmethod
    def load(cls, path):
        m, d = cls(), np.load(path)
        for i, l in enumerate(m.layers):
            for n, a, _ in l.params():
                a[...] = d[f"{i}_{n}"]
        m.meta = {k[5:]: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files if k.startswith("meta_")}
        return m

    def n_params(self):
        return sum(a.size for l in self.layers for _, a, _ in l.params())


class Adam:
    def __init__(self, model, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.model, self.lr, self.b1, self.b2, self.eps, self.t = model, lr, b1, b2, eps, 0
        self.m, self.v = {}, {}

    def step(self):
        self.t += 1
        for i, l in enumerate(self.model.layers):
            for n, a, gname in l.params():
                g, k = getattr(l, gname), (i, n)
                self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g
                self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g * g
                mh = self.m[k] / (1 - self.b1 ** self.t)
                vh = self.v[k] / (1 - self.b2 ** self.t)
                a -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)


def bce_grad(logits, y):
    """Binary cross-entropy on sigmoid(logits). Returns (loss, dLoss/dlogits)."""
    p = 1 / (1 + np.exp(-logits))
    loss = -np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))
    return loss, ((p - y) / len(y)).astype(np.float32)


def gradcam(model, x1):
    """Grad-CAM for one image x1 (48,48,3). Returns (prob, heatmap 12x12 in [0,1])."""
    model.train_mode(False)
    logit, act = model.logits(x1[None], keep=BreastCNN.LAST_CONV + 1)  # ReLU after last conv
    g = model.backward_to(np.ones((1, 1), np.float32), BreastCNN.LAST_CONV + 1)
    w = g[0].mean(axis=(0, 1))                    # channel importance
    cam = np.maximum((act[0] * w).sum(-1), 0)
    cam = cam / (cam.max() + 1e-8)
    return float(1 / (1 + np.exp(-logit[0]))), cam

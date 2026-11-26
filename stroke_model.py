"""
Stroke-sequence letter recognizer — inference needs only NumPy.

The image recognizer (letter_model.py) draws your strokes into a 28x28 picture, which
throws away the order and direction you wrote them in. This model reads the path itself:

  stroke_features()   turns strokes into a fixed-length sequence of SEQ_LEN points,
                      evenly spaced along the path, each with position, direction and
                      whether the pen was down. The training script uses this SAME function.
  StrokeRecognizer    a small 1D convolutional network over that sequence, trained in
                      PyTorch by train_strokes.py and run here with plain NumPy.

Writing speed is deliberately ignored: points are spaced by distance, not time, so a slow
and a fast "S" look the same.
"""

import json

import numpy as np

from letter_model import LETTERS, render_strokes

SEQ_LEN = 64
N_FEATURES = 5           # x, y, cos(direction), sin(direction), pen down
KIND = "stroke_cnn"


# ---------------------------------------------------------------- features

def stroke_features(strokes, n=SEQ_LEN):
    """strokes: list of point sequences [(x, y), ...] in any units, in writing order.
    Returns a float32 (n, 5) array, or None if there is nothing to read.

    The strokes are joined into one path; the jumps between strokes (pen up) are part of
    the path but marked pen=0, so the model sees where the next stroke starts. The letter
    is centered and scaled to fit a unit box (aspect ratio kept), like render_strokes()."""
    strokes = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in strokes if len(s)]
    if not strokes:
        return None
    pts, pen = [], []          # pen[i] describes the segment from pts[i] to pts[i + 1]
    for i, s in enumerate(strokes):
        pts.append(s)
        pen.append(np.ones(len(s)))
        pen[-1][-1] = 0.0      # from the last point of a stroke to the next stroke: pen up
    pts = np.concatenate(pts)
    pen = np.concatenate(pen)[:-1]

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = float((hi - lo).max())
    if size < 1e-9:
        return None
    pts = (pts - (lo + hi) / 2.0) / size * 2.0          # longest side spans [-1, 1]

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = seg > 1e-9                                   # drop repeated points
    if not keep.any():
        return None
    a, b, pen, seg = pts[:-1][keep], pts[1:][keep], pen[keep], seg[keep]
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    # n points evenly spaced along the path, each taking the segment it falls on
    d = (np.arange(n) + 0.5) / n * cum[-1]
    k = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    frac = ((d - cum[k]) / seg[k])[:, None]
    xy = a[k] + (b[k] - a[k]) * frac
    direction = (b[k] - a[k]) / seg[k][:, None]
    return np.concatenate([xy, direction, pen[k][:, None]], axis=1).astype(np.float32)


# ---------------------------------------------------------------- network

def conv1d(x, W, b):
    """x: (C_in, L), W: (C_out, C_in, k) as stored by PyTorch, 'same' padding, stride 1."""
    c_out, c_in, k = W.shape
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    L = x.shape[1]
    windows = np.stack([xp[:, j:j + L] for j in range(k)], axis=-1)        # (C_in, L, k)
    cols = windows.transpose(1, 0, 2).reshape(L, c_in * k)
    return (cols @ W.reshape(c_out, c_in * k).T + b).T                    # (C_out, L)


class StrokeRecognizer:
    """Same interface as LetterRecognizer.predict(), so evaluate.py can compare them."""

    def __init__(self, weights_path):
        w = np.load(weights_path)
        if str(w["kind"]) != KIND:
            raise ValueError(f"{weights_path} is not a stroke model")
        self.meta = json.loads(str(w["meta"]))
        self.convs = []
        i = 0
        while f"conv{i}_W" in w.files:
            self.convs.append((w[f"conv{i}_W"], w[f"conv{i}_b"], bool(w[f"conv{i}_pool"])))
            i += 1
        self.fc = (w["fc0_W"], w["fc0_b"])
        self.out = (w["fc1_W"], w["fc1_b"])
        self.mean, self.std = w["mean"], w["std"]
        self._check = (w["check_input"], w["check_logits"])

    def logits(self, feats):
        """feats: (SEQ_LEN, N_FEATURES). Returns (logits, embedding)."""
        x = ((feats - self.mean) / self.std).T                            # (C, L)
        for W, b, pool in self.convs:
            x = np.maximum(conv1d(x, W, b), 0.0)
            if pool:
                x = x[:, : x.shape[1] // 2 * 2].reshape(x.shape[0], -1, 2).max(axis=2)
        x = np.concatenate([x.mean(axis=1), x.max(axis=1)])
        emb = np.maximum(self.fc[0] @ x + self.fc[1], 0.0)
        return self.out[0] @ emb + self.out[1], emb

    def self_check(self, atol=1e-3):
        """True if this NumPy code reproduces the outputs PyTorch saved at export time."""
        got = np.stack([self.logits(f)[0] for f in self._check[0]])
        return bool(np.allclose(got, self._check[1], atol=atol))

    def predict_features(self, feats, top=3):
        z, _ = self.logits(feats)
        e = np.exp(z - z.max())
        probs = e / e.sum()
        order = np.argsort(probs)[::-1][:top]
        return [(LETTERS[i], float(probs[i])) for i in order]

    def predict(self, strokes, top=3):
        """Returns (top-k [(letter, prob)], rendered image) or ([], None).
        The image is only for display (e.g. evaluate.py's error sheet)."""
        feats = stroke_features(strokes)
        if feats is None:
            return [], None
        return self.predict_features(feats, top), render_strokes(strokes)

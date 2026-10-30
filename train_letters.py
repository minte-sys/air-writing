#!/usr/bin/env python3
"""
Train the air-writing letter recognizer (A–Z) on synthetic data.

Air-written letters are just strokes, so instead of scanned pen handwriting we generate
training data from stroke templates of each capital letter (several writing variants each),
distorted heavily so the network learns the letter's shape rather than one exact drawing:
  - smooth random warping of the whole letter
  - rotation, shear and stretched aspect ratio
  - strokes shifted so they don't meet exactly, ends that overshoot or stop short
  - small "hooks" at stroke ends (pen-lift lag in air writing), tremor noise
  - varying stroke thickness
Every sample is rendered with letter_model.render_strokes(), the same function the app uses.

The model is a multi-layer perceptron (784 -> 256 -> 128 -> 26) trained with scikit-learn,
then exported to letter_mlp.npz so the app can run it with NumPy alone.

Usage:  pip install scikit-learn      (only needed for training)
        python train_letters.py       (about a few minutes on a laptop)
        python train_letters.py --preview   (saves a grid of training samples to preview.png)
"""

import argparse
import math
import time

import cv2
import numpy as np

from letter_model import IMG, LETTERS, render_strokes


# ---------------------------------------------------------------- templates
# Coordinates in a unit box: x to the right, y DOWN (screen coordinates).

def arc(cx, cy, rx, ry, a0, a1):
    """Points on an ellipse from angle a0 to a1 (degrees, screen coords: -90 = top)."""
    n = max(8, int(abs(a1 - a0) / 8))
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(a0, a1, n)]


O_SHAPE = arc(0.5, 0.5, 0.5, 0.5, -90, 270)
P_BOWL = [(0, 1), (0, 0), (0.4, 0)] + arc(0.4, 0.27, 0.5, 0.27, -90, 90) + [(0, 0.54)]

TEMPLATES = {
    "A": [[[(0, 1), (0.5, 0), (1, 1)], [(0.22, 0.6), (0.78, 0.6)]],
          [[(0, 1), (0.42, 0), (0.58, 0), (1, 1)], [(0.2, 0.55), (0.8, 0.55)]]],
    "B": [[[(0, 0), (0, 1)],
           [(0, 0), (0.45, 0)] + arc(0.45, 0.25, 0.4, 0.25, -90, 90) + [(0, 0.5)],
           [(0, 0.5), (0.5, 0.5)] + arc(0.5, 0.75, 0.45, 0.25, -90, 90) + [(0, 1)]],
          [[(0, 1), (0, 0)] + arc(0.0, 0.25, 0.75, 0.25, -90, 90)
           + arc(0.0, 0.75, 0.85, 0.25, -90, 90)]],
    "C": [[arc(0.55, 0.5, 0.5, 0.5, -50, -310)],
          [arc(0.5, 0.5, 0.5, 0.5, -30, -320)]],
    "D": [[[(0, 0), (0, 1)], [(0, 0), (0.35, 0)] + arc(0.35, 0.5, 0.6, 0.5, -90, 90) + [(0, 1)]],
          [[(0, 1), (0, 0)] + arc(0.0, 0.5, 0.95, 0.5, -90, 90)]],
    "E": [[[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, 0.5), (0.75, 0.5)]],
          [[(1, 0), (0, 0), (0, 0.5), (0.75, 0.5)], [(0, 0.5), (0, 1), (1, 1)]]],
    "F": [[[(1, 0), (0, 0), (0, 1)], [(0, 0.48), (0.75, 0.48)]]],
    "G": [[arc(0.5, 0.5, 0.5, 0.5, -45, -360) + [(0.55, 0.5)]],
          [arc(0.5, 0.5, 0.5, 0.5, -45, -360), [(1, 0.5), (1, 1)], [(0.6, 0.5), (1, 0.5)]]],
    "H": [[[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.5), (1, 0.5)]]],
    "I": [[[(0.5, 0), (0.5, 1)]],
          [[(0.2, 0), (0.8, 0)], [(0.5, 0), (0.5, 1)], [(0.2, 1), (0.8, 1)]]],
    "J": [[[(1, 0), (1, 0.7)] + arc(0.5, 0.7, 0.5, 0.3, 0, 180)],
          [[(0.5, 0), (1.4, 0)], [(1, 0), (1, 0.7)] + arc(0.5, 0.7, 0.5, 0.3, 0, 180)]],
    "K": [[[(0, 0), (0, 1)], [(0.9, 0), (0, 0.55), (0.95, 1)]],
          [[(0, 0), (0, 1)], [(0.9, 0), (0, 0.5)], [(0.3, 0.35), (0.95, 1)]]],
    "L": [[[(0, 0), (0, 1), (0.75, 1)]]],
    "M": [[[(0, 1), (0, 0), (0.5, 0.6), (1, 0), (1, 1)]],
          [[(0, 1), (0.15, 0), (0.5, 0.75), (0.85, 0), (1, 1)]]],
    "N": [[[(0, 1), (0, 0), (1, 1), (1, 0)]]],
    "O": [[O_SHAPE],
          [arc(0.5, 0.5, 0.5, 0.5, -100, -460)]],
    "P": [[P_BOWL]],
    "Q": [[O_SHAPE, [(0.6, 0.7), (1.05, 1.05)]]],
    "R": [[P_BOWL, [(0.3, 0.54), (0.95, 1)]],
          [P_BOWL + [(0.95, 1)]]],
    "S": [[arc(0.5, 0.26, 0.42, 0.26, -20, -270) + arc(0.5, 0.74, 0.45, 0.26, -90, 160)]],
    "T": [[[(0, 0), (1, 0)], [(0.5, 0), (0.5, 1)]]],
    "U": [[[(0, 0), (0, 0.6)] + arc(0.5, 0.6, 0.5, 0.4, 180, 0) + [(1, 0)]],
          [[(0, 0), (0, 0.6)] + arc(0.5, 0.6, 0.5, 0.4, 180, 0) + [(1, 0)], [(1, 0), (1, 1)]]],
    "V": [[[(0, 0), (0.5, 1), (1, 0)]]],
    "W": [[[(0, 0), (0.25, 1), (0.5, 0.35), (0.75, 1), (1, 0)]],
          [[(0, 0), (0.2, 1), (0.5, 0), (0.8, 1), (1, 0)]]],
    "X": [[[(0, 0), (1, 1)], [(1, 0), (0, 1)]]],
    "Y": [[[(0, 0), (0.5, 0.5)], [(1, 0), (0.5, 0.5), (0.5, 1)]],
          [[(0, 0), (0.5, 0.5), (1, 0)], [(0.5, 0.5), (0.5, 1)]]],
    "Z": [[[(0, 0), (1, 0), (0, 1), (1, 1)]],
          [[(0, 0), (1, 0), (0, 1), (1, 1)], [(0.25, 0.5), (0.75, 0.5)]]],
}
assert sorted(TEMPLATES) == list(LETTERS)


# ---------------------------------------------------------------- augmentation

def resample(stroke, step=0.02):
    s = np.asarray(stroke, dtype=np.float64)
    if len(s) < 2:
        return s
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
    d = np.concatenate([[0], np.cumsum(seg)])
    if d[-1] < 1e-9:
        return s[:1]
    t = np.linspace(0, d[-1], max(2, int(d[-1] / step) + 1))
    return np.stack([np.interp(t, d, s[:, 0]), np.interp(t, d, s[:, 1])], axis=1)


def adjust_ends(s, rng):
    """Overshoot or stop short at each end of a stroke."""
    for end in (0, -1):
        if len(s) < 3 or rng.random() > 0.5:
            continue
        nb = 1 if end == 0 else -2
        direction = s[end] - s[nb]
        n = np.linalg.norm(direction)
        if n < 1e-9:
            continue
        amount = rng.uniform(-0.06, 0.08)
        if amount > 0:
            extra = s[end] + direction / n * amount
            s = np.vstack([extra[None], s]) if end == 0 else np.vstack([s, extra[None]])
        else:
            k = max(1, int(-amount / 0.02))
            s = s[k:] if end == 0 else s[:-k]
            if len(s) < 2:
                break
    return s


def add_hook(s, rng):
    """A short tail at the start or end (pen-down / pen-up lag)."""
    angle = rng.uniform(0, 2 * math.pi)
    length = rng.uniform(0.03, 0.1)
    tail = np.linspace(0, length, 4)[1:, None] * np.array([math.cos(angle), math.sin(angle)])
    if rng.random() < 0.5:
        return np.vstack([(s[0] + tail)[::-1], s])
    return np.vstack([s, s[-1] + tail])


def augment(variant, rng):
    strokes = [resample(s) for s in variant]
    strokes = [s + rng.normal(0, 0.025, 2) for s in strokes]                  # strokes don't meet exactly
    strokes = [adjust_ends(s, rng) for s in strokes]
    strokes = [add_hook(s, rng) if rng.random() < 0.25 and len(s) > 2 else s for s in strokes]

    # smooth random warp of the whole letter
    k = rng.uniform(0.02, 0.06, size=4)
    f = rng.uniform(0.4, 1.5, size=4) * 2 * math.pi
    ph = rng.uniform(0, 2 * math.pi, size=4)

    def warp(p):
        x, y = p[:, 0], p[:, 1]
        dx = k[0] * np.sin(f[0] * y + ph[0]) + k[1] * np.sin(f[1] * x + ph[1])
        dy = k[2] * np.sin(f[2] * x + ph[2]) + k[3] * np.sin(f[3] * y + ph[3])
        return np.stack([x + dx, y + dy], axis=1)

    # affine: rotation, shear, aspect ratio
    a = math.radians(rng.uniform(-14, 14))
    sh = rng.uniform(-0.3, 0.3)
    sx, sy = rng.uniform(0.7, 1.25), rng.uniform(0.85, 1.15)
    A = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ \
        np.array([[1, sh], [0, 1]]) @ np.diag([sx, sy])

    out = []
    for s in strokes:
        s = warp(s) @ A.T
        s = s + rng.normal(0, 0.004, s.shape)                                  # tremor
        out.append(s)
    return out, rng.uniform(0.06, 0.13)


def make_dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * len(LETTERS), IMG * IMG), np.float32)
    y = np.zeros(len(X), np.int64)
    i = 0
    for ci, letter in enumerate(LETTERS):
        variants = TEMPLATES[letter]
        for _ in range(n_per_class):
            strokes, thick = augment(variants[rng.integers(len(variants))], rng)
            img = render_strokes(strokes, thickness=thick)
            X[i] = img.reshape(-1)
            y[i] = ci
            i += 1
    return X, y


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4000)
    ap.add_argument("--out", default="letter_mlp.npz")
    ap.add_argument("--preview", action="store_true", help="save preview.png of samples and exit")
    args = ap.parse_args()

    if args.preview:
        X, _ = make_dataset(8, seed=1)
        grid = X.reshape(len(LETTERS), 8, IMG, IMG).transpose(1, 2, 0, 3).reshape(8 * IMG, len(LETTERS) * IMG)
        cv2.imwrite("preview.png", (grid * 255).astype(np.uint8))
        print("saved preview.png (columns A..Z)")
        return

    from sklearn.neural_network import MLPClassifier

    t0 = time.time()
    X, y = make_dataset(args.per_class, seed=0)
    Xv, yv = make_dataset(300, seed=12345)
    print(f"generated {len(X)} training / {len(Xv)} validation images in {time.time() - t0:.0f}s")

    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-3
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256,
                        learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                        validation_fraction=0.08, n_iter_no_change=5, random_state=0, verbose=True)
    clf.fit((X - mean) / std, y)
    acc = (clf.predict((Xv - mean) / std) == yv).mean()
    print(f"validation accuracy on unseen synthetic letters: {acc:.3f}  ({time.time() - t0:.0f}s total)")

    arrays = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        arrays[f"W{i}"] = W.astype(np.float32)
        arrays[f"b{i}"] = b.astype(np.float32)
    np.savez_compressed(args.out, **arrays)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

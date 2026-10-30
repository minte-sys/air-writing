"""
Letter recognition for air writing — needs only NumPy + OpenCV.

  render_strokes()     turns strokes (lists of points) into a normalized 28x28 image.
                       The app and the training script use this SAME function, so the
                       model sees exactly the kind of image it was trained on.
  LetterRecognizer     a small neural network (784 -> 256 -> 128 -> 26) run with NumPy,
                       plus personalization: every letter you correct is stored, and the
                       recognizer blends the network's guess with your nearest saved examples.
"""

import json
import os

import cv2
import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMG = 28                 # model input size
RENDER = 112             # render at 4x, then downsample for smooth edges
PAD = 12                 # margin inside the render canvas
BASE_THICKNESS = 0.09    # stroke thickness as a fraction of the letter's size


# ---------------------------------------------------------------- rendering

def render_strokes(strokes, thickness=BASE_THICKNESS):
    """strokes: list of point sequences [(x, y), ...] in any units.
    Returns a float32 (28, 28) image in [0, 1], or None if there is nothing to draw.
    The letter is scaled to fit (aspect ratio kept) and centered, so size and
    position on screen don't matter."""
    pts = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in strokes if len(s)]
    if not pts:
        return None
    all_pts = np.concatenate(pts)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    size = max(float((hi - lo).max()), 1e-6)
    box = RENDER - 2 * PAD
    scale = box / size
    center = (lo + hi) / 2.0

    canvas = np.zeros((RENDER, RENDER), np.uint8)
    t = max(2, int(round(thickness * box)))
    for p in pts:
        q = (p - center) * scale + RENDER / 2.0
        q = np.round(q * 16).astype(np.int32)                 # 4 bits of sub-pixel precision
        if len(q) == 1:
            cv2.circle(canvas, (int(q[0, 0]), int(q[0, 1])), (t // 2) * 16, 255, -1, cv2.LINE_AA, 4)
        else:
            cv2.polylines(canvas, [q.reshape(-1, 1, 2)], False, 255, t, cv2.LINE_AA, 4)

    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
    small = cv2.resize(canvas, (IMG, IMG), interpolation=cv2.INTER_AREA).astype(np.float32)
    peak = small.max()
    return small / peak if peak > 0 else None


# ---------------------------------------------------------------- recognizer

class LetterRecognizer:
    SIM_MIN = 0.85       # cosine similarity needed for a saved example to count
    K = 5                # neighbours consulted among your saved examples
    PERSONAL_WEIGHT = 1.0

    def __init__(self, weights_path, samples_path=None):
        w = np.load(weights_path)
        self.layers = []
        i = 0
        while f"W{i}" in w.files:
            self.layers.append((w[f"W{i}"].astype(np.float32), w[f"b{i}"].astype(np.float32)))
            i += 1
        self.mean = w["mean"].astype(np.float32)
        self.std = w["std"].astype(np.float32)

        self.samples_path = samples_path
        self.sample_imgs, self.sample_labels = [], []
        self._emb = np.zeros((0, self.layers[-1][0].shape[0]), np.float32)
        if samples_path and os.path.exists(samples_path):
            try:
                with open(samples_path) as f:
                    data = json.load(f)
                for item in data:
                    self._add(np.array(item["img"], np.float32).reshape(IMG, IMG) / 255.0, item["label"])
            except (OSError, ValueError, KeyError):
                pass

    # --- network
    def _forward(self, img):
        x = (img.reshape(-1) - self.mean) / self.std
        for W, b in self.layers[:-1]:
            x = np.maximum(x @ W + b, 0.0)
        emb = x
        W, b = self.layers[-1]
        logits = emb @ W + b
        e = np.exp(logits - logits.max())
        return e / e.sum(), emb

    @staticmethod
    def _unit(v):
        return v / (np.linalg.norm(v) + 1e-8)

    # --- personalization
    def _add(self, img, label):
        _, emb = self._forward(img)
        self.sample_imgs.append(img)
        self.sample_labels.append(label)
        self._emb = np.vstack([self._emb, self._unit(emb)[None]])

    def teach(self, img, label):
        """Store a corrected example of your handwriting and save it to disk."""
        if img is None or label not in LETTERS:
            return
        self._add(img, label)
        if self.samples_path:
            data = [{"label": l, "img": (im * 255).round().astype(int).reshape(-1).tolist()}
                    for im, l in zip(self.sample_imgs, self.sample_labels)]
            with open(self.samples_path, "w") as f:
                json.dump(data, f)

    @property
    def n_personal(self):
        return len(self.sample_labels)

    # --- prediction
    def predict_image(self, img, top=3):
        probs, emb = self._forward(img)
        if len(self.sample_labels):
            sims = self._emb @ self._unit(emb)
            votes = np.zeros(len(LETTERS), np.float32)
            for j in np.argsort(sims)[::-1][: self.K]:
                if sims[j] > self.SIM_MIN:   # closer match -> stronger vote (0..1)
                    votes[LETTERS.index(self.sample_labels[j])] += (sims[j] - self.SIM_MIN) / (1 - self.SIM_MIN)
            if votes.sum() > 0:
                probs = probs + self.PERSONAL_WEIGHT * votes
                probs = probs / probs.sum()
        order = np.argsort(probs)[::-1][:top]
        return [(LETTERS[i], float(probs[i])) for i in order]

    def predict(self, strokes, top=3):
        """Returns (top-k [(letter, prob)], rendered image) or ([], None)."""
        img = render_strokes(strokes)
        if img is None:
            return [], None
        return self.predict_image(img, top), img

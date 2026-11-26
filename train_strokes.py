#!/usr/bin/env python3
"""
Train the stroke-sequence recognizer (stroke_model.py) on synthetic letters.

Uses the same letter templates and distortions as train_letters.py, plus distortions that
only matter when the model sees writing order and direction:
  - strokes written in a different order, or in the opposite direction
  - two strokes joined, when the app misses a pen-up (the debouncer holds the gesture)
  - one stroke split in two, when the app sees a pen-up that wasn't meant

Training uses PyTorch (pip install -r requirements-train.txt); the app and evaluate.py
run the exported weights with NumPy only.

    python train_strokes.py                    # a few minutes on a laptop CPU
    python evaluate.py --weights letter_mlp.npz stroke_cnn.npz    # compare on real letters
"""

import argparse
import json
import os
import time

import numpy as np

from letter_model import LETTERS, LetterRecognizer
from stroke_model import KIND, N_FEATURES, SEQ_LEN, StrokeRecognizer, stroke_features
from train_letters import TEMPLATES, augment

HERE = os.path.dirname(os.path.abspath(__file__))

P_SHUFFLE = 0.15     # strokes in a different order
P_REVERSE = 0.15     # each stroke, written the other way
P_MERGE = 0.10       # missed pen-up: two strokes become one
P_SPLIT = 0.10       # extra pen-up: a stroke is cut in two


# ---------------------------------------------------------------- data

def vary_writing_order(variant, rng):
    strokes = [list(s) for s in variant]
    if len(strokes) > 1 and rng.random() < P_SHUFFLE:
        strokes = [strokes[i] for i in rng.permutation(len(strokes))]
    strokes = [s[::-1] if rng.random() < P_REVERSE else s for s in strokes]
    if len(strokes) > 1 and rng.random() < P_MERGE:
        i = int(rng.integers(len(strokes) - 1))
        strokes[i:i + 2] = [strokes[i] + strokes[i + 1]]
    return strokes


def split_a_stroke(strokes, rng):
    if rng.random() >= P_SPLIT:
        return strokes
    i = int(rng.integers(len(strokes)))
    s = strokes[i]
    if len(s) < 12:
        return strokes
    cut = int(rng.integers(4, len(s) - 4))
    gap = int(rng.integers(0, 3))                        # a few points lost while the pen was "up"
    return strokes[:i] + [s[:cut], s[cut + gap:]] + strokes[i + 1:]


def synthetic_letter(letter, rng, vary=True):
    variants = TEMPLATES[letter]
    variant = variants[rng.integers(len(variants))]
    if vary:
        variant = vary_writing_order(variant, rng)
    strokes, _ = augment(variant, rng)
    return split_a_stroke(strokes, rng) if vary else strokes


def make_dataset(n_per_class, seed, vary=True, keep_strokes=False):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * len(LETTERS), SEQ_LEN, N_FEATURES), np.float32)
    y = np.zeros(len(X), np.int64)
    kept = []
    i = 0
    for ci, letter in enumerate(LETTERS):
        for _ in range(n_per_class):
            strokes = synthetic_letter(letter, rng, vary)
            X[i], y[i] = stroke_features(strokes), ci
            if keep_strokes:
                kept.append(strokes)
            i += 1
    return X, y, kept


# ---------------------------------------------------------------- model

CONVS = [(64, 5, False), (64, 5, True), (128, 5, True), (128, 3, False)]     # (channels, kernel, pool after)
EMBED = 128


def build_model():
    import torch
    import torch.nn as nn

    class StrokeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c_in = [], N_FEATURES
            for c_out, k, pool in CONVS:
                layers += [nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.ReLU()]
                if pool:
                    layers.append(nn.MaxPool1d(2))
                c_in = c_out
            self.convs = nn.Sequential(*layers)
            self.fc = nn.Linear(2 * c_in, EMBED)
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(EMBED, len(LETTERS))

        def forward(self, x):                      # x: (batch, SEQ_LEN, N_FEATURES)
            h = self.convs(x.transpose(1, 2))
            h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
            return self.out(self.drop(torch.relu(self.fc(h))))

    return StrokeCNN()


def export(model, path, mean, std, check_x, meta):
    import torch

    model.eval()
    with torch.no_grad():
        check_logits = model(torch.from_numpy((check_x - mean) / std)).numpy()
    arrays = {"kind": np.array(KIND), "meta": np.array(json.dumps(meta)),
              "mean": mean, "std": std, "check_input": check_x, "check_logits": check_logits}
    convs = [m for m in model.convs if isinstance(m, torch.nn.Conv1d)]
    for i, (conv, (_, _, pool)) in enumerate(zip(convs, CONVS)):
        arrays[f"conv{i}_W"] = conv.weight.detach().numpy()
        arrays[f"conv{i}_b"] = conv.bias.detach().numpy()
        arrays[f"conv{i}_pool"] = np.array(pool)
    for name, lin in (("fc0", model.fc), ("fc1", model.out)):
        arrays[f"{name}_W"] = lin.weight.detach().numpy()
        arrays[f"{name}_b"] = lin.bias.detach().numpy()
    np.savez_compressed(path, **{k: (v.astype(np.float32) if v.dtype == np.float64 else v)
                                 for k, v in arrays.items()})


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "stroke_cnn.npz"))
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 1)

    t0 = time.time()
    X, y, _ = make_dataset(args.per_class, seed=args.seed)
    Xv, yv, val_strokes = make_dataset(200, seed=12345, keep_strokes=True)
    print(f"generated {len(X)} training / {len(Xv)} validation sequences in {time.time() - t0:.0f}s")

    mean = X.reshape(-1, N_FEATURES).mean(axis=0)
    std = X.reshape(-1, N_FEATURES).std(axis=0) + 1e-3
    Xt, yt = torch.from_numpy((X - mean) / std), torch.from_numpy(y)
    Xvt = torch.from_numpy((Xv - mean) / std)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(X) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i + args.batch]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(Xvt).argmax(dim=1).numpy() == yv).mean()
        print(f"epoch {epoch:2d}  loss {total / len(Xt):.4f}  validation accuracy {acc:.4f}"
              f"  ({time.time() - t0:.0f}s)")

    meta = {"seq_len": SEQ_LEN, "per_class": args.per_class, "epochs": args.epochs,
            "val_acc_synthetic": float(acc), "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    export(model, args.out, mean.astype(np.float32), std.astype(np.float32), Xv[::len(Xv) // 8][:8], meta)
    rec = StrokeRecognizer(args.out)
    assert rec.self_check(), "NumPy inference does not match PyTorch: export is broken"
    print(f"saved {args.out} (NumPy inference matches PyTorch)")

    # Same synthetic validation letters through the image model, for reference only.
    # Real recordings (evaluate.py) are what decide which model is better.
    mlp_path = os.path.join(HERE, "letter_mlp.npz")
    if os.path.exists(mlp_path):
        mlp = LetterRecognizer(mlp_path)
        mlp_acc = np.mean([mlp.predict(s, top=1)[0][0][0] == LETTERS[c] for s, c in zip(val_strokes, yv)])
        np_acc = np.mean([rec.predict_features(f, top=1)[0][0] == LETTERS[c] for f, c in zip(Xv, yv)])
        print(f"on the same {len(yv)} synthetic validation letters: "
              f"stroke model {np_acc:.4f}, image model {mlp_acc:.4f}")


if __name__ == "__main__":
    main()

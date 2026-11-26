#!/usr/bin/env python3
"""
Measure how well letter models recognize REAL air-written letters.

Record samples first:   python air_writing.py --record --writer alex
Then evaluate:          python evaluate.py
Compare two models:     python evaluate.py --weights letter_mlp.npz new_model.npz

The report shows top-1 / top-3 accuracy with a 95% confidence interval, accuracy per
letter and per writer, and the letter pairs mixed up most often. With two or more models
it also runs a McNemar test on each pair, which says whether the difference between two
models on the same samples is bigger than chance.

Personalization (my_letters.json) is deliberately NOT used: the numbers describe the
model itself, as a new user would experience it.

Optional outputs:
  --confusion confusion.png   confusion matrix image (rows: true letter, columns: guess)
  --errors errors.png         every misrecognized letter, as the model saw it
  --json report.json          all numbers, for comparing runs or plotting
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

import cv2
import numpy as np

from letter_model import IMG, LETTERS, LetterRecognizer
from recording import DEFAULT_DIR, load_samples, stroke_points

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_PER_LETTER = 5        # warn when a letter has fewer samples than this


# ---------------------------------------------------------------- statistics

def wilson_interval(k, n, z=1.96):
    """95% confidence interval for a proportion k/n. Unlike the textbook p ± 1.96·SE,
    it stays inside [0, 1] and works for small n and for accuracies near 100%."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_p(b, c):
    """Exact two-sided McNemar test. b = samples only model A got right, c = only model B.
    Returns the chance of a split at least this uneven if both models were equally good."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- evaluation

def evaluate(recognizer, samples):
    """Run the recognizer on every sample. Returns one result dict per sample."""
    results = []
    for s in samples:
        preds, img = recognizer.predict(stroke_points(s), top=3)
        guesses = [p[0] for p in preds]
        results.append({
            "id": s.get("id"), "label": s["label"], "writer": s.get("writer", "?"),
            "pred": guesses[0] if guesses else None, "top3": guesses,
            "conf": preds[0][1] if preds else 0.0, "img": img,
        })
    return results


def summarize(results):
    n = len(results)
    top1 = sum(r["pred"] == r["label"] for r in results)
    top3 = sum(r["label"] in r["top3"] for r in results)

    per_letter = {}
    for letter in LETTERS:
        rs = [r for r in results if r["label"] == letter]
        if rs:
            per_letter[letter] = {"n": len(rs), "acc": sum(r["pred"] == letter for r in rs) / len(rs)}

    per_writer = {}
    for writer in sorted({r["writer"] for r in results}):
        rs = [r for r in results if r["writer"] == writer]
        per_writer[writer] = {"n": len(rs), "acc": sum(r["pred"] == r["label"] for r in rs) / len(rs)}

    confusion = np.zeros((len(LETTERS), len(LETTERS)), np.int64)
    for r in results:
        if r["pred"] is not None:
            confusion[LETTERS.index(r["label"]), LETTERS.index(r["pred"])] += 1

    pairs = Counter((r["label"], r["pred"]) for r in results if r["pred"] != r["label"])
    lo, hi = wilson_interval(top1, n)
    return {
        "n": n,
        "top1": top1 / n if n else 0.0,
        "top1_ci95": [lo, hi],
        "top3": top3 / n if n else 0.0,
        # every letter counts equally, so common letters in the data can't hide weak ones
        "balanced": float(np.mean([v["acc"] for v in per_letter.values()])) if per_letter else 0.0,
        "per_letter": per_letter,
        "per_writer": per_writer,
        "confused_pairs": [{"true": t, "pred": p, "count": c} for (t, p), c in pairs.most_common(10)],
        "confusion": confusion.tolist(),
    }


# ---------------------------------------------------------------- output

def pct(x):
    return f"{100 * x:5.1f}%"


def print_dataset(samples):
    counts = Counter(s["label"] for s in samples)
    writers = Counter(s.get("writer", "?") for s in samples)
    print(f"{len(samples)} samples from {len(writers)} writer(s): "
          + ", ".join(f"{w} ({c})" for w, c in sorted(writers.items())))
    missing = [l for l in LETTERS if counts[l] == 0]
    few = [f"{l}:{counts[l]}" for l in LETTERS if 0 < counts[l] < MIN_PER_LETTER]
    if missing:
        print(f"  no samples for: {' '.join(missing)}")
    if few:
        print(f"  fewer than {MIN_PER_LETTER} samples (noisy): {' '.join(few)}")


def print_report(name, rep):
    lo, hi = rep["top1_ci95"]
    print(f"\n=== {name}")
    print(f"top-1 accuracy   {pct(rep['top1'])}   (95% CI {pct(lo).strip()} - {pct(hi).strip()})")
    print(f"top-3 accuracy   {pct(rep['top3'])}")
    print(f"balanced (mean per letter)  {pct(rep['balanced'])}")

    if len(rep["per_writer"]) > 1:
        print("per writer:  " + "   ".join(f"{w} {pct(v['acc']).strip()} (n={v['n']})"
                                          for w, v in rep["per_writer"].items()))

    worst = sorted(rep["per_letter"].items(), key=lambda kv: (kv[1]["acc"], -kv[1]["n"]))
    print("per letter (worst first):")
    line = []
    for letter, v in worst:
        line.append(f"{letter} {100 * v['acc']:3.0f}% ({v['n']:>2})")
        if len(line) == 6:
            print("  " + "   ".join(line))
            line = []
    if line:
        print("  " + "   ".join(line))

    if rep["confused_pairs"]:
        print("most confused (true -> guess):  " + ",  ".join(
            f"{p['true']}->{p['pred']} x{p['count']}" for p in rep["confused_pairs"]))


def print_comparison(names, all_results):
    print("\n=== model comparison (McNemar test on the same samples)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = all_results[i], all_results[j]
            only_a = sum(ra["pred"] == ra["label"] and rb["pred"] != rb["label"] for ra, rb in zip(a, b))
            only_b = sum(rb["pred"] == rb["label"] and ra["pred"] != ra["label"] for ra, rb in zip(a, b))
            p = mcnemar_p(only_a, only_b)
            verdict = "real difference" if p < 0.05 else "could be chance"
            print(f"{names[i]} vs {names[j]}:  only first right {only_a},  only second right {only_b},"
                  f"  p = {p:.3g}  ({verdict})")


def save_confusion(path, rep):
    cm = np.array(rep["confusion"], np.float64)
    rows = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    cell, margin = 28, 36
    size = margin + cell * len(LETTERS)
    img = np.full((size, size, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, letter in enumerate(LETTERS):
        pos = margin + i * cell
        cv2.putText(img, letter, (pos + 8, 24), font, 0.5, (60, 60, 60), 1, cv2.LINE_AA)     # guess
        cv2.putText(img, letter, (10, pos + 19), font, 0.5, (60, 60, 60), 1, cv2.LINE_AA)    # true
    for i in range(len(LETTERS)):
        for j in range(len(LETTERS)):
            if cm[i, j] == 0:
                continue
            x, y = margin + j * cell, margin + i * cell
            shade = int(255 * (1 - frac[i, j]))
            color = (255, shade, shade) if i == j else (shade, shade, 255)     # blue right, red wrong
            cv2.rectangle(img, (x + 1, y + 1), (x + cell - 1, y + cell - 1), color, -1)
            label = str(int(cm[i, j]))
            ink = (255, 255, 255) if frac[i, j] > 0.55 else (40, 40, 40)
            tw = cv2.getTextSize(label, font, 0.38, 1)[0][0]
            cv2.putText(img, label, (x + (cell - tw) // 2, y + 18), font, 0.38, ink, 1, cv2.LINE_AA)
    for k in range(len(LETTERS) + 1):
        p = margin + k * cell
        cv2.line(img, (margin, p), (size, p), (225, 225, 225), 1)
        cv2.line(img, (p, margin), (p, size), (225, 225, 225), 1)
    cv2.imwrite(path, img)


def save_errors(path, results, cols=12, scale=3):
    errors = [r for r in results if r["pred"] != r["label"] and r["img"] is not None]
    if not errors:
        return 0
    tile, caption = IMG * scale, 18
    cols = min(cols, len(errors))
    rows = math.ceil(len(errors) / cols)
    sheet = np.full((rows * (tile + caption), cols * tile, 3), 30, np.uint8)
    for k, r in enumerate(errors):
        y, x = (k // cols) * (tile + caption), (k % cols) * tile
        letter = cv2.resize((r["img"] * 255).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST)
        sheet[y:y + tile, x:x + tile] = letter[..., None]
        cv2.rectangle(sheet, (x, y), (x + tile - 1, y + tile - 1), (70, 70, 70), 1)
        cv2.putText(sheet, f"{r['label']}->{r['pred']} {100 * r['conf']:.0f}%", (x + 4, y + tile + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)
    return len(errors)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate letter models on recorded real air-writing.")
    ap.add_argument("data", nargs="*", default=[DEFAULT_DIR], help="recording files or folders")
    ap.add_argument("--weights", nargs="+", default=[os.path.join(HERE, "letter_mlp.npz")],
                    help="one or more model files to evaluate and compare")
    ap.add_argument("--writer", action="append", help="only use samples from this writer (repeatable)")
    ap.add_argument("--confusion", help="save the first model's confusion matrix as an image")
    ap.add_argument("--errors", help="save an image of the first model's mistakes")
    ap.add_argument("--json", help="save the full report as JSON")
    args = ap.parse_args(argv)

    try:
        samples, problems = load_samples(args.data)
    except FileNotFoundError as e:
        sys.exit(f"Not found: {e}\nRecord some letters first:  python air_writing.py --record --writer NAME")
    for p in problems:
        print(f"skipped {p}", file=sys.stderr)
    if args.writer:
        wanted = set(args.writer)
        samples = [s for s in samples if s.get("writer") in wanted]
    if not samples:
        sys.exit("No samples to evaluate.\nRecord some letters first:  python air_writing.py --record --writer NAME")

    print_dataset(samples)
    names = [os.path.basename(w) for w in args.weights]
    if len(set(names)) < len(names):
        names = list(args.weights)
    all_results, reports = [], {}
    for name, weights in zip(names, args.weights):
        results = evaluate(LetterRecognizer(weights), samples)
        all_results.append(results)
        reports[name] = summarize(results)
        print_report(name, reports[name])
    if len(names) > 1:
        print_comparison(names, all_results)

    if args.confusion:
        save_confusion(args.confusion, reports[names[0]])
        print(f"\nsaved confusion matrix to {args.confusion}")
    if args.errors:
        n = save_errors(args.errors, all_results[0])
        print(f"saved {n} mistakes to {args.errors}" if n else "no mistakes to save")
    if args.json:
        out = {"samples": len(samples), "data": args.data, "models": reports,
               "predictions": {name: [{k: r[k] for k in ("id", "label", "writer", "pred", "top3", "conf")}
                                      for r in res] for name, res in zip(names, all_results)}}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"saved report to {args.json}")
    return reports


if __name__ == "__main__":
    main()

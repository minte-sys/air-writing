# Air Writing ✍️

Paint and write text in the air with your index finger, using only a webcam.

Hand tracking runs on MediaPipe, and a small neural network recognizes the capital letters you write. It also learns your handwriting from your corrections.

<!-- Add a demo GIF here once you record one:  ![demo](demo.gif) -->

## Features

- **Air painting** with a neon glow effect and 4 colors
- **Gesture controls**, no keyboard needed:
  - ☝️ index finger: draw / erase
  - ✌️ index + middle finger: pointer for the on-screen toolbar
  - ✊ fist held for 1 second: clear everything
- **Text mode**: write a capital letter, pause, and it appears as typed text
- **Personalized recognition**: corrections are saved and used to recognize your style
- **Smooth tracking** with a One Euro filter and debounced gestures
- Export your drawing as PNG and the strokes as JSON

## How it works

| Stage | What happens |
|---|---|
| Capture | OpenCV reads webcam frames (~30 fps), mirrored horizontally |
| Hand detection | MediaPipe Hands finds the palm, then returns 21 landmarks per frame and tracks them between frames |
| Gesture recognition | A finger counts as extended when its tip is farther from the wrist than its middle joint. This test still works when the hand is rotated |
| Debouncing | A gesture must hold for 3 frames before the pen changes state, so one bad frame doesn't break a stroke |
| Smoothing | A One Euro filter removes jitter when moving slowly without adding lag when moving fast |
| Strokes | Fingertip points are collected into strokes; lines are never drawn between strokes |
| Rendering | New segments are drawn once on a persistent canvas, then blended over the video with a glow |
| Letter recognition | Strokes are scaled, centered and rendered to a 28×28 image, then classified by an MLP (784 → 256 → 128 → 26) running in NumPy |
| Personalization | Corrected letters are stored, and similar future letters are matched against them using the network's internal features |

### Training data

There is no large dataset of air-written letters, so the recognizer is trained on **synthetic data**. Stroke templates for A–Z (several writing styles each) are distorted with random warping, rotation, shear, stroke gaps, overshoots, pen-lift hooks and tremor. This produces 104,000 training images, rendered the same way the app renders your writing. Accuracy on unseen synthetic letters is 99.8%. Real air-writing scores lower, which is why the app learns from corrections.

## Installation

Requires Python 3.9+ and a webcam.

```bash
git clone https://github.com/minte-sys/air-writing.git
cd air-writing
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Usage

```bash
python air_writing.py
```

The first run downloads MediaPipe's hand landmark model (~8 MB).

Options: `--camera 1` (another webcam), `--width 1280 --height 720`, `--letter-pause 1.5` (longer pause before a letter is recognized), `--debounce 3`.

### Controls

| Key | Action |
|---|---|
| `m` | Switch paint / text mode |
| `1` `2` `3` | Replace the last letter with that guess |
| `Shift` + letter | Replace the last letter with the one you type |
| `Space` / `Backspace` | Add a space / delete (text mode) |
| `z` | Undo last stroke |
| `c` | Clear |
| `g` | Toggle glow |
| `s` | Toggle smoothing |
| `e` | Export PNG + JSON |
| `q` / `Esc` | Quit |

## Measuring accuracy on real air-writing

The 99.8% above is measured on synthetic letters. To find out how the recognizer does on real handwriting, record some labeled letters and evaluate on them.

**1. Record.** The app asks for one letter at a time, in shuffled order. Write it and pause, the same as in text mode.

```bash
python air_writing.py --record --writer alex              # 3 rounds of A-Z = 78 letters
python air_writing.py --record --writer alex --rounds 1   # a quicker session
```

| Key | Action |
|---|---|
| `Backspace` | Discard the unfinished letter, or delete the last saved one and write it again |
| `n` | Skip this letter for now (it is asked again at the end) |
| `z` | Undo the last stroke of the current letter |

Each letter is appended to `data/recordings/<writer>/<session>.jsonl` with its strokes as the app saw them and a timestamp for every point. The timing isn't used by the current model, but it lets a future stroke-sequence model train and test on the same recordings. Use a nickname for `--writer`: it ends up in folder names and reports.

**2. Evaluate.**

```bash
python evaluate.py                                       # all recordings in data/recordings
python evaluate.py --confusion cm.png --errors err.png   # also save a confusion matrix and the mistakes
python evaluate.py --weights letter_mlp.npz new.npz      # compare two models on the same letters
```

The report gives top-1 and top-3 accuracy with a 95% confidence interval, accuracy per letter and per writer, and the most confused pairs. When you compare models, a McNemar test says whether the difference is bigger than chance. Your personal corrections (`my_letters.json`) are not used, so the numbers describe what a new user gets.

A useful test set has letters from several people (at least 5 samples of every letter). Don't train on the same recordings you evaluate on.

## Retraining the letter model

```bash
pip install -r requirements-train.txt
python train_letters.py              # takes a few minutes, writes letter_mlp.npz
python train_letters.py --preview    # saves preview.png with sample training letters
```

## Project structure

```
air_writing.py          main app: tracking, gestures, drawing, toolbar, text mode
letter_model.py         stroke rendering + NumPy letter recognizer with personalization
letter_mlp.npz          trained network weights
train_letters.py        synthetic dataset generation and training
stroke_model.py         stroke-sequence features + NumPy 1D CNN recognizer
stroke_cnn.npz          trained stroke-sequence weights
train_strokes.py        stroke-sequence training (PyTorch) and export
recording.py            recording mode: letter prompts, saving and loading samples
evaluate.py             accuracy report on recorded letters, model comparison
tests/                  pytest tests (no webcam needed)
requirements.txt        runtime dependencies
requirements-train.txt  extra dependencies for retraining
requirements-dev.txt    everything plus pytest
```

## Stroke-sequence model (experimental)

The image recognizer draws your strokes into a picture, so it can't tell which way, or in what order, you wrote them. `stroke_model.py` reads the path itself. Your strokes are joined into one path, with the jumps between them marked as pen-up. Then 64 points are spaced evenly along it, each with its position, direction and whether the pen was down. A small 1D convolutional network classifies that sequence. Training runs in PyTorch; the app and `evaluate.py` run it in NumPy, so the app doesn't need PyTorch.

Training uses the same synthetic letters as the image model, plus air-writing mistakes that only matter when order is visible: strokes in another order or direction, a missed pen-up that joins two strokes, and an extra pen-up that splits one.

```bash
pip install -r requirements-train.txt
python train_strokes.py                                   # writes stroke_cnn.npz
python evaluate.py --weights letter_mlp.npz stroke_cnn.npz
```

The last command compares the two models on your recordings. The app still uses the image model; the stroke model replaces it only once it wins on real letters.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Future work

- Lowercase letters and digits
- Word-level correction using a dictionary
- Sending recognized text to other apps as keystrokes
- Training on real recorded air-writing samples

"""
Recording real air-written letters, and loading them back.

A recording session prompts for letters in a shuffled order (every letter once per
round) and appends each sample as one JSON line to

    data/recordings/<writer>/<session>.jsonl

JSON Lines keeps a crashed session's samples and makes undo a simple truncate.
Each line holds the strokes exactly as the app saw them (after smoothing), with a
timestamp per point. Keeping the timing lets a future stroke-sequence model use
the same recordings, not just the 28x28 image recognizer.

Sample format (schema 1):
    {"schema": 1, "id": "...", "label": "K", "writer": "alex",
     "session": "20261126-153000", "recorded_at": "2026-11-26T15:30:12",
     "frame": [1280, 720], "smoothing": true, "letter_pause": 1.0,
     "strokes": [{"pts": [[x, y], ...], "t": [seconds since letter start, ...]}, ...]}
"""

import glob
import json
import os
import random
import re
import time
import uuid

from letter_model import LETTERS

SCHEMA = 1
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "recordings")


def safe_name(name):
    """Writer names become folder names: keep them to letters, digits, - and _."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not cleaned:
        raise ValueError("writer name must contain at least one letter or digit")
    return cleaned


class Recorder:
    """Prompts for letters and saves labeled samples to a JSONL file."""

    def __init__(self, writer, out_dir=DEFAULT_DIR, rounds=3, letters=LETTERS, seed=None):
        self.writer = safe_name(writer)
        self.session = time.strftime("%Y%m%d-%H%M%S")
        folder = os.path.join(out_dir, self.writer)
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, f"{self.session}.jsonl")

        rng = random.Random(seed)
        self.queue = []
        for _ in range(rounds):
            r = list(letters)
            rng.shuffle(r)
            self.queue += r
        self.total = len(self.queue)
        self._saved = []          # (label, file offset before the line) for undo

    @property
    def target(self):
        """The letter to write next, or None when the session is finished."""
        return self.queue[0] if self.queue else None

    @property
    def n_saved(self):
        return len(self._saved)

    @property
    def done(self):
        return not self.queue

    def save(self, strokes, frame_size, smoothing, letter_pause):
        """strokes: [{"pts": [(x, y), ...], "t": [abs seconds, ...]}, ...]. Saves them
        labeled with the current target and moves to the next prompt."""
        if self.done:
            return None
        t0 = min(s["t"][0] for s in strokes if s.get("t"))
        sample = {
            "schema": SCHEMA,
            "id": uuid.uuid4().hex,
            "label": self.target,
            "writer": self.writer,
            "session": self.session,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame": list(frame_size),
            "smoothing": bool(smoothing),
            "letter_pause": letter_pause,
            "strokes": [{"pts": [[int(x), int(y)] for x, y in s["pts"]],
                         "t": [round(t - t0, 4) for t in s["t"]]} for s in strokes],
        }
        with open(self.path, "a", encoding="utf-8") as f:
            offset = f.tell()
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")
        self._saved.append((sample["label"], offset))
        self.queue.pop(0)
        return sample

    def undo(self):
        """Delete the last saved sample from the file and prompt for that letter again."""
        if not self._saved:
            return None
        label, offset = self._saved.pop()
        with open(self.path, "r+", encoding="utf-8") as f:
            f.truncate(offset)
        self.queue.insert(0, label)
        return label

    def skip(self):
        """Move the current prompt to the end of the queue."""
        if len(self.queue) > 1:
            self.queue.append(self.queue.pop(0))


# ---------------------------------------------------------------- loading

def find_files(paths):
    """Expand files and folders (searched recursively) into a sorted list of .jsonl files."""
    files = set()
    for p in paths:
        if os.path.isdir(p):
            files.update(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        elif os.path.isfile(p):
            files.add(p)
        else:
            raise FileNotFoundError(p)
    return sorted(files)


def load_samples(paths):
    """Read samples from .jsonl files or folders. Returns (samples, problems), where
    problems lists lines that were skipped and why."""
    samples, problems = [], []
    for path in find_files(paths):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                where = f"{path}:{n}"
                try:
                    s = json.loads(line)
                except ValueError:
                    problems.append(f"{where}: not valid JSON")
                    continue
                if s.get("schema") != SCHEMA:
                    problems.append(f"{where}: unknown schema {s.get('schema')!r}")
                elif s.get("label") not in LETTERS:
                    problems.append(f"{where}: bad label {s.get('label')!r}")
                elif not any(st.get("pts") for st in s.get("strokes", [])):
                    problems.append(f"{where}: no points")
                else:
                    s["_source"] = where
                    samples.append(s)
    return samples, problems


def stroke_points(sample):
    """The sample's strokes as plain point lists, ready for LetterRecognizer.predict()."""
    return [st["pts"] for st in sample["strokes"] if st.get("pts")]

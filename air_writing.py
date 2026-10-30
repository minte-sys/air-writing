#!/usr/bin/env python3
"""
Air Writing v2 — paint and write text in the air with your index finger.

Gestures
  index finger only          draw (or erase, if the eraser is selected)
  index + middle finger      pointer: hover a toolbar button to press it (no drawing)
  fist, held for 1 second    clear everything
  open hand / anything else  pen up

Toolbar (top of the screen, press by hovering with two fingers)
  4 colors · eraser · ABC (switch between paint mode and text mode) · space · delete · clear

Text mode
  Write one CAPITAL letter, then pause ~1 second: it is recognized and added to the text bar.
  The recognizer learns your handwriting from corrections:
    1 / 2 / 3          replace the last letter with the 1st/2nd/3rd guess shown
    Shift + letter     replace the last letter with that letter
  Corrections are saved to my_letters.json and used from then on.

Keyboard
  m mode   g glow   z undo   c clear   s smoothing   e export   space   backspace   q / Esc quit

Files needed in the same folder: letter_model.py, letter_mlp.npz
Run:  python air_writing.py        (options: --camera 1 --width 1280 --height 720 --letter-pause 1.0)
"""

import argparse
import json
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from letter_model import LetterRecognizer

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- configuration

PALETTE = [("Cyan", (255, 220, 0)), ("Pink", (200, 80, 255)),
           ("Green", (90, 255, 90)), ("Gold", (0, 210, 255))]
TEXT_COLOR = (255, 255, 255)
THICKNESS = 8
ERASER_RADIUS = 35
MIN_STEP_PX = 2.0
EXTEND_RATIO = 1.1
TOOLBAR_H = 80
HOVER_SECONDS = 0.35        # how long to hover a button to press it
FIST_SECONDS = 1.0          # how long to hold a fist to clear
MIN_LETTER_PX = 40          # smaller scribbles are ignored in text mode

WRIST, INDEX_TIP, PALM = 0, 8, 9
FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
HAND_MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
LETTER_WEIGHTS = os.path.join(HERE, "letter_mlp.npz")
MY_LETTERS = os.path.join(HERE, "my_letters.json")


# ---------------------------------------------------------------- hand tracking

class HandTracker:
    """MediaPipe Hands: classic `mp.solutions` API if available, otherwise the Tasks API."""

    def __init__(self, det_conf=0.6, track_conf=0.5):
        self._legacy = self._tasks = None
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            self._legacy = solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=1,
                min_detection_confidence=det_conf, min_tracking_confidence=track_conf)
            return

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(HAND_MODEL_PATH):
            print("Downloading hand landmark model (one-time)...")
            try:
                urllib.request.urlretrieve(MODEL_URL, HAND_MODEL_PATH)
            except OSError as e:
                if os.path.exists(HAND_MODEL_PATH):
                    os.remove(HAND_MODEL_PATH)
                raise SystemExit(f"Could not download the model ({e}).\nDownload it manually from:\n"
                                 f"  {MODEL_URL}\nand save it next to this script as:\n  {HAND_MODEL_PATH}")
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=det_conf, min_hand_presence_confidence=det_conf,
            min_tracking_confidence=track_conf)
        self._tasks = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb):
        """The 21 landmarks as normalized (x, y), or None if no hand."""
        if self._legacy is not None:
            result = self._legacy.process(rgb)
            if not result.multi_hand_landmarks:
                return None
            landmarks = result.multi_hand_landmarks[0].landmark
        else:
            ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
            self._last_ts = ts
            result = self._tasks.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            if not result.hand_landmarks:
                return None
            landmarks = result.hand_landmarks[0]
        return [(lm.x, lm.y) for lm in landmarks]

    def close(self):
        for t in (self._legacy, self._tasks):
            if t is not None:
                t.close()


# ---------------------------------------------------------------- gestures

def finger_extended(pts, tip, pip):
    """Rotation-invariant: extended if the tip is clearly farther from the wrist than the PIP joint."""
    wx, wy = pts[WRIST]
    return math.hypot(pts[tip][0] - wx, pts[tip][1] - wy) > \
        math.hypot(pts[pip][0] - wx, pts[pip][1] - wy) * EXTEND_RATIO


def classify_gesture(pts):
    ext = {name: finger_extended(pts, *j) for name, j in FINGERS.items()}
    i, m, r, p = ext["index"], ext["middle"], ext["ring"], ext["pinky"]
    if i and not (m or r or p):
        return "write"
    if i and m and not (r or p):
        return "select"
    if not (i or m or r or p):
        return "fist"
    return "idle"


class GestureDebouncer:
    """Switches to a new gesture only after it is seen for `frames` frames in a row."""

    def __init__(self, frames=3):
        self.frames, self.state = frames, "idle"
        self._candidate, self._count = None, 0

    def update(self, raw):
        if raw == self.state:
            self._candidate, self._count = None, 0
        elif raw == self._candidate:
            self._count += 1
            if self._count >= self.frames:
                self.state, self._candidate, self._count = raw, None, 0
        else:
            self._candidate, self._count = raw, 1
        return self.state


# ---------------------------------------------------------------- smoothing

class OneEuroFilter:
    """One Euro filter (Casiez et al., 2012): smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._x, self._dx, self._t = None, 0.0, None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff) / dt)

    def __call__(self, x, t):
        if self._x is None:
            self._x, self._t = x, t
            return x
        dt = max(t - self._t, 1e-6)
        self._t = t
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * ((x - self._x) / dt) + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x = a * x + (1 - a) * self._x
        return self._x


# ---------------------------------------------------------------- strokes & canvas

def draw_stroke(canvas, s):
    pts = s["pts"]
    if len(pts) == 1:
        cv2.circle(canvas, pts[0], s["thick"] // 2 + 1, s["color"], -1, cv2.LINE_AA)
    elif len(pts) > 1:
        cv2.polylines(canvas, [np.array(pts, np.int32).reshape(-1, 1, 2)], False,
                      s["color"], s["thick"], cv2.LINE_AA)


def redraw(canvas, strokes):
    canvas[:] = 0
    for s in strokes:
        draw_stroke(canvas, s)


def erase_at(strokes, p, radius):
    """Remove points within `radius` of p, splitting strokes where they are cut."""
    out, changed = [], False
    for s in strokes:
        pts = np.array(s["pts"], np.float32)
        keep = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) > radius
        if keep.all():
            out.append(s)
            continue
        changed = True
        run = []
        for pt, k in zip(s["pts"], keep):
            if k:
                run.append(pt)
            elif run:
                out.append({**s, "pts": run})
                run = []
        if run:
            out.append({**s, "pts": run})
    return out, changed


CORE_KERNEL = np.ones((5, 5), np.uint8)


def composite(frame, canvas, glow):
    """Neon look: blurred glow underneath, sharp strokes on top, whitish core.
    Uses OpenCV's masked copy (fast) rather than NumPy boolean indexing (slow at 1280x720)."""
    _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    if not cv2.countNonZero(mask):
        return
    if glow:
        h, w = canvas.shape[:2]
        small = cv2.resize(canvas, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(frame, 1.0, halo, 1.6, 0, dst=frame)          # saturating add
    cv2.copyTo(canvas, mask, frame)
    if glow:
        core = cv2.addWeighted(canvas, 0.45, canvas, 0, 140)           # mix toward white
        cv2.copyTo(core, cv2.erode(mask, CORE_KERNEL), frame)


# ---------------------------------------------------------------- drawing helpers

OUTLINE = [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    # Outline drawn as shifted copies at the SAME thickness: in OpenCV 5 a thicker line also
    # widens the glyphs, so a thick black copy would not line up with the text.
    x, y = org
    for dx, dy in OUTLINE:
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def text_width(text, scale, thick):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]


def make_buttons(w):
    spec = [(f"color:{i}", name, col) for i, (name, col) in enumerate(PALETTE)]
    spec += [("eraser", "Eraser", (200, 200, 200)), ("mode", "ABC", (255, 255, 255)),
             ("space", "Space", (255, 255, 255)), ("delete", "Del", (255, 255, 255)),
             ("clear", "Clear", (80, 80, 255))]
    bw = w / len(spec)
    return [{"key": k, "label": lab, "color": c, "x0": int(i * bw), "x1": int((i + 1) * bw)}
            for i, (k, lab, c) in enumerate(spec)]


def button_at(buttons, p):
    if p is None or p[1] >= TOOLBAR_H:
        return None
    for b in buttons:
        if b["x0"] <= p[0] < b["x1"]:
            return b
    return None


# ---------------------------------------------------------------- app

class App:
    def __init__(self, w, h, letter_pause):
        self.w, self.h = w, h
        self.letter_pause = letter_pause
        self.canvas = np.zeros((h, w, 3), np.uint8)
        self.strokes, self.current = [], None
        self.color_idx, self.tool, self.mode = 0, "pen", "paint"
        self.glow, self.smoothing = True, True
        self.text = ""
        self.last_pred = None              # {"img", "preds"} of the last recognized letter
        self.pops = []                     # recognized-letter animations
        self.last_activity = 0.0
        self.now = time.monotonic()        # time of the current frame
        self.message, self.message_until = "", 0.0
        self.buttons = make_buttons(w)
        self.hover_btn, self.hover_since, self.hover_armed = None, 0.0, True
        self.fist_since, self.fist_done = None, False

        self.recognizer = None
        if os.path.exists(LETTER_WEIGHTS):
            self.recognizer = LetterRecognizer(LETTER_WEIGHTS, MY_LETTERS)
        else:
            self.say(f"letter_mlp.npz not found: text mode disabled", 5)

    # --- small helpers
    def say(self, msg, secs=2.5):
        self.message, self.message_until = msg, time.monotonic() + secs

    @property
    def color(self):
        return TEXT_COLOR if self.mode == "text" else PALETTE[self.color_idx][1]

    def pending_letter(self):
        return [s for s in self.strokes if s.get("letter")]

    def finish_stroke(self):
        if self.current is not None:
            if len(self.current["pts"]) == 1:
                draw_stroke(self.canvas, self.current)
            self.strokes.append(self.current)
            self.current = None
            self.last_activity = self.now

    def remove_pending(self):
        self.strokes = [s for s in self.strokes if not s.get("letter")]
        redraw(self.canvas, self.strokes)

    # --- text mode
    def recognize_pending(self, now):
        pending = self.pending_letter()
        if not pending or self.recognizer is None:
            return
        pts = np.concatenate([np.array(s["pts"], np.float32) for s in pending])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        self.remove_pending()
        if max(x1 - x0, y1 - y0) < MIN_LETTER_PX:
            return                                  # too small: probably an accidental touch
        preds, img = self.recognizer.predict([s["pts"] for s in pending])
        if not preds:
            return
        self.text += preds[0][0]
        self.last_pred = {"img": img, "preds": preds}
        self.pops.append((preds[0][0], (int((x0 + x1) / 2), int((y0 + y1) / 2)), int(y1 - y0), now))

    def correct_last(self, letter):
        if not self.text or self.last_pred is None or not self.text[-1].isalpha():
            return
        self.text = self.text[:-1] + letter
        self.recognizer.teach(self.last_pred["img"], letter)
        self.last_pred = None
        self.say(f"Learned your '{letter}'  ({self.recognizer.n_personal} saved examples)")

    def add_space(self, now):
        self.recognize_pending(now)
        if self.text and not self.text.endswith(" "):
            self.text += " "
        self.last_pred = None

    def delete(self):
        if self.pending_letter():
            self.remove_pending()                    # first discard an unfinished letter
        elif self.text:
            self.text = self.text[:-1]
            self.last_pred = None

    # --- actions shared by toolbar and keyboard
    def act(self, key, now):
        if key.startswith("color:"):
            self.color_idx, self.tool = int(key.split(":")[1]), "pen"
            if self.mode == "text":
                self.set_mode("paint", now)
        elif key == "eraser":
            self.tool = "eraser"
        elif key == "mode":
            self.set_mode("text" if self.mode == "paint" else "paint", now)
        elif key == "space":
            self.add_space(now)
        elif key == "delete":
            self.delete()
        elif key == "clear":
            self.strokes, self.current, self.text, self.last_pred = [], None, "", None
            self.canvas[:] = 0

    def set_mode(self, mode, now):
        self.finish_stroke()
        if mode == "paint":
            self.recognize_pending(now)
        elif self.recognizer is None:
            self.say("Text mode needs letter_mlp.npz next to the script", 4)
            return
        self.mode, self.tool = mode, "pen"
        self.say("Text mode: write CAPITAL letters, pause after each" if mode == "text" else "Paint mode")

    # --- per-frame update
    def update(self, gesture, tip, now):
        self.now = now
        drawing = gesture == "write" and tip is not None and tip[1] >= TOOLBAR_H

        if drawing and self.tool == "eraser":
            self.finish_stroke()
            self.strokes, changed = erase_at(self.strokes, tip, ERASER_RADIUS)
            if changed:
                redraw(self.canvas, self.strokes)
            self.last_activity = now
        elif drawing:
            if self.current is None:
                self.current = {"pts": [tip], "color": self.color, "thick": THICKNESS,
                                "letter": self.mode == "text"}
            elif math.dist(self.current["pts"][-1], tip) >= MIN_STEP_PX:
                cv2.line(self.canvas, self.current["pts"][-1], tip, self.current["color"],
                         THICKNESS, cv2.LINE_AA)
                self.current["pts"].append(tip)
            self.last_activity = now
        else:
            self.finish_stroke()

        # toolbar hovering
        btn = button_at(self.buttons, tip) if gesture == "select" else None
        if btn is not self.hover_btn:
            self.hover_btn, self.hover_since, self.hover_armed = btn, now, True
        elif btn is not None and self.hover_armed and now - self.hover_since >= HOVER_SECONDS:
            self.hover_armed = False                 # one press per hover
            self.act(btn["key"], now)

        # fist to clear
        if gesture == "fist":
            if self.fist_since is None:
                self.fist_since, self.fist_done = now, False
            elif not self.fist_done and now - self.fist_since >= FIST_SECONDS:
                self.act("clear", now)
                self.fist_done = True
                self.say("Cleared")
        else:
            self.fist_since = None

        # text mode: recognize after a pause
        if (self.mode == "text" and self.current is None and self.pending_letter()
                and now - self.last_activity >= self.letter_pause):
            self.recognize_pending(now)

    # --- rendering
    def render(self, frame, gesture, tip, palm, now, fps):
        composite(frame, self.canvas, self.glow)
        self.draw_pending_timer(frame, now)
        self.draw_pops(frame, now)
        self.draw_toolbar(frame, now)
        if self.mode == "text":
            self.draw_text_bar(frame, now)

        if gesture == "fist" and self.fist_since is not None and palm is not None and not self.fist_done:
            frac = min(1.0, (now - self.fist_since) / FIST_SECONDS)
            cv2.ellipse(frame, palm, (40, 40), -90, 0, 360 * frac, (80, 80, 255), 6, cv2.LINE_AA)
            put_text(frame, "Clear", (palm[0] - 30, palm[1] + 65), 0.6, (80, 80, 255))

        if tip is not None:
            if gesture == "write" and self.tool == "eraser":
                cv2.circle(frame, tip, ERASER_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
            elif gesture == "write":
                cv2.circle(frame, tip, 9, self.color, -1, cv2.LINE_AA)
                cv2.circle(frame, tip, 9, (255, 255, 255), 2, cv2.LINE_AA)
            elif gesture == "select":
                cv2.circle(frame, tip, 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, tip, 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, tip, 6, (160, 160, 160), 2, cv2.LINE_AA)

        status = {"write": "Erasing" if self.tool == "eraser" else "Writing", "select": "Pointer",
                  "fist": "Fist", "idle": "Pen up"}.get(gesture, "No hand") if tip or palm else "No hand"
        put_text(frame, f"{'TEXT' if self.mode == 'text' else 'PAINT'}  |  {status}", (16, TOOLBAR_H + 34), 0.8)
        put_text(frame, f"{fps:4.1f} fps  smoothing {'on' if self.smoothing else 'off'}  "
                        f"glow {'on' if self.glow else 'off'}", (16, TOOLBAR_H + 62), 0.5)
        if now < self.message_until:
            put_text(frame, self.message, (16, TOOLBAR_H + 92), 0.65, (0, 255, 255))

    def draw_toolbar(self, frame, now):
        bar = frame[:TOOLBAR_H].copy()
        cv2.rectangle(bar, (0, 0), (self.w, TOOLBAR_H), (25, 25, 25), -1)
        for b in self.buttons:
            x0, x1, cx = b["x0"] + 6, b["x1"] - 6, (b["x0"] + b["x1"]) // 2
            active = ((b["key"] == f"color:{self.color_idx}" and self.tool == "pen" and self.mode == "paint")
                      or (b["key"] == "eraser" and self.tool == "eraser")
                      or (b["key"] == "mode" and self.mode == "text"))
            dim = b["key"] in ("space", "delete") and self.mode != "text"
            cv2.rectangle(bar, (x0, 8), (x1, TOOLBAR_H - 8), (60, 60, 60) if active else (40, 40, 40), -1)
            if b["key"].startswith("color:"):
                cv2.circle(bar, (cx, 32), 13, b["color"], -1, cv2.LINE_AA)
            elif b["key"] == "eraser":
                cv2.circle(bar, (cx, 32), 13, b["color"], 2, cv2.LINE_AA)
            label_col = (110, 110, 110) if dim else b["color"]
            tw = text_width(b["label"], 0.5, 1)
            label_y = 62 if (b["key"].startswith("color:") or b["key"] == "eraser") else 48
            cv2.putText(bar, b["label"], (cx - tw // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        label_col, 1, cv2.LINE_AA)
            if active:
                cv2.rectangle(bar, (x0, 8), (x1, TOOLBAR_H - 8), (255, 255, 255), 2)
            if b is self.hover_btn and self.hover_armed:
                frac = min(1.0, (now - self.hover_since) / HOVER_SECONDS)
                cv2.rectangle(bar, (x0, TOOLBAR_H - 14), (int(x0 + (x1 - x0) * frac), TOOLBAR_H - 8),
                              (0, 255, 255), -1)
        cv2.addWeighted(bar, 0.85, frame[:TOOLBAR_H], 0.15, 0, dst=frame[:TOOLBAR_H])

    def draw_text_bar(self, frame, now):
        y0 = self.h - 110
        region = frame[y0:self.h - 30]
        cv2.addWeighted(region, 0.35, np.zeros_like(region), 0.65, 0, dst=region)
        cursor = "|" if int(now * 2) % 2 == 0 else " "
        shown, scale = self.text, 1.6
        while shown and text_width(shown + "|", scale, 3) > self.w - 40:
            shown = shown[1:]
        put_text(frame, shown + cursor, (20, self.h - 50), scale, (255, 255, 255), 3)
        if self.last_pred:
            alts = "   ".join(f"{i + 1}:{l} {p * 100:.0f}%" for i, (l, p) in enumerate(self.last_pred["preds"]))
            put_text(frame, f"Guesses  {alts}     (press 1-3 or Shift+letter to correct)",
                     (20, y0 - 10), 0.55, (200, 200, 200), 1)
        put_text(frame, "m mode  g glow  z undo  c clear  s smooth  e export  q quit",
                 (20, self.h - 10), 0.5, (200, 200, 200), 1)

    def draw_pending_timer(self, frame, now):
        pending = self.pending_letter()
        if self.mode != "text" or not pending or self.current is not None:
            return
        pts = np.concatenate([np.array(s["pts"], np.int32) for s in pending])
        x0, y0 = (int(v) - 15 for v in pts.min(axis=0))
        x1, y1 = (int(v) + 15 for v in pts.max(axis=0))
        frac = min(1.0, (now - self.last_activity) / self.letter_pause)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 120, 120), 1)
        cv2.rectangle(frame, (x0, y1 + 6), (int(x0 + (x1 - x0) * frac), y1 + 12), (0, 255, 255), -1)

    def draw_pops(self, frame, now):
        alive = []
        for letter, (cx, cy), size, t0 in self.pops:
            age = (now - t0) / 0.7
            if age >= 1:
                continue
            alive.append((letter, (cx, cy), size, t0))
            scale = max(1.0, size / 30) * (1 + 0.4 * age)
            thick = max(2, int(scale * 2))
            tw = text_width(letter, scale, thick)
            overlay = frame.copy()
            cv2.putText(overlay, letter, (cx - tw // 2, cy + int(scale * 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 255, 255), thick, cv2.LINE_AA)
            cv2.addWeighted(overlay, 1 - age, frame, age, 0, dst=frame)      # fade out
        self.pops = alive

    def export(self):
        path = os.path.join(HERE, time.strftime("air_%Y%m%d_%H%M%S.json"))
        with open(path, "w") as f:
            json.dump({"width": self.w, "height": self.h, "text": self.text,
                       "strokes": [{"pts": [list(p) for p in s["pts"]], "color": list(s["color"])}
                                   for s in self.strokes]}, f)
        cv2.imwrite(path.replace(".json", ".png"), self.canvas)
        self.say(f"Saved {os.path.basename(path)} and .png")

    def key(self, k, now):
        self.now = now
        if k in (ord("q"), 27):
            return False
        if k == ord("m"):
            self.act("mode", now)
        elif k == ord("g"):
            self.glow = not self.glow
        elif k == ord("z"):
            self.finish_stroke()
            if self.strokes:
                self.strokes.pop()
                redraw(self.canvas, self.strokes)
        elif k == ord("c"):
            self.act("clear", now)
        elif k == ord("e"):
            self.export()
        elif k == ord(" ") and self.mode == "text":
            self.add_space(now)
        elif k in (8, 127) and self.mode == "text":
            self.delete()
        elif k in (ord("1"), ord("2"), ord("3")) and self.last_pred:
            i = k - ord("1")
            if i < len(self.last_pred["preds"]):
                self.correct_last(self.last_pred["preds"][i][0])
        elif ord("A") <= k <= ord("Z") and self.mode == "text":
            self.correct_last(chr(k))
        return True


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Paint and write text in the air.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--debounce", type=int, default=3, help="frames needed to change gesture")
    ap.add_argument("--letter-pause", type=float, default=1.0, help="seconds of pause that end a letter")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from camera {args.camera}. Close other apps using it, or try --camera 1.")
    h, w = frame.shape[:2]

    tracker = HandTracker()
    app = App(w, h, args.letter_pause)
    gestures = GestureDebouncer(args.debounce)
    fx, fy = OneEuroFilter(), OneEuroFilter()
    fps, t_prev = 0.0, time.monotonic()

    window = "Air Writing"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            norm = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            tip = palm = None
            raw = "none"
            if norm is not None:
                pts = [(x * w, y * h) for x, y in norm]
                raw = classify_gesture(pts)
                tx = min(max(pts[INDEX_TIP][0], 0), w - 1)
                ty = min(max(pts[INDEX_TIP][1], 0), h - 1)
                if app.smoothing:
                    tx, ty = fx(tx, now), fy(ty, now)
                tip = (int(round(tx)), int(round(ty)))
                palm = (int(pts[PALM][0]), int(pts[PALM][1]))
            else:
                fx.reset()
                fy.reset()
            gesture = gestures.update(raw)

            app.update(gesture, tip, now)

            dt, t_prev = now - t_prev, now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt
            app.render(frame, gesture, tip, palm, now, fps)
            cv2.imshow(window, frame)

            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                if k == ord("s"):
                    app.smoothing = not app.smoothing
                    fx.reset()
                    fy.reset()
                elif not app.key(k, now):
                    break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import json

import numpy as np
import pytest

from letter_model import LETTERS
from recording import Recorder, load_samples, safe_name, stroke_points
from train_letters import TEMPLATES, augment


def synthetic_strokes(letter, rng, t0=100.0, size=300, origin=(400, 200)):
    """A letter from the training templates, in pixels, with app-style absolute timestamps."""
    strokes, _ = augment(TEMPLATES[letter][0], rng)
    out, t = [], t0
    for s in strokes:
        pts = [(int(origin[0] + x * size), int(origin[1] + y * size)) for x, y in s]
        times = [t + 0.03 * i for i in range(len(pts))]
        t = times[-1] + 0.2
        out.append({"pts": pts, "t": times})
    return out


def test_safe_name():
    assert safe_name(" Alex B. ") == "Alex_B"
    with pytest.raises(ValueError):
        safe_name("../")


def test_prompts_every_letter_each_round(tmp_path):
    rec = Recorder("a", tmp_path, rounds=2, seed=0)
    assert rec.total == 52
    assert sorted(rec.queue) == sorted(LETTERS * 2)
    assert sorted(rec.queue[:26]) == list(LETTERS)


def test_save_undo_skip(tmp_path):
    rng = np.random.default_rng(0)
    rec = Recorder("alex", tmp_path, rounds=1, letters="ABC", seed=1)
    first, second = rec.queue[0], rec.queue[1]

    s = rec.save(synthetic_strokes(first, rng), (1280, 720), True, 1.0)
    assert s["label"] == first and rec.target == second
    assert s["strokes"][0]["t"][0] == 0.0                     # times relative to the letter start
    rec.save(synthetic_strokes(second, rng), (1280, 720), True, 1.0)

    assert rec.undo() == second and rec.target == second
    lines = open(rec.path).read().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["label"] == first

    third = rec.queue[1]
    rec.skip()
    assert rec.queue == [third, second]


def test_load_samples_skips_bad_lines(tmp_path):
    rng = np.random.default_rng(0)
    rec = Recorder("alex", tmp_path, rounds=1, letters="AB", seed=0)
    rec.save(synthetic_strokes(rec.target, rng), (640, 480), False, 1.0)
    with open(rec.path, "a") as f:
        f.write("not json\n")
        f.write(json.dumps({"schema": 1, "label": "a", "strokes": []}) + "\n")
        f.write("\n")

    samples, problems = load_samples([str(tmp_path)])
    assert len(samples) == 1 and len(problems) == 2
    assert stroke_points(samples[0])[0][0] == samples[0]["strokes"][0]["pts"][0]


def test_app_records_a_letter(tmp_path):
    """Drive the real App with fake fingertip positions: write, pause, check the saved sample."""
    from air_writing import App

    rec = Recorder("tester", tmp_path, rounds=1, letters="L", seed=0)
    app = App(1280, 720, 1.0, rec)
    assert app.mode == "record"

    t = 10.0
    for y in range(200, 501, 10):              # down stroke of the L
        app.update("write", (400, y), t)
        t += 0.03
    app.update("idle", (400, 500), t)
    t += 0.2
    for x in range(400, 601, 10):              # bottom stroke
        app.update("write", (x, 500), t)
        t += 0.03
    app.update("idle", (600, 500), t)
    app.update("idle", (600, 500), t + 1.5)    # pause ends the letter

    assert rec.done and app.strokes == []
    sample = json.loads(open(rec.path).read())
    assert sample["label"] == "L" and sample["frame"] == [1280, 720]
    assert len(sample["strokes"]) == 2
    for st in sample["strokes"]:
        assert len(st["pts"]) == len(st["t"])
        assert st["t"] == sorted(st["t"])

    app.record_back()                          # Backspace: redo the saved letter
    assert rec.target == "L" and open(rec.path).read() == ""


def test_erasing_keeps_timestamps_aligned():
    from air_writing import erase_at

    stroke = {"pts": [(x, 0) for x in range(0, 100, 10)], "t": [float(i) for i in range(10)]}
    pieces, changed = erase_at([stroke], (50, 0), 15)
    assert changed and len(pieces) == 2
    for p in pieces:
        assert len(p["pts"]) == len(p["t"])
    assert pieces[1]["t"][0] == 7.0

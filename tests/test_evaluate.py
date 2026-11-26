import json

import numpy as np
import pytest

import evaluate
from recording import Recorder
from test_recording import synthetic_strokes


def test_wilson_interval():
    lo, hi = evaluate.wilson_interval(90, 100)
    assert lo == pytest.approx(0.8256, abs=1e-3) and hi == pytest.approx(0.9448, abs=1e-3)
    assert evaluate.wilson_interval(10, 10)[1] == 1.0
    assert evaluate.wilson_interval(0, 0) == (0.0, 0.0)


def test_mcnemar():
    assert evaluate.mcnemar_p(0, 0) == 1.0
    assert evaluate.mcnemar_p(5, 5) == 1.0
    assert evaluate.mcnemar_p(10, 0) == pytest.approx(2 / 2 ** 10)
    assert evaluate.mcnemar_p(3, 7) == pytest.approx(evaluate.mcnemar_p(7, 3))


@pytest.fixture
def recordings(tmp_path):
    """Two writers, every letter once, drawn from the synthetic templates."""
    rng = np.random.default_rng(42)
    for writer in ("ann", "bob"):
        rec = Recorder(writer, tmp_path / "data", rounds=1, seed=len(writer))
        while not rec.done:
            rec.save(synthetic_strokes(rec.target, rng), (1280, 720), True, 1.0)
    return tmp_path


def test_end_to_end(recordings, capsys):
    out = recordings
    reports = evaluate.main([str(out / "data"),
                             "--confusion", str(out / "cm.png"), "--errors", str(out / "err.png"),
                             "--json", str(out / "report.json")])
    rep = reports["letter_mlp.npz"]
    assert rep["n"] == 52
    assert rep["top1"] > 0.9                    # synthetic letters are the easy case
    assert rep["top3"] >= rep["top1"]
    assert set(rep["per_writer"]) == {"ann", "bob"}
    assert np.array(rep["confusion"]).sum() == 52

    text = capsys.readouterr().out
    assert "52 samples from 2 writer(s)" in text and "top-1 accuracy" in text
    assert (out / "cm.png").exists()
    saved = json.load(open(out / "report.json"))
    assert len(saved["predictions"]["letter_mlp.npz"]) == 52


def test_compare_same_model_twice(recordings, capsys):
    weights = evaluate.os.path.join(evaluate.HERE, "letter_mlp.npz")
    evaluate.main([str(recordings / "data"), "--weights", weights, weights, "--writer", "ann"])
    text = capsys.readouterr().out
    assert "26 samples from 1 writer(s)" in text
    assert "only first right 0,  only second right 0" in text


def test_no_data(tmp_path):
    with pytest.raises(SystemExit):
        evaluate.main([str(tmp_path)])

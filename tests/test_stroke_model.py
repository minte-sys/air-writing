import os

import numpy as np
import pytest

from stroke_model import N_FEATURES, SEQ_LEN, StrokeRecognizer, conv1d, stroke_features

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(HERE, "stroke_cnn.npz")
L_SHAPE = [[(0, 0), (0, 10)], [(0, 10), (6, 10)]]


def test_feature_shape_and_pen():
    f = stroke_features(L_SHAPE)
    assert f.shape == (SEQ_LEN, N_FEATURES) and f.dtype == np.float32
    assert set(np.unique(f[:, 4])) == {1.0}          # strokes touch: no pen-up distance
    f2 = stroke_features([[(0, 0), (10, 0)], [(0, 10), (10, 10)]])
    assert 0.0 in f2[:, 4] and 1.0 in f2[:, 4]       # jump between strokes is marked pen up
    assert np.abs(f[:, :2]).max() <= 1.0 + 1e-6


def test_features_ignore_position_and_size():
    a = stroke_features(L_SHAPE)
    b = stroke_features([[(x * 7 + 300, y * 7 + 50) for x, y in s] for s in L_SHAPE])
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_features_keep_direction():
    fwd = stroke_features([[(0, 0), (10, 0)]])
    back = stroke_features([[(10, 0), (0, 0)]])
    assert fwd[0, 2] == pytest.approx(1.0) and back[0, 2] == pytest.approx(-1.0)


def test_degenerate_input():
    assert stroke_features([]) is None
    assert stroke_features([[(5, 5)]]) is None
    assert stroke_features([[(5, 5), (5, 5)]]) is None


def test_conv1d_matches_naive():
    rng = np.random.default_rng(0)
    x, W, b = rng.normal(size=(3, 9)), rng.normal(size=(4, 3, 5)), rng.normal(size=4)
    xp = np.pad(x, ((0, 0), (2, 2)))
    naive = np.array([[np.sum(W[o] * xp[:, t:t + 5]) + b[o] for t in range(9)] for o in range(4)])
    np.testing.assert_allclose(conv1d(x, W, b), naive, atol=1e-10)


def test_training_augmentations_keep_points():
    from train_strokes import split_a_stroke, vary_writing_order

    rng = np.random.default_rng(0)
    variant = [[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]]
    for _ in range(200):
        out = vary_writing_order(variant, rng)
        assert sorted(p for s in out for p in s) == sorted(p for s in variant for p in s)
    long = [np.arange(40.0).reshape(20, 2)]
    for _ in range(200):
        out = split_a_stroke(long, rng)
        assert 1 <= len(out) <= 2 and sum(len(s) for s in out) >= 18


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="stroke_cnn.npz not trained yet")
def test_trained_model():
    rec = StrokeRecognizer(WEIGHTS)
    assert rec.self_check()                          # NumPy reproduces PyTorch's outputs
    preds, img = rec.predict(L_SHAPE)
    assert preds[0][0] == "L" and img.shape == (28, 28)

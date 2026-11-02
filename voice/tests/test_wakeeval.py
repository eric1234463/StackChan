import numpy as np

from wakeword.wakeeval import REFRACTORY, WINDOW, detections


def test_fires_when_the_window_mean_reaches_the_cutoff():
    p = np.array([0.0] * 10 + [1.0] * 5 + [0.0] * 5)
    assert detections(p, 0.9) == [14]  # needs all 5 high frames for a 0.9 mean
    assert detections(p, 0.2) == [10]  # one high frame is enough for 0.2


def test_first_window_after_reset_is_ignored():
    p = np.array([1.0] * WINDOW + [0.0] * 10)
    assert detections(p, 0.5) == [WINDOW]  # the first WINDOW inferences are skipped, then the mean still holds


def test_refractory_suppresses_repeat_firings():
    p = np.array([0.0] * 5 + [1.0] * 60)
    hits = detections(p, 0.9)
    assert hits[1] - hits[0] == REFRACTORY + 1

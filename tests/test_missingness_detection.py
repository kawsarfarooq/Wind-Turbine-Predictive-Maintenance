import numpy as np

from experiments.missingness_detection_study import missing_mask


def test_missingness_masks_are_deterministic_and_nonempty():
    for pattern in ["random", "block", "dropout"]:
        a = missing_mask(1000, pattern, .1, 7)
        b = missing_mask(1000, pattern, .1, 7)
        np.testing.assert_array_equal(a, b)
        assert 0 < a.mean() <= .2

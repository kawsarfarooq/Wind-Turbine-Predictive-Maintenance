import pandas as pd

from experiments.care_temporal_analysis import exploratory_operating_points


def test_operating_points_respect_false_alarm_constraint():
    summary = pd.DataFrame({
        "farm": ["A", "A", "B", "B", "C", "C"],
        "smooth_steps": [6, 144, 36, 432, 36, 432],
        "threshold_quantile": [0.999, 0.995, 0.995, 0.999, 0.999, 0.99],
        "detection_rate": [0.5, 0.8, 0.5, 0.2, 0.4, 0.45],
        "normal_false_alarm_rate": [0.0, 0.4, 0.1, 0.0, 0.1, 0.1],
        "detection_minus_false_alarm": [0.5, 0.4, 0.4, 0.2, 0.3, 0.35],
    })

    selected = exploratory_operating_points(summary)

    assert selected.smooth_steps.tolist() == [6, 36, 432]
    assert (selected.normal_false_alarm_rate <= 0.2).all()

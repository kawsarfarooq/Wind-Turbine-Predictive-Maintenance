import numpy as np
import pandas as pd

from care_benchmark import empirical_percentile, summarize_events


def test_empirical_percentile_uses_training_distribution():
    train = np.array([1.0, 2.0, 3.0, 4.0])
    score = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
    got = empirical_percentile(train, score)
    np.testing.assert_allclose(got, [0.0, 0.25, 0.5, 1.0, 1.0])


def test_event_summary_compares_anomalies_with_normal_controls():
    events = pd.DataFrame({
        "farm": ["A"] * 4,
        "asset": ["a", "b", "c", "d"],
        "cluster_id": ["A:a", "A:b", "A:c", "A:d"],
        "representation": ["raw"] * 4,
        "detector": ["gmm"] * 4,
        "normal_statuses": ["0"] * 4,
        "label": ["normal", "normal", "anomaly", "anomaly"],
        "event_mean_percentile": [0.1, 0.2, 0.8, 0.9],
        "alarm": [False, False, True, True],
    })
    summary = summarize_events(events)
    farm = summary[summary["farm"] == "A"].iloc[0]
    assert farm["event_roc_auc"] == 1.0
    assert farm["event_pr_auc"] == 1.0
    assert np.isclose(farm["anomaly_normal_gap"], 0.7)
    assert farm["detection_rate"] == 1.0
    assert farm["normal_false_alarm_rate"] == 0.0
    assert 0.0 <= farm["event_roc_auc_ci_low"] <= 1.0
    assert 0.0 <= farm["event_roc_auc_ci_high"] <= 1.0

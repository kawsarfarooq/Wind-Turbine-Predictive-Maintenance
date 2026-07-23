import numpy as np
import pandas as pd

from care_benchmark import (build_representation, causal_rolling_median,
                            empirical_percentile, summarize_events,
                            temporal_score_grid)


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


def test_quadratic_residual_captures_nonlinear_normal_behaviour():
    power = np.linspace(-2.0, 2.0, 300)
    df = pd.DataFrame({
        "power_avg": power,
        "bearing_temp_avg": 3.0 * power ** 2 + 0.5 * power,
    })
    meta = {}
    train_mask = np.ones(len(df), dtype=bool)
    linear, _, _, _ = build_representation(
        df, meta, train_mask, "residual")
    quadratic, mode, _, _ = build_representation(
        df, meta, train_mask, "quadratic_residual")
    assert mode == "quadratic_residual"
    assert np.mean(quadratic ** 2) < 0.01 * np.mean(linear ** 2)


def test_causal_smoothing_does_not_use_future_values():
    original = causal_rolling_median([1.0, 9.0, 2.0, 4.0], 3)
    changed_future = causal_rolling_median([1.0, 9.0, 2.0, 400.0], 3)
    np.testing.assert_allclose(original[:3], changed_future[:3])


def test_temporal_grid_reuses_scores_across_windows_and_quantiles():
    percentile = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    train_mask = np.array([True, True, True, False, False])
    eval_mask = ~train_mask
    rows = temporal_score_grid(
        percentile, train_mask, eval_mask, [1, 2], [0.5, 0.9])

    assert len(rows) == 4
    assert {(row["smooth_steps"], row["threshold_quantile"])
            for row in rows} == {(1, 0.5), (1, 0.9), (2, 0.5), (2, 0.9)}
    for smooth_steps in [1, 2]:
        thresholds = [row["threshold_percentile"] for row in rows
                      if row["smooth_steps"] == smooth_steps]
        assert thresholds == sorted(thresholds)

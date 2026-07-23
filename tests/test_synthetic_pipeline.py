import numpy as np
import pandas as pd

from stage3_rul import daily_features
from stage4_dfl import episodes
from synth_data import generate_farm


def test_each_turbine_has_one_right_censored_final_episode():
    df = generate_farm(n_turbines=3, n_days=400, seed=7)

    for _, turbine in df.groupby("turbine"):
        by_episode = turbine.groupby("episode")["event_observed"].first()
        assert int((~by_episode).sum()) == 1
        assert not bool(by_episode.iloc[-1])

        censored = turbine[turbine["episode"] == by_episode.index[-1]]
        assert not censored["failed_now"].any()
        assert (censored["rul_h"] > 0).all()


def test_observed_episode_count_matches_failure_count():
    df = generate_farm(n_turbines=2, n_days=500, seed=3)

    for _, turbine in df.groupby("turbine"):
        observed = turbine.groupby("episode")["event_observed"].first().sum()
        assert int(observed) == int(turbine["failed_now"].sum())


def test_policy_episode_builder_excludes_censoring():
    df = generate_farm(n_turbines=1, n_days=500, seed=5)
    residual = np.zeros(len(df))
    feat = daily_features(df.reset_index(drop=True), residual)

    policy_episodes = list(episodes(feat))
    assert len(policy_episodes) == int(df["failed_now"].sum())


def test_rolling_features_reset_at_episode_boundary():
    n_days = 16
    episode = np.repeat([0, 1], 8 * 24)
    df = pd.DataFrame({
        "episode": episode,
        "rul_h": np.tile(np.arange(8 * 24, 0, -1), 2),
        "event_observed": True,
    })
    residual = np.r_[np.full(8 * 24, 10.0), np.zeros(8 * 24)]

    feat = daily_features(df, residual)
    first_new_episode = feat.iloc[8]
    assert first_new_episode["res_mean"] == 0.0
    assert first_new_episode["res_mean_7d"] == 0.0
    assert first_new_episode["res_trend_7d"] == 0.0
    assert len(feat) == n_days


def test_noise_levels_preserve_failure_pairing_and_nonnegative_rul():
    clean = generate_farm(n_turbines=2, n_days=400, seed=11, noise_level=0.0)
    noisy = generate_farm(n_turbines=2, n_days=400, seed=11, noise_level=1.5)

    pd.testing.assert_series_equal(clean["failed_now"], noisy["failed_now"])
    pd.testing.assert_series_equal(clean["event_observed"], noisy["event_observed"])
    pd.testing.assert_series_equal(clean["rul_h"], noisy["rul_h"])
    assert (noisy["rul_h"] >= 0).all()


def test_enlarging_farm_keeps_existing_turbines_identical():
    small = generate_farm(n_turbines=2, n_days=200, seed=13, noise_level=1.0)
    large = generate_farm(n_turbines=4, n_days=200, seed=13, noise_level=1.0)

    pd.testing.assert_frame_equal(
        small.reset_index(drop=True),
        large[large["turbine"] < 2].reset_index(drop=True),
    )

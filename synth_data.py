"""Synthetic wind-turbine SCADA generator with degradation + failure logs.

Produces hourly data per turbine:
    wind speed -> power (cubic, capped at rated)
    bearing temperature = f(power, ambient) + degradation drift + noise
Each turbine goes through repeated run-to-failure episodes: a latent
degradation ramp starts some random time before failure and pushes the
bearing temperature up. Failure times are logged, so RUL labels and
detection lead times can be computed.

TODO(real data): replace `generate_farm()` with a loader for the EDP
Open Data wind farm SCADA + failure logs, keeping the same column names.
"""
import numpy as np
import pandas as pd

RATED_POWER = 2000.0  # kW
COLS = ["turbine", "t", "wind", "ambient", "power", "bearing_temp",
        "episode", "rul_h", "event_observed", "failed_now"]


def _wind_series(n, rng):
    """AR(1) wind speed, m/s, clipped to [0, 25]."""
    w = np.empty(n)
    w[0] = 8.0
    for i in range(1, n):
        w[i] = 0.97 * w[i - 1] + 0.03 * 9.0 + rng.normal(0, 0.7)
    return np.clip(w, 0.0, 25.0)


def _power_curve(wind):
    """Simple cubic power curve with cut-in 3 m/s, rated 12 m/s, cut-out 25."""
    p = RATED_POWER * ((wind - 3.0) / (12.0 - 3.0)) ** 3
    p = np.where(wind < 3.0, 0.0, p)
    p = np.where(wind >= 12.0, RATED_POWER, p)
    return np.clip(p, 0.0, RATED_POWER)


def _one_turbine(tid, n_hours, rng, noise_level=0.0, noise_seed=0):
    """noise_level=0 reproduces the original clean generator exactly.
    noise_level>0 degrades the observability of failures three ways:
      * stronger measurement noise on bearing temperature
      * false temperature spikes unrelated to any failure
      * weaker / shorter degradation ramps (some failures give little warning)
    A separate RNG stream is used for the noise so the base data is identical
    across noise levels (paired comparison)."""
    wind = _wind_series(n_hours, rng)
    power = _power_curve(wind) * (1 + rng.normal(0, 0.02, n_hours))
    day = np.arange(n_hours) / 24.0
    ambient = 12.0 + 8.0 * np.sin(2 * np.pi * day / 365.0) \
        + 3.0 * np.sin(2 * np.pi * day) + rng.normal(0, 0.8, n_hours)

    # healthy bearing temp: ambient + load-dependent heating
    base_temp = ambient + 18.0 * (power / RATED_POWER) + rng.normal(0, 0.9, n_hours)

    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    # The noise stream is independent of the base simulator but still changes
    # across simulator seeds. Holding seed/turbine fixed gives a paired noise
    # experiment, while changing seed now produces a genuinely new replicate.
    nrng = np.random.default_rng(
        np.random.SeedSequence([int(noise_seed), 10_000, int(tid)]))

    # degradation episodes: failure every ~90-200 days, ramp starts 15-45 d before
    degr = np.zeros(n_hours)
    episode = np.zeros(n_hours, dtype=int)
    rul = np.full(n_hours, np.nan)
    event_observed = np.zeros(n_hours, dtype=bool)
    failed_now = np.zeros(n_hours, dtype=bool)
    t, ep = 0, 0
    while t < n_hours:
        life = int(rng.uniform(90, 200) * 24)
        latent_fail_t = t + life
        episode_end = min(latent_fail_t, n_hours - 1)
        observed = latent_fail_t < n_hours
        ramp_len = int(rng.uniform(15, 45) * 24)
        amp = 10.0
        if noise_level > 0:                       # weak-warning failures
            # Bounds keep high-noise experiments physically meaningful:
            # degradation can become weak/short, but never negative.
            len_low = max(0.2, 1.0 - 0.8 * noise_level)
            amp_low = max(0.1, 1.0 - 0.7 * noise_level)
            ramp_len = int(ramp_len * nrng.uniform(len_low, 1.0))
            amp = amp * nrng.uniform(amp_low, 1.0)
        ramp_len = max(ramp_len, 24)
        ramp_start = max(t, latent_fail_t - ramp_len)
        if ramp_start <= episode_end:
            idx = np.arange(ramp_start, episode_end + 1)
            frac = (idx - ramp_start) / max(1, latent_fail_t - ramp_start)
            degr[idx] += amp * frac ** 2
        idx_episode = np.arange(t, episode_end + 1)
        episode[idx_episode] = ep
        rul[idx_episode] = latent_fail_t - idx_episode
        event_observed[idx_episode] = observed
        if observed:
            failed_now[latent_fail_t] = True
        t = episode_end + 1
        ep += 1

    temp = base_temp + degr
    if noise_level > 0:
        temp = temp + nrng.normal(0, 1.8 * noise_level, n_hours)
        n_spikes = nrng.poisson(6 * noise_level * (n_hours / (365 * 24)))
        for _ in range(n_spikes):                 # false spikes (no failure)
            s = nrng.integers(0, n_hours - 48)
            dur = nrng.integers(6, 48)
            temp[s:s + dur] += nrng.uniform(4, 9)

    df = pd.DataFrame({
        "turbine": tid, "t": np.arange(n_hours), "wind": wind,
        "ambient": ambient, "power": power,
        "bearing_temp": temp,
        "episode": episode, "rul_h": rul,
        "event_observed": event_observed, "failed_now": failed_now,
    })
    return df


def generate_farm(n_turbines=5, n_days=730, seed=0, noise_level=0.0):
    rng = np.random.default_rng(seed)
    frames = [_one_turbine(i, n_days * 24, rng, noise_level, noise_seed=seed)
              for i in range(n_turbines)]
    return pd.concat(frames, ignore_index=True)[COLS]


if __name__ == "__main__":
    df = generate_farm()
    print(df.head())
    print("rows:", len(df), "failures:", int(df.failed_now.sum()))

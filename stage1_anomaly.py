"""Stage 1 -- anomaly detection (Lecture Blocks 1-2).

1. Normal-behaviour model: bearing_temp ~ power + ambient (fit on healthy data).
2. KDE on residuals with a sliding window -> anomaly score = -log density.
3. GMM on multivariate features (Block 2 alternative).
4. Autoencoder (MLP, identity target) reconstruction error (Block 2 alternative).
Threshold calibration: quantile of healthy-period scores.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = ["wind", "ambient", "power", "bearing_temp"]


def healthy_mask(df, guard_h=45 * 24):
    """'Healthy' = more than guard_h hours away from the next failure."""
    return df["rul_h"] > guard_h


def fit_normal_behaviour(df_healthy):
    X = df_healthy[["power", "ambient"]].values
    y = df_healthy["bearing_temp"].values
    return LinearRegression().fit(X, y)


def residuals(df, nb_model):
    pred = nb_model.predict(df[["power", "ambient"]].values)
    return df["bearing_temp"].values - pred


def kde_sliding_score(res, window=30 * 24, step=24, bandwidth=0.4):
    """-log density of each new day of residuals under a KDE fit on the
    preceding `window` hours. Returns per-sample scores (nan inside warmup)."""
    scores = np.full(len(res), np.nan)
    for start in range(window, len(res), step):
        ref = res[start - window:start].reshape(-1, 1)
        kde = KernelDensity(bandwidth=bandwidth).fit(ref)
        blk = res[start:start + step].reshape(-1, 1)
        scores[start:start + step] = -kde.score_samples(blk)
    return scores


class GMMDetector:
    def __init__(self, n_components=4):
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_components, random_state=0)

    def fit(self, df_healthy):
        X = self.scaler.fit_transform(df_healthy[FEATURES].values)
        self.gmm.fit(X)
        return self

    def score(self, df):
        X = self.scaler.transform(df[FEATURES].values)
        return -self.gmm.score_samples(X)


class AEDetector:
    """Autoencoder via MLPRegressor with identity targets (bottleneck=2)."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.ae = MLPRegressor(hidden_layer_sizes=(8, 2, 8), max_iter=300,
                               random_state=0)

    def fit(self, df_healthy, max_rows=20000):
        X = self.scaler.fit_transform(df_healthy[FEATURES].values)
        if len(X) > max_rows:
            idx = np.random.default_rng(0).choice(len(X), max_rows, replace=False)
            X = X[idx]
        self.ae.fit(X, X)
        return self

    def score(self, df):
        X = self.scaler.transform(df[FEATURES].values)
        rec = self.ae.predict(X)
        return ((X - rec) ** 2).mean(axis=1)


def calibrate_threshold(healthy_scores, q=0.995):
    return np.nanquantile(healthy_scores, q)


def daily_alarm(scores, thr, min_frac=0.5):
    """Alarm on a day if >= min_frac of its hourly scores exceed thr."""
    n_days = len(scores) // 24
    s = scores[:n_days * 24].reshape(n_days, 24)
    with np.errstate(invalid="ignore"):
        frac = np.nanmean(s > thr, axis=1)
    return frac >= min_frac


def detection_lead_times(df_t, alarms_daily):
    """Hours between first sustained alarm and each failure in one turbine."""
    leads = []
    fail_hours = df_t.index[df_t["failed_now"]].values
    for fh in fail_hours:
        fd = fh // 24
        ep = df_t.loc[fh, "episode"]
        ep_days = df_t.index[df_t["episode"] == ep].values // 24
        d0, d1 = ep_days.min(), fd
        alarm_days = [d for d in range(d0, d1 + 1)
                      if d < len(alarms_daily) and alarms_daily[d]]
        leads.append((fd - alarm_days[0]) * 24 if alarm_days else None)
    return leads

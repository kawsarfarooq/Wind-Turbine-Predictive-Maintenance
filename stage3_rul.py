"""Stage 3 -- failure-within-N-days classifier (Lecture Block 4).

Daily features per turbine: rolling stats of the normal-behaviour residual.
Label: failure occurs within the next N days.
Model: gradient boosting. (Hyperparameter tuning hook: plug these params
into any Bayesian-optimisation loop, e.g. scikit-optimize's gp_minimize.)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

N_DAYS_HORIZON = 14


def daily_features(df_t, res):
    """One row per day: residual mean/max + 7-day trend, plus label & rul."""
    n_days = len(df_t) // 24
    r = res[:n_days * 24].reshape(n_days, 24)
    hourly_episode = df_t["episode"].values[:n_days * 24].reshape(n_days, 24)
    feat = pd.DataFrame({
        "res_mean": r.mean(axis=1),
        "res_max": r.max(axis=1),
        "episode": hourly_episode[:, -1],
    })
    grouped = feat.groupby("episode", sort=False)["res_mean"]
    feat["res_mean_7d"] = grouped.transform(
        lambda x: x.rolling(7, min_periods=1).mean())
    feat["res_trend_7d"] = grouped.transform(
        lambda x: x.diff(7).fillna(0.0))
    rul_d = df_t["rul_h"].values[:n_days * 24].reshape(n_days, 24)[:, -1] / 24.0
    feat["rul_days"] = rul_d
    feat["label"] = (rul_d <= N_DAYS_HORIZON).astype(int)
    if "event_observed" in df_t:
        feat["event_observed"] = (
            df_t["event_observed"].values[:n_days * 24]
            .reshape(n_days, 24)[:, -1].astype(bool))
    else:
        # Backward compatibility for external data frames created before the
        # censoring flag was introduced.
        feat["event_observed"] = True
    return feat


def train_classifier(feat_train):
    X = feat_train[["res_mean", "res_max", "res_mean_7d", "res_trend_7d"]].values
    y = feat_train["label"].values
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X, y)
    return clf


def predict_proba(clf, feat):
    X = feat[["res_mean", "res_max", "res_mean_7d", "res_trend_7d"]].values
    return clf.predict_proba(X)[:, 1]

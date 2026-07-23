"""End-to-end driver: synthetic farm -> Stage 1 -> 2 -> 3 -> 4 comparison.

Turbines 0-2 = train, 3-4 = test (turbine-level split, no leakage).
"""
import numpy as np
import pandas as pd

from synth_data import generate_farm
from stage1_anomaly import (healthy_mask, fit_normal_behaviour, residuals,
                            kde_sliding_score, GMMDetector, AEDetector,
                            calibrate_threshold)
from stage2_imputation import run_imputation_study
from stage3_rul import daily_features, train_classifier
from stage4_dfl import (episodes, train_dfl, policy_dfl, policy_reactive,
                        policy_threshold, policy_classifier, evaluate)

TRAIN_T, TEST_T = [0, 1, 2], [3, 4]


def main():
    df = generate_farm(n_turbines=5, n_days=730, seed=0)
    print(f"data: {len(df)} rows, {int(df.failed_now.sum())} failures\n")

    # ---------- Stage 1 ----------
    train_df = df[df.turbine.isin(TRAIN_T)]
    nb = fit_normal_behaviour(train_df[healthy_mask(train_df)])

    t0 = df[df.turbine == 0].reset_index(drop=True)
    res0 = residuals(t0, nb)
    kde_scores = kde_sliding_score(res0[:120 * 24])          # first 120 days
    h = healthy_mask(t0)[:120 * 24].values
    thr = calibrate_threshold(kde_scores[h])
    print("Stage 1 (turbine 0, first 120 d):")
    print(f"  KDE threshold (99.5% healthy quantile): {thr:.2f}")

    gmm = GMMDetector().fit(train_df[healthy_mask(train_df)])
    ae = AEDetector().fit(train_df[healthy_mask(train_df)])
    t3 = df[df.turbine == 3].reset_index(drop=True)
    pre = t3["rul_h"] <= 24 * 3          # last 3 days before failure
    ok = t3["rul_h"] > 24 * 45
    for name, det in [("GMM", gmm), ("Autoencoder", ae)]:
        s = det.score(t3)
        print(f"  {name}: healthy score {np.mean(s[ok]):.2f} vs "
              f"pre-failure {np.mean(s[pre]):.2f}")

    # ---------- Stage 2 ----------
    print("\nStage 2 imputation RMSE (deg C), turbine 0:")
    for k, v in run_imputation_study(t0["bearing_temp"].values[:5000]).items():
        print(f"  {k:7s}: {v:.3f}")

    # ---------- Stage 3 ----------
    feats = {}
    for tid in TRAIN_T + TEST_T:
        dt = df[df.turbine == tid].reset_index(drop=True)
        feats[tid] = daily_features(dt, residuals(dt, nb))
    feat_train = pd.concat([feats[t] for t in TRAIN_T], ignore_index=True)
    clf = train_classifier(feat_train.dropna(subset=["rul_days"]))
    print("\nStage 3: classifier trained "
          f"({int(feat_train.label.sum())} positive days)")

    # ---------- Stage 4 ----------
    train_eps = [e for t in TRAIN_T for e in episodes(feats[t])]
    test_eps = [e for t in TEST_T for e in episodes(feats[t])]

    # standardize features for the linear DFL policy
    allX = np.vstack([X for X, _ in train_eps])
    mu, sd = allX.mean(0), allX.std(0) + 1e-9
    scale = lambda eps: [((X - mu) / sd, r) for X, r in eps]
    train_s, test_s = scale(train_eps), scale(test_eps)
    w, b = train_dfl(train_s, seed=0)

    policies = {
        "reactive (run-to-failure)": policy_reactive,
        "residual threshold": lambda X, r: policy_threshold(X, r, thr=3.0),
        "predict-then-optimize": lambda X, r: policy_classifier(X, r, clf, 0.5),
        "decision-focused (DFL)": lambda X, r: policy_dfl((X - mu) / sd, r, w, b),
    }
    print(f"\nStage 4 cost simulation on {len(test_eps)} held-out episodes "
          "(k EUR/episode):")
    for name, m in evaluate(policies, test_eps).items():
        print(f"  {name:28s} mean cost {m['mean_cost']:7.1f}   "
              f"failures avoided {100 * m['failures_avoided']:.0f}%")


if __name__ == "__main__":
    main()

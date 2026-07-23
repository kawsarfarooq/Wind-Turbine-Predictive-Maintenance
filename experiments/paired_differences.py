"""Paired per-episode cost differences: tuned PTO vs DFL (report table).

Because the noise design is paired (same failures at same times), the
statistically defensible comparison is per-episode: for each of the 10
held-out episodes, cost(PTO) - cost(DFL) at each noise level. DFL cost per
episode is averaged over 3 training seeds. Cost model: C_prev=50,
C_fail=300.

Outputs: paired_episode_differences.csv + printed summary.
Run:  python paired_differences.py     (~3-4 min)
"""
import numpy as np
import pandas as pd

from sweep_noise import build, tune_pto_threshold
import stage4_dfl as s4
from stage4_dfl import train_dfl, policy_dfl, policy_classifier, episode_cost

NOISE_LEVELS = [0.0, 1.0, 1.5]
N_SEEDS = 3
s4.C_FAIL = 300.0


def main():
    rows = []
    for noise in NOISE_LEVELS:
        clf, train_eps, test_eps, mu, sd = build(noise)
        train_s = [((X - mu) / sd, r) for X, r in train_eps]

        p_thr = tune_pto_threshold(clf, train_eps)
        pto = np.array([episode_cost(policy_classifier(X, r, clf, p_thr), r)
                        for X, r in test_eps])

        dfl_per_seed = []
        for seed in range(N_SEEDS):
            w, b = train_dfl(train_s, seed=seed)
            dfl_per_seed.append(
                [episode_cost(policy_dfl((X - mu) / sd, r, w, b), r)
                 for X, r in test_eps])
        dfl = np.mean(dfl_per_seed, axis=0)     # per-episode, seed-averaged

        for i, (p, d) in enumerate(zip(pto, dfl)):
            rows.append({"noise": noise, "episode": i, "pto_cost": p,
                         "dfl_cost": round(d, 1), "diff_pto_minus_dfl":
                         round(p - d, 1),
                         "cheaper": "DFL" if d < p else
                                    ("PTO" if p < d else "tie")})

        diff = pto - dfl
        n_dfl = int((diff > 0).sum())
        n_pto = int((diff < 0).sum())
        print(f"noise={noise:.1f}  mean diff (PTO-DFL) = {diff.mean():+6.1f} "
              f"+/- {diff.std():5.1f} k EUR   "
              f"DFL cheaper on {n_dfl}/10, PTO cheaper on {n_pto}/10"
              f"{', ties on ' + str(10 - n_dfl - n_pto) + '/10' if n_dfl + n_pto < 10 else ''}")

    pd.DataFrame(rows).to_csv("paired_episode_differences.csv", index=False)
    print("\nsaved: paired_episode_differences.csv")


if __name__ == "__main__":
    main()

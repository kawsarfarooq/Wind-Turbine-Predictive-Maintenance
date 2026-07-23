import pandas as pd

from experiments.cost_aware_benchmark import cluster_interval


def test_cluster_interval_is_ordered():
    rows = pd.DataFrame({
        "cluster": ["a", "a", "b", "b"],
        "cost": [1.0, 2.0, 3.0, 4.0],
    })
    low, high = cluster_interval(rows, n_boot=100, seed=1)
    assert low <= rows.cost.mean() <= high

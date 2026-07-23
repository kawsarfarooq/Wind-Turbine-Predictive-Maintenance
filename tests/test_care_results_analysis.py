import numpy as np
import pandas as pd

from experiments.care_results_analysis import ensure_cluster_ids


def test_ensure_cluster_ids_fills_only_missing_appended_rows():
    events = pd.DataFrame({
        "farm": ["A", "C"],
        "asset": ["T01", "T09"],
        "cluster_id": ["existing-cluster", np.nan],
    })

    result = ensure_cluster_ids(events)

    assert result.cluster_id.tolist() == ["existing-cluster", "C:T09"]

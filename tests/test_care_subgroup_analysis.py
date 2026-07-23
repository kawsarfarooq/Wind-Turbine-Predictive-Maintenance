import numpy as np
import pandas as pd

from experiments.care_subgroup_analysis import fault_category, leave_one_asset_out


def test_fault_category_uses_documented_keyword_priority():
    assert fault_category("Gearbox bearings damaged") == "Drivetrain / bearings"
    assert fault_category("Pitch failure - defect fan") == "Pitch / blade / hub"
    assert (fault_category("Communication fault BK1120")
            == "Communication / control")
    assert (fault_category("Schwingungen Umrichter Drehmomenten Level 1")
            == "Electrical / converter / transformer")


def test_leave_one_asset_out_reports_auc_change_for_each_asset():
    events = pd.DataFrame({
        "farm": ["A"] * 6,
        "asset": ["x", "x", "y", "y", "z", "z"],
        "label": ["normal", "anomaly"] * 3,
        "event_mean_percentile": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
    })

    result = leave_one_asset_out(events)

    assert result.asset_removed.tolist() == ["x", "y", "z"]
    np.testing.assert_allclose(result.full_event_roc_auc, 1.0)
    np.testing.assert_allclose(result.leave_one_out_event_roc_auc, 1.0)
    np.testing.assert_allclose(result.delta_event_roc_auc, 0.0)

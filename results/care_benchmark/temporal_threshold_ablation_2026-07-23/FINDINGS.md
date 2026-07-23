# CARE temporal and threshold ablation findings

Configurations were fixed from the preceding representation ablation:
linear-GMM for Farm A, exploratory quadratic-IF for Farm B, and raw-GMM
for Farm C. Intervals are 95% asset-clustered bootstrap intervals.

| Farm | Configuration | Best smoothing | ROC-AUC (95% CI) | 24 h ROC-AUC | Detection / false alarms at 24 h, q=0.995 |
|---|---|---:|---:|---:|---:|
| A | linear-GMM | none (10 min) | 0.883 (0.648 to 1.000) | 0.858 | 50.0% / 20.0% |
| B | quadratic-IF | 24 h | 0.519 (0.204 to 0.778) | 0.519 | 16.7% / 22.2% |
| C | raw-GMM | none (10 min) | 0.762 (0.618 to 0.867) | 0.740 | 48.1% / 22.6% |

Exploratory operating points maximizing detection minus false alarms
subject to a normal-event false-alarm rate of at most 20%:

| Farm | Smoothing | Training quantile | Detection | Normal false alarms |
|---|---:|---:|---:|---:|
| A | 1 h | 0.999 | 50.0% | 0.0% |
| B | 6 h | 0.995 | 50.0% | 11.1% |
| C | 72 h | 0.990 | 44.4% | 9.7% |

Smoothing selection is exploratory because the same labeled CARE events
were used to compare windows; it is not an independently validated tuning
result. Threshold quantiles are estimated from smoothed scores at normal-
training timestamps, while labeled event controls are reserved for evaluation.

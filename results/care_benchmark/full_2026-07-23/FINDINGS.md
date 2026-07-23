# CARE full-benchmark findings

All intervals are 95% asset-clustered bootstrap intervals.

| Farm | Best method | ROC-AUC (95% CI) | Anomaly-normal gap | Detection | Normal false alarms |
|---|---|---:|---:|---:|---:|
| A | residual-gmm | 0.858 (0.639–1.000) | +0.092 | 50.0% | 20.0% |
| B | raw-iforest | 0.426 (0.127–0.810) | -0.047 | 0.0% | 0.0% |
| C | raw-gmm | 0.740 (0.594–0.848) | +0.146 | 48.1% | 22.6% |

Interpretation: residual normalization is strongly beneficial on Farm A,
does not rescue Farm B, and is inferior to raw GMM on Farm C. Therefore
condition normalization is a farm-dependent design choice, not a universal
improvement. Farm B remains a negative transfer result because normal
prediction windows score at least as highly as anomaly windows.

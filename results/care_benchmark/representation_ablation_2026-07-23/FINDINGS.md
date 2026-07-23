# CARE representation-ablation findings

All intervals are 95% asset-clustered bootstrap intervals.

| Farm | Best method | ROC-AUC (95% CI) | Anomaly-normal gap | Detection | Normal false alarms |
|---|---|---:|---:|---:|---:|
| A | linear-gmm | 0.858 (0.639–1.000) | +0.092 | 50.0% | 20.0% |
| B | quadratic-iforest | 0.519 (0.204–0.778) | -0.009 | 16.7% | 22.2% |
| C | raw-gmm | 0.740 (0.594–0.848) | +0.146 | 48.1% | 22.6% |

Interpretation: adding pairwise nonlinear operating-condition terms does
not outperform the strongest existing representation. Linear residuals
remain best on Farm A, raw features remain best on Farm C, and no tested
representation reliably separates Farm B. Normal-behaviour model complexity
is therefore a farm-dependent choice rather than a universal improvement.

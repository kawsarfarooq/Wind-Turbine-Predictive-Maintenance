# Wind Turbine Predictive Maintenance

This 9-CFU course project studies the full path from wind-turbine SCADA data to
maintenance decisions. It combines a real-data anomaly-detection benchmark on
the [CARE to Compare dataset](https://zenodo.org/records/15846963) with a
controlled synthetic study of missingness, failure prediction, and maintenance
cost. Real-data detection claims and synthetic cost claims are kept separate.

## Research question

> How robust are residual-based anomaly detectors across wind farms, operating
> conditions, fault types, and missing-data conditions, and how do their errors
> affect maintenance cost in a controlled synthetic environment?

## Main results

- **CARE benchmark (95 events, 36 assets):** linear residual-GMM is strongest
  on Farm A (ROC-AUC 0.858), no tested method reliably separates Farm B, and
  raw GMM is strongest on Farm C (ROC-AUC 0.740).
- **Model ablation:** a quadratic normal-behaviour model does not improve the
  best farm-specific representations. This is a retained negative result.
- **Temporal robustness:** causal smoothing and training-calibrated thresholds
  expose a clear detection/false-alarm trade-off; reported operating points are
  exploratory rather than independently validated.
- **Fault and asset analysis:** Farms A and C contain detectable subgroups;
  Farm B is heterogeneous and remains near chance. Farm C is most stable to
  removing one asset.
- **Missingness study:** linear interpolation has the lowest reconstruction
  error and strongest downstream discrimination in all tested corrupted
  settings. Under 30% sensor dropout, ROC-AUC falls from 0.980 to 0.824 and
  healthy-day false alarms rise from 11.0% to 39.7%.
- **Cost-aware benchmark:** across 62 held-out failure episodes and 21
  turbine/data clusters, calibrated predict-then-optimize (PTO) is the best
  feasible learned policy at 89.0 k EUR/episode (clustered 95% interval
  78.1-100.0). DFL costs 146.6 and does not beat the fair PTO baseline.

All economic values are controlled synthetic quantities, not operator-calibrated
cost estimates.

## Repository structure

```text
.
|-- synth_data.py                 synthetic SCADA and failure generator
|-- stage1_anomaly.py             normal behaviour and anomaly detection
|-- stage2_imputation.py          reusable imputation methods
|-- stage3_rul.py                 daily features and failure prediction
|-- stage4_dfl.py                 policy learning and cost simulation
|-- care_benchmark.py             canonical multi-farm CARE benchmark
|-- real_data_care.py             preliminary CARE validation entry point
|-- run_all.py                    end-to-end synthetic smoke test
|-- experiments/                  reproducible analyses and ablations
|-- results/                      dated, report-ready evidence
|-- tests/                        regression tests
`-- RESEARCH_PLAN.md              work packages and report outline
```

The large CARE dataset belongs in `data/CARE_To_Compare/`; it and the local
`.python/` runtime are intentionally excluded from Git.

## Setup and verification

Python 3.12 is recommended.

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

If the checked-out project-local runtime is available, replace
`.\.venv\Scripts\python.exe` with `.\.python\python.exe`.

## Reproduce the main analyses

```powershell
# End-to-end synthetic smoke test
& '.\.python\python.exe' run_all.py

# Full CARE benchmark
& '.\.python\python.exe' care_benchmark.py data\CARE_To_Compare `
  --output results\care_benchmark\full_run

# Representation analysis with clustered intervals
& '.\.python\python.exe' -m experiments.care_results_analysis `
  results\care_benchmark\full_run

# Missingness-to-detection study
& '.\.python\python.exe' -m experiments.missingness_detection_study `
  --output results\missingness_reproduction

# Corrected multi-seed cost-aware benchmark
& '.\.python\python.exe' -m experiments.cost_aware_benchmark `
  --output results\cost_aware_reproduction
```

Every final result directory contains tables, metadata, figures, and a concise
`FINDINGS.md`. The work-package rationale and report outline are documented in
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

## Interpretation rules

1. CARE validates anomaly detection only; it does not validate real RUL or cost.
2. Synthetic costs demonstrate controlled decision consequences only.
3. Use all events and normal controls, prevent temporal leakage, and preserve
   asset dependence in uncertainty estimates.
4. Treat post-hoc operating points as exploratory and compare DFL against a
   fairly tuned baseline.
5. Do not cite synthetic artifacts produced before the corrections described
   in `results/CORRECTED_BASELINE_2026-07-23.md`.

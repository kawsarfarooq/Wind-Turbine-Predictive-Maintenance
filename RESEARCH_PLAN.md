# Extended research plan

## Project question

How robust are residual-based anomaly detectors across wind farms, operating
conditions, fault types, and missing-data conditions, and how do their errors
affect maintenance cost in a controlled synthetic environment?

## Research questions

1. Does normal-behaviour residualization improve anomaly-vs-normal separation?
2. Does the answer generalize across CARE Farms A, B, and C and fault types?
3. How do missingness and imputation affect detection rather than only RMSE?
4. How do detection and prediction errors propagate into maintenance cost?

## Work packages

### WP1 — Correctness and reproducibility: complete

- Correct right-censored synthetic episodes.
- Prevent forced/negative high-noise degradation ramps.
- Reset rolling features at episode boundaries.
- Make simulator seeds produce independent noise replicates.
- Add a project-local Python 3.12 runtime, dependency specification, and tests.

### WP2 — Multi-farm CARE benchmark: complete

- Evaluate all 95 local CARE datasets from 36 assets.
- Compare raw and linear-residual representations.
- Compare PCA-GMM and PCA-Isolation-Forest detectors.
- Use normal-event controls and training-relative percentile scores.
- Report event ROC-AUC, PR-AUC, detection, false alarms, and separation gaps.
- Add asset-clustered bootstrap confidence intervals.

Main result: residual-GMM is strongest on Farm A, no tested method separates
Farm B, and raw GMM is strongest on Farm C. Residualization is farm-dependent.

### WP3 — Residual and detector ablations: complete

- Compare linear residuals with a regularized quadratic normal-behaviour model: complete.
- Compare causal smoothing windows and training-calibrated threshold quantiles: complete.
- Produce fault-type and asset-level interpretations: complete.
- Retain linear-GMM for Farm A and raw-GMM for Farm C downstream: complete.

Nonlinear-ablation result: quadratic residuals do not improve on the strongest
existing representation. Linear residual-GMM remains strongest on Farm A,
raw GMM remains strongest on Farm C, and Farm B remains unresolved. This
negative result narrows the next experiments to temporal smoothing, threshold
calibration, and subgroup analysis rather than adding still more model capacity.

Temporal-ablation result: 10-minute/1-hour scores have the highest descriptive
ROC-AUC on Farms A and C, but permissive event alarms create many false alarms.
Under an exploratory 20% normal-event false-alarm constraint, 1-hour smoothing
with q=0.999 gives 50.0% detection and 0.0% false alarms on Farm A; 72-hour
smoothing with q=0.990 gives 44.4% detection and 9.7% false alarms on Farm C.
Farm B remains statistically indistinguishable from chance, so its apparent
6-hour operating point must not be treated as validated. This motivated the
fault-type and asset-level analysis that completes WP3 below.

Subgroup result: Farm A performance is supported by hydraulic and drivetrain
events. Farm C is supported mainly by hydraulic and pitch/blade/hub events and
is stable to removing any one asset. Farm B combines weakly positive drivetrain
events with below-control transformer events and remains unstable around chance.
Categories with fewer than three anomaly events or fewer than two anomaly assets
are retained as descriptive evidence only.

### WP4 — Missingness-to-detection study: complete

- Inject random, block, and sensor-dropout missingness.
- Compare forward fill, linear, and median imputation under paired masks.
- Measure imputation RMSE, event discrimination, detection, and false alarms.

Result: linear interpolation has the lowest masked-value RMSE and highest
downstream ROC-AUC in every corrupted setting. Under 30% random missingness its
ROC-AUC is 0.962 and the healthy-day false-alarm rate is 63.4%; under 30%
sensor dropout its ROC-AUC is 0.824 and false alarms are 39.7%, compared with
0.980 and 11.0% on clean data. Detection saturates at 100%, showing why ROC-AUC
and false alarms must accompany event detection. This is a controlled synthetic
study; GP imputation was excluded from the final matrix because the long series
would make the comparison computationally disproportionate.

### WP5 — Cost-aware controlled study: complete

- Regenerate corrected multi-seed synthetic benchmarks.
- Evaluate calibrated PTO, DFL, probabilistic optimal stopping, and an oracle.
- Report cost, failure rate, wasted life, regret, and turbine-clustered intervals.

Result: 62 held-out failure episodes across three independent data seeds and
21 turbine/data clusters were evaluated. The oracle lower bound is 50.5 k EUR
per episode. Fairly calibrated PTO is the best feasible learned policy at
89.0 (clustered 95% interval 78.1-100.0), ahead of DFL at 146.6 and a calibrated
residual threshold at 147.4. Reactive maintenance costs 300.0 and fails in
every episode. DFL therefore does not beat the strong fair baseline.

## Report structure

1. Introduction and industrial motivation
2. CARE and synthetic data
3. Normal-behaviour and anomaly-detection methods
4. Evaluation and statistical protocol
5. Multi-farm CARE results
6. Robustness and missingness results
7. Controlled cost-aware maintenance study
8. Discussion, limitations, and conclusion

Every reported experiment must retain its event table, summary, configuration,
software versions, figures, and interpretation under a dated results folder.

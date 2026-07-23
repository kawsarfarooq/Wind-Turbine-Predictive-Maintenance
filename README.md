# Wind Turbine Predictive Maintenance

> **Result status (2026-07-23):** Synthetic censoring, high-noise
> degradation, episode-boundary features, and seed independence were corrected.
> Synthetic CSV/PNG outputs produced before this date are retained as legacy
> artifacts and must not be cited as final results. See
> `results/CORRECTED_BASELINE_2026-07-23.md` for the first corrected smoke test.

End-to-end pipeline mapping to the AI in Industry lecture blocks:

| File | Stage | Lecture blocks |
|---|---|---|
| `synth_data.py` | Synthetic SCADA farm with degradation + failure logs | (use case) |
| `stage1_anomaly.py` | KDE + sliding window, GMM, autoencoder, threshold calibration | 1, 2 |
| `stage2_imputation.py` | ffill / linear / Gaussian Process imputation study | 3 |
| `stage3_rul.py` | "failure within N days" classifier on daily residual features | 4 |
| `stage4_dfl.py` | Cost model, REINFORCE-trained maintenance policy, policy simulation | 8 |
| `run_all.py` | Driver: train on turbines 0–2, evaluate on 3–4 | — |

## Run

```bash
pip install numpy pandas scikit-learn
python run_all.py            # ~1-2 min
```

This workspace also contains a self-contained runtime. On PowerShell:

```powershell
& '.\.python\python.exe' -m pytest -q -p no:cacheprovider
& '.\.python\python.exe' run_all.py
```

Expected output: KDE threshold, GMM/AE healthy-vs-pre-failure score gap,
imputation RMSEs, classifier training summary, and the policy cost table.

## Extended CARE benchmark

`care_benchmark.py` is the main real-data evaluation. It processes all 95 CARE
datasets across Farms A, B, and C, compares raw and residual representations,
compares PCA-GMM with PCA-Isolation-Forest, and evaluates anomaly events against
normal-event controls using comparable training-percentile scores.

```powershell
& '.\.python\python.exe' care_benchmark.py data\CARE_To_Compare `
  --output results\care_benchmark\full_run
& '.\.python\python.exe' care_results_analysis.py `
  results\care_benchmark\full_run
```

The verified run in `results/care_benchmark/full_2026-07-23/` found:

- Farm A: residual-GMM ROC-AUC 0.858 (asset-bootstrap CI 0.639–1.000).
- Farm B: no tested method separates anomaly from normal controls.
- Farm C: raw GMM ROC-AUC 0.740 (CI 0.594–0.848).
- Across all farms, raw GMM ROC-AUC is 0.624 and residual GMM is 0.608.

The key finding is that residual condition normalization is farm-dependent: it
strongly helps Farm A, does not rescue Farm B, and is inferior to raw GMM on
Farm C. See `RESEARCH_PLAN.md` for the remaining extended study.

## Experiment 1: cost-ratio sensitivity (`sweep_cost_ratio.py`)

Sweeps C_fail over {100,300,600,1000} with C_prev=50. Two honest findings
you should discuss in the report rather than hide:

1. **A fairly tuned predict-then-optimize is a strong baseline.** When the
   classifier threshold is re-tuned on training episodes for each cost
   setting (the fair version of PTO), it picks thr=0.95 and reaches ~66.5
   on the test episodes at every ratio -- on this synthetic data the
   degradation signal is clean and monotone, so a well-tuned threshold
   policy is near-optimal. DFL matches it at low asymmetry (~70 at 2x) but
   becomes conservative and higher-variance at large ratios (~115-135 at
   12-20x, std up to +/-45 across seeds).
2. **Score-function gradients are scale- and variance-sensitive.** The
   naive single-sample REINFORCE diverged when C_fail grew; per-batch
   advantage normalization (batch=16) was required for stable training.
   This is exactly the variance issue Block 8 motivates importance
   sampling for -- implementing IS gradient reuse and showing it shrinks
   the +/-45 seed variance is a natural extension.

Implication for real data: DFL's edge should appear when the signal is
noisier / less monotone than this generator (where simple thresholds stop
being near-optimal) -- a testable hypothesis for the EDP dataset, and you
can also make the synthetic signal noisier to probe it.

## Experiment 2: signal-quality sweep (`sweep_noise.py`)

**Main finding of the project (after the enlarged-test-set analysis):**
a fairly cost-tuned predict-then-optimize policy is robust across signal
quality. The apparent crossover to DFL seen with 10 test episodes did NOT
survive enlargement to 66 paired episodes (`enlarged_paired_analysis.py`,
training unchanged, 12 held-out turbines): at noise 0 the means are tied
(PTO 71.7 vs DFL 73.3), at noise 1.0 PTO is significantly better (96.0
vs 132.0, Wilcoxon p=0.006), and at noise 1.5 DFL leads by only +9.1 k
EUR (p=0.11, not significant). The project's central lesson is therefore
methodological: fair baselines and adequate test power reverse the naive
conclusion that DFL wins.

Supporting report assets:
- `episode_trace.py` -> `episode_trace.png`: the same failure observed at
  noise 0.0 vs 1.5 (report Fig. 1) -- shows the clean monotone ramp
  disappearing under noise.
- `paired_differences.py` -> `paired_episode_differences.csv`: paired
  per-episode PTO-vs-DFL costs at noise {0, 1.0, 1.5} on the ORIGINAL 10
  test episodes (kept for the small-vs-large sample comparison).
- `enlarged_paired_analysis.py` -> `enlarged_paired_differences.csv` +
  `enlarged_paired_summary.csv`: the same paired analysis on 66 held-out
  episodes (12 test turbines; turbines 0-4 verified bit-identical to all
  earlier experiments), with Wilcoxon signed-rank tests. This is the
  statistically defensible version -- see the main finding above.


Tests the hypothesis from Experiment 1 directly. `generate_farm()` now
takes `noise_level` (0 = original clean data), which degrades
observability three ways -- stronger sensor noise, benign false
temperature spikes, and weak-warning failures (shortened/attenuated
degradation ramps) -- while keeping the SAME 20 failures at the SAME
times at every level (noise uses a separate RNG stream), so the
comparison is paired.

Result (C_prev=50; mean cost per test episode):

| noise | PTO (tuned) | DFL @C_fail=300 | DFL @C_fail=1000 |
|---|---|---|---|
| 0.0 | 66.5 | 86.8 | 133.0 |
| 0.5 | 82.2 | 104.9 | 116.2 |
| 1.0 | 120.0 | **115.3** | **114.5** |
| 1.5 | 163.4 / 169.3 | **146.8** | **153.7** |

The hypothesis holds: tuned PTO dominates on clean signals, the gap
narrows as observability degrades, and the ranking flips around
noise=1.0, with DFL ~10% cheaper at noise=1.5 (where PTO's tuned
threshold also starts missing failures at C_fail=300). Caveats for the
report: only 10 test episodes (report the paired per-episode differences,
not just means) and DFL seed variance is non-trivial in some cells.

## Key ideas per stage

- **Stage 1.** Normal-behaviour model (bearing temp ~ power + ambient) →
  residuals → KDE anomaly score with sliding window (non-stationarity);
  threshold = 99.5% quantile of healthy scores. GMM/autoencoder score the
  full multivariate feature vector (Block 2).
- **Stage 2.** Artificial gap masking → compare imputers by RMSE.
  Extension worth doing: rerun Stage 1 on imputed data and measure the
  *downstream* effect on false/missed alarms.
- **Stage 3.** Classification policy (failure within 14 days). Hyperparams
  are exposed — plug into surrogate-based Bayesian optimisation
  (e.g. `skopt.gp_minimize`) for the Block 4 tuning component.
- **Stage 4.** Explicit cost model (C_fail=300, C_prev=50, C_waste=1 /day).
  Stochastic linear policy trained with score-function gradients + moving
  baseline. Deployed deterministically (act when p>0.5). Compare against
  reactive / threshold / predict-then-optimize in an episode replay.

## Real-data validation: CARE dataset (`real_data_care.py`)

CARE ("CARE to Compare", Zenodo record 15846963, CC BY-SA 4.0) is the
selected real-world dataset: 95 SCADA datasets across 3 wind farms with
labeled anomaly events, per-timestamp status labels, and a train/
prediction split. Division of labor in this project:
- **Synthetic data** -> controlled cost-policy simulation and the
  PTO-vs-DFL experiments (paired design, known ground truth, enough
  failures). This remains the core of the project.
- **CARE (Wind Farm A, Avg columns only)** -> preliminary real-data
  validation of the anomaly-detection stage ONLY. CARE's labels support
  detection validation, not the cost simulation; no RUL/DFL is attempted
  on it.

The CARE validation is BATCH-BASED across all of Wind Farm A (22
datasets: 12 labeled anomalies + 10 normals), not a hand-picked event.
For every dataset, `real_data_care.py --all` trains a scaler->PCA->GMM
normal-behaviour detector on that dataset's own training year (status-0
rows, ONLY Avg sensor columns -- the dataset's Known Data Issues flag
Min/Max/Std as unreliable), then reports per event: score elevation over
training, threshold-free AUC (event-window vs own training scores), and
the fraction of event points above the 99.5% training threshold; normal
datasets provide the false-alarm control. Outputs:
`care_farmA_batch_summary.csv`, `care_farmA_batch_by_type.csv`
(per-fault-type aggregation), `care_farmA_batch_overview.png`.

The unzipped CARE dataset is stored under `data/CARE_To_Compare`:

```powershell
python real_data_care.py ".\data\CARE_To_Compare" --all
python real_data_care.py ".\data\CARE_To_Compare" --all --residual
python real_data_care_farmB.py ".\data\CARE_To_Compare"
```

Single-event mode (score-over-time plot for one anomaly + one normal)
still works without --all, with --anomaly-event / --normal-event to pick.
Batch result on real data (22/22 processed). RAW Avg features: mean
anomaly AUC 0.473 (chance), 2/12 alarms, 0/10 false alarms. RESIDUAL
mode (--residual, faithful to synthetic Stage 1): mean anomaly AUC
0.608, 12/12 anomalies above chance, 6/12 alarms, but 2/10 false alarms.
Condition-normalisation is the transferable ingredient; gearbox events
stay near chance either way.

## Moving to real data

Replace `generate_farm()` with a loader for the EDP Open Data wind farm
set (SCADA + failure logs), keeping columns
`turbine, t, wind, ambient, power, bearing_temp, episode, rul_h, failed_now`.
Caveats to discuss in the report: few real failure events (consider
leave-one-failure-out evaluation), label noise in maintenance logs, and
keeping the turbine-level train/test split to avoid leakage.

## Suggested experiments for the report

1. GMM vs autoencoder detection lead time (hours before failure).
2. Imputation quality → downstream detection impact.
3. Cost-ratio sensitivity: sweep C_fail/C_prev ∈ {2, 6, 10, 20} and show
   how the DFL advantage over predict-then-optimize grows with asymmetry.
4. Importance sampling for gradient reuse in Stage 4 (Block 8 extension).

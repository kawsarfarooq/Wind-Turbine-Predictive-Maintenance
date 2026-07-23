# Corrected end-to-end baseline — 2026-07-23

Command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '.\.python\python.exe' run_all.py
```

Environment:

- Python 3.12.10
- NumPy 2.5.1
- pandas 3.0.5
- scikit-learn 1.9.0
- SciPy 1.18.0
- Matplotlib 3.11.1

Regression suite: `6 passed in 7.30s`.

## Output

| Stage | Result |
|---|---|
| Data | 87,600 rows, 20 observed failures |
| KDE threshold | 3.90 |
| GMM score | healthy 0.52; pre-failure 49.92 |
| Autoencoder score | healthy 0.33; pre-failure 0.44 |
| Imputation RMSE | ffill 5.093; linear 4.183; GP 3.889 |
| Classifier training labels | 169 positive days |
| Held-out policy episodes | 8 observed failures |
| Reactive cost | 300.0 k EUR/episode; 0% avoided |
| Residual-threshold cost | 149.8 k EUR/episode; 100% avoided |
| Untuned PTO cost | 81.7 k EUR/episode; 100% avoided |
| DFL cost | 54.0 k EUR/episode; 100% avoided |

These policy values are a smoke-test baseline, not the final comparison. PTO
still uses the untuned 0.5 threshold and DFL uses one training seed. The fair,
multi-seed cost benchmark must be regenerated before drawing conclusions.

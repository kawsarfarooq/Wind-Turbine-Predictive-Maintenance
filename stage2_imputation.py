"""Stage 2 -- missing-value imputation (Lecture Block 3).

Mask artificial gaps in bearing_temp, impute with:
  * forward fill
  * linear interpolation
  * Gaussian Process on time (local window around each gap)
Report imputation RMSE. (Extension: rerun Stage 1 on imputed data to
measure downstream detection impact.)
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def mask_gaps(series, n_gaps=20, gap_len=12, seed=0):
    rng = np.random.default_rng(seed)
    y = series.copy().astype(float)
    truth = []
    for _ in range(n_gaps):
        s = rng.integers(gap_len, len(y) - gap_len)
        truth.append((s, series[s:s + gap_len].copy()))
        y[s:s + gap_len] = np.nan
    return y, truth


def impute_ffill(y):
    return pd.Series(y).ffill().bfill().values


def impute_linear(y):
    return pd.Series(y).interpolate("linear").ffill().bfill().values


def impute_gp(y, context=48):
    """GP over time using `context` observed hours on each side of a gap."""
    out = y.copy()
    isnan = np.isnan(y)
    i = 0
    while i < len(y):
        if isnan[i]:
            j = i
            while j < len(y) and isnan[j]:
                j += 1
            lo, hi = max(0, i - context), min(len(y), j + context)
            t = np.arange(lo, hi)
            obs = ~np.isnan(y[lo:hi])
            k = 1.0 * RBF(length_scale=8.0) + WhiteKernel(noise_level=0.5)
            gp = GaussianProcessRegressor(kernel=k, normalize_y=True)
            gp.fit(t[obs].reshape(-1, 1), y[lo:hi][obs])
            out[i:j] = gp.predict(np.arange(i, j).reshape(-1, 1))
            i = j
        else:
            i += 1
    return out


def rmse_on_gaps(imputed, truth):
    errs = [imputed[s:s + len(v)] - v for s, v in truth]
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))


def run_imputation_study(series, seed=0):
    y, truth = mask_gaps(series, seed=seed)
    return {
        "ffill": rmse_on_gaps(impute_ffill(y), truth),
        "linear": rmse_on_gaps(impute_linear(y), truth),
        "gp": rmse_on_gaps(impute_gp(y), truth),
    }

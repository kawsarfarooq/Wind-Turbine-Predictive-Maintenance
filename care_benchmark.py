"""Unified CARE anomaly benchmark for Wind Farms A, B, and C.

The benchmark is deliberately event-centric. Every CARE dataset is trained on
its own normal training window, then evaluated on its labeled anomaly window or
normal prediction window. Scores are converted to empirical percentiles of the
training distribution so event-level results are comparable across datasets.

Main comparisons:
  * raw Avg sensor features vs linear normal-behaviour residuals;
  * PCA-GMM vs PCA-Isolation-Forest anomaly detectors;
  * status 0 only vs CARE's documented normal states {0, 2};
  * anomaly events vs normal-event controls in every wind farm.

Example:
    python care_benchmark.py data/CARE_To_Compare --quick 1
    python care_benchmark.py data/CARE_To_Compare
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from real_data_care import (avg_sensor_columns, find_dataset_file, sniff_sep,
                            split_covariates_targets)


DEFAULT_ROOT = Path("data/CARE_To_Compare")
DEFAULT_OUTPUT = Path("results/care_benchmark")
SMOOTH_STEPS = 144                 # trailing 24 h at 10-minute resolution
THRESHOLD_QUANTILE = 0.995
MAX_TRAIN_ROWS = 20_000
RANDOM_SEED = 0


def discover_farms(root: Path) -> dict[str, tuple[Path, pd.DataFrame, dict]]:
    farms = {}
    for info_path in sorted(root.rglob("event_info*.csv")):
        farm_dir = info_path.parent
        farm = farm_dir.name.strip().split()[-1].upper()
        events = pd.read_csv(info_path, sep=sniff_sep(info_path))
        events.columns = [c.strip().lower() for c in events.columns]
        csvs = {p.stem: p for p in farm_dir.rglob("*.csv")
                if "event_info" not in p.name
                and "feature_description" not in p.name}
        farms[farm] = (farm_dir, events, csvs)
    return farms


def load_avg_dataset(path: Path):
    """Read metadata and average-sensor columns without loading Min/Max/Std.

    Farm C has up to 957 columns. Selecting columns at CSV parse time reduces
    both I/O and peak memory while preserving the benchmark's Avg-only design.
    """
    sep = sniff_sep(path)
    original = list(pd.read_csv(path, sep=sep, nrows=0).columns)
    lower = {c: c.strip().lower() for c in original}
    meta = {}
    for key, needles in [("time", ["time_stamp", "timestamp", "time"]),
                         ("train", ["train_test", "train"]),
                         ("status", ["status_type_id", "status"]),
                         ("asset", ["asset_id"]), ("rowid", ["id"])]:
        for c in original:
            name = lower[c]
            if any(name == needle or name.startswith(needle)
                   for needle in needles):
                meta[key] = name
                break
    missing = {"time", "train", "status"} - set(meta)
    if missing:
        raise ValueError(f"{path} lacks metadata columns {missing}")
    bad = ("min", "max", "std")
    usecols = []
    for c in original:
        name = lower[c]
        if name in meta.values():
            usecols.append(c)
            continue
        if any(part in name.split("_") for part in bad) or name.endswith(bad):
            continue
        usecols.append(c)
    df = pd.read_csv(path, sep=sep, usecols=usecols)
    df.columns = [c.strip().lower() for c in df.columns]
    df[meta["time"]] = pd.to_datetime(df[meta["time"]], errors="coerce")
    return df.sort_values(meta["time"]).reset_index(drop=True), meta


def _subsample(X: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), max_rows, replace=False)]


class PCAGMM:
    def __init__(self, max_pca=10, n_components=4, seed=RANDOM_SEED):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=max_pca, random_state=seed)
        self.model = GaussianMixture(
            n_components=n_components, covariance_type="full",
            reg_covar=1e-4, random_state=seed)

    def fit(self, X):
        X_fit = _subsample(X, MAX_TRAIN_ROWS, RANDOM_SEED)
        scaled = self.scaler.fit_transform(X_fit)
        n_pca = min(self.pca.n_components, scaled.shape[1], len(scaled))
        self.pca.set_params(n_components=n_pca)
        self.model.fit(self.pca.fit_transform(scaled))
        return self

    def score(self, X):
        z = self.pca.transform(self.scaler.transform(X))
        return -self.model.score_samples(z)


class PCAIsolationForest:
    def __init__(self, max_pca=10, seed=RANDOM_SEED):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=max_pca, random_state=seed)
        self.model = IsolationForest(
            n_estimators=200, contamination="auto", n_jobs=1,
            random_state=seed)

    def fit(self, X):
        X_fit = _subsample(X, MAX_TRAIN_ROWS, RANDOM_SEED)
        scaled = self.scaler.fit_transform(X_fit)
        n_pca = min(self.pca.n_components, scaled.shape[1], len(scaled))
        self.pca.set_params(n_components=n_pca)
        self.model.fit(self.pca.fit_transform(scaled))
        return self

    def score(self, X):
        z = self.pca.transform(self.scaler.transform(X))
        return -self.model.score_samples(z)


DETECTORS = {"gmm": PCAGMM, "iforest": PCAIsolationForest}


def prepare_frame(df: pd.DataFrame, cols: list[str], train_mask: np.ndarray):
    """Causal numeric preparation using past values and train medians.

    Unlike bidirectional interpolation, forward fill does not use future sensor
    readings to construct an earlier anomaly score. Leading gaps and fully
    missing past values fall back to medians from normal training rows only.
    """
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan)
    train = X.loc[train_mask]
    keep = [c for c in cols if train[c].notna().any()
            and float(train[c].std()) > 1e-9]
    X = X[keep]
    medians = X.loc[train_mask].median()
    X = X.ffill().fillna(medians)
    keep = [c for c in keep if X[c].notna().all()]
    return X[keep], keep


def build_representation(df, meta, train_mask, representation):
    sensor_cols = avg_sensor_columns(df, meta)
    if representation == "raw":
        frame, keep = prepare_frame(df, sensor_cols, train_mask)
        return frame.values, "raw", 0, len(keep)

    covariates, targets = split_covariates_targets(sensor_cols)
    if not covariates or not targets:
        raise ValueError("residual representation: no covariates/targets found")
    cov, keep_cov = prepare_frame(df, covariates, train_mask)
    tgt, keep_tgt = prepare_frame(df, targets, train_mask)
    if not keep_cov or not keep_tgt:
        raise ValueError("residual representation: degenerate covariates/targets")
    reg = LinearRegression().fit(
        cov.values[train_mask], tgt.values[train_mask])
    residual = tgt.values - reg.predict(cov.values)
    return residual, "linear_residual", len(keep_cov), len(keep_tgt)


def empirical_percentile(train_scores, scores):
    ordered = np.sort(np.asarray(train_scores, dtype=float))
    return np.searchsorted(ordered, scores, side="right") / len(ordered)


def event_mask(df, meta, row):
    pred = (df[meta["train"]].astype(str).str.lower() != "train").values
    if row["event_label"] != "anomaly":
        return pred
    start = pd.to_datetime(row.get("event_start"), errors="coerce")
    end = pd.to_datetime(row.get("event_end"), errors="coerce")
    if pd.isna(start):
        return pred
    mask = np.asarray(df[meta["time"]] >= start)
    if pd.notna(end):
        mask = mask & np.asarray(df[meta["time"]] <= end)
    return mask


def normal_train_mask(df, meta, normal_statuses):
    status = pd.to_numeric(df[meta["status"]], errors="coerce")
    return np.asarray(
        (df[meta["train"]].astype(str).str.lower() == "train")
        & status.isin(normal_statuses))


def score_loaded_event(df, meta, row, farm, X, mode, n_cov, n_targets,
                       train_mask, detector_name, normal_statuses):
    if train_mask.sum() < 1000:
        raise ValueError(f"only {int(train_mask.sum())} normal training rows")
    detector = DETECTORS[detector_name]().fit(X[train_mask])
    raw_score = detector.score(X)
    percentile = empirical_percentile(raw_score[train_mask], raw_score)
    smooth = pd.Series(percentile).rolling(
        SMOOTH_STEPS, min_periods=1).median().to_numpy()
    threshold = float(np.quantile(smooth[train_mask], THRESHOLD_QUANTILE))
    mask = event_mask(df, meta, row)
    if not mask.any():
        raise ValueError("empty evaluation window")

    train_score, eval_score = smooth[train_mask], smooth[mask]
    auc_y = np.r_[np.zeros(len(train_score)), np.ones(len(eval_score))]
    auc_s = np.r_[train_score, eval_score]
    alarms = eval_score > threshold
    asset = row.get("asset", row.get("asset_id", "unknown"))
    return {
        "farm": farm,
        "asset": asset,
        "cluster_id": f"{farm}:{asset}",
        "event_id": row["event_id"],
        "label": row["event_label"],
        "description": row.get("event_description", ""),
        "representation": mode,
        "detector": detector_name,
        "normal_statuses": "+".join(map(str, sorted(normal_statuses))),
        "n_train": int(train_mask.sum()),
        "n_eval": int(mask.sum()),
        "n_covariates": n_cov,
        "n_targets": n_targets,
        "threshold_percentile": threshold,
        "auc_vs_train": float(roc_auc_score(auc_y, auc_s)),
        "event_mean_percentile": float(eval_score.mean()),
        "event_p95_percentile": float(np.quantile(eval_score, 0.95)),
        "frac_above_threshold": float(alarms.mean()),
        "alarm": bool(alarms.any()),
    }


def analyse_event(path, row, farm, representation, detector_name,
                  normal_statuses):
    """Convenience wrapper for one configuration of one event."""
    df, meta = load_avg_dataset(path)
    train_mask = normal_train_mask(df, meta, normal_statuses)
    X, mode, n_cov, n_targets = build_representation(
        df, meta, train_mask, representation)
    return score_loaded_event(
        df, meta, row, farm, X, mode, n_cov, n_targets, train_mask,
        detector_name, normal_statuses)


def cluster_bootstrap(group, n_boot=1000, seed=RANDOM_SEED):
    """Asset-clustered intervals for event ROC-AUC and separation gap."""
    rng = np.random.default_rng(seed)
    clusters = group["cluster_id"].dropna().unique()
    aucs, gaps = [], []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat(
            [group[group["cluster_id"] == cluster] for cluster in sampled],
            ignore_index=True)
        y = (boot["label"] == "anomaly").astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        score = boot["event_mean_percentile"].to_numpy()
        anomaly = score[y == 1]
        normal = score[y == 0]
        aucs.append(roc_auc_score(y, score))
        gaps.append(float(anomaly.mean() - normal.mean()))
    if not aucs:
        return (np.nan,) * 4
    return (
        float(np.quantile(aucs, 0.025)),
        float(np.quantile(aucs, 0.975)),
        float(np.quantile(gaps, 0.025)),
        float(np.quantile(gaps, 0.975)),
    )


def _classification_summary(group):
    y = (group["label"] == "anomaly").astype(int).to_numpy()
    score = group["event_mean_percentile"].to_numpy()
    anomaly = group[group["label"] == "anomaly"]
    normal = group[group["label"] == "normal"]
    auc_low, auc_high, gap_low, gap_high = cluster_bootstrap(group)
    return pd.Series({
        "n_events": len(group),
        "n_anomaly": len(anomaly),
        "n_normal": len(normal),
        "event_roc_auc": roc_auc_score(y, score)
            if len(np.unique(y)) == 2 else np.nan,
        "event_pr_auc": average_precision_score(y, score)
            if len(np.unique(y)) == 2 else np.nan,
        "event_roc_auc_ci_low": auc_low,
        "event_roc_auc_ci_high": auc_high,
        "mean_anomaly_percentile": anomaly["event_mean_percentile"].mean(),
        "mean_normal_percentile": normal["event_mean_percentile"].mean(),
        "anomaly_normal_gap": (
            anomaly["event_mean_percentile"].mean()
            - normal["event_mean_percentile"].mean()),
        "anomaly_normal_gap_ci_low": gap_low,
        "anomaly_normal_gap_ci_high": gap_high,
        "detection_rate": anomaly["alarm"].mean(),
        "normal_false_alarm_rate": normal["alarm"].mean(),
    })


def summarize_events(events):
    if "cluster_id" not in events:
        events = events.copy()
        events["cluster_id"] = (
            events["farm"].astype(str) + ":" + events["asset"].astype(str))
    keys = ["farm", "representation", "detector", "normal_statuses"]
    per_farm = events.groupby(keys, dropna=False).apply(
        _classification_summary, include_groups=False).reset_index()
    all_farms = events.assign(farm="ALL").groupby(keys, dropna=False).apply(
        _classification_summary, include_groups=False).reset_index()
    return pd.concat([per_farm, all_farms], ignore_index=True)


def metadata(args):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "farms": args.farms,
        "representations": args.representations,
        "detectors": args.detectors,
        "normal_status_sets": [
            "+".join(map(str, sorted(values)))
            for values in args.normal_status_sets],
        "smooth_steps": SMOOTH_STEPS,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "max_train_rows": MAX_TRAIN_ROWS,
        "quick": args.quick,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--farms", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--representations", nargs="+",
                        choices=["raw", "residual"],
                        default=["raw", "residual"])
    parser.add_argument("--detectors", nargs="+", choices=DETECTORS,
                        default=["gmm", "iforest"])
    parser.add_argument("--normal-status-sets", nargs="+",
                        default=["0", "0+2"],
                        help="sets such as 0 or 0+2")
    parser.add_argument("--quick", type=int, default=0,
                        help="process at most N events per farm (smoke test)")
    args = parser.parse_args()

    args.farms = [f.upper() for f in args.farms]
    args.normal_status_sets = [
        {int(v) for v in item.split("+")} for item in args.normal_status_sets]
    farms = discover_farms(args.root)
    rows = []
    for farm in args.farms:
        if farm not in farms:
            print(f"Farm {farm}: not found; skipping")
            continue
        _, event_info, csvs = farms[farm]
        selected = event_info
        if args.quick:
            # Preserve both classes where possible in smoke tests.
            selected = pd.concat([
                event_info[event_info.event_label == label].head(args.quick)
                for label in ["anomaly", "normal"]
            ])
        for _, event in selected.iterrows():
            path = find_dataset_file(csvs, event["event_id"])
            if path is None:
                print(f"Farm {farm} event {event['event_id']}: dataset not found")
                continue
            try:
                df, meta = load_avg_dataset(path)
            except Exception as exc:
                print(f"Farm {farm} event {event['event_id']}: LOAD ERROR {exc}")
                continue
            mask_groups = {}
            for statuses in args.normal_status_sets:
                mask = normal_train_mask(df, meta, statuses)
                key = np.packbits(mask).tobytes()
                mask_groups.setdefault(
                    key, {"mask": mask, "status_sets": []})["status_sets"].append(
                        statuses)
            for mask_group in mask_groups.values():
                train_mask = mask_group["mask"]
                status_sets = mask_group["status_sets"]
                for representation in args.representations:
                    try:
                        X, mode, n_cov, n_targets = build_representation(
                            df, meta, train_mask, representation)
                    except Exception as exc:
                        for detector in args.detectors:
                            for statuses in status_sets:
                                rows.append({
                                    "farm": farm,
                                    "event_id": event["event_id"],
                                    "label": event["event_label"],
                                    "representation": representation,
                                    "detector": detector,
                                    "normal_statuses": "+".join(
                                        map(str, sorted(statuses))),
                                    "error": str(exc)[:300],
                                })
                        print(f"Farm {farm} event {event['event_id']} "
                              f"{representation}/"
                              f"{[sorted(s) for s in status_sets]}: "
                              f"REPRESENTATION ERROR {exc}")
                        continue
                    for detector in args.detectors:
                        primary_statuses = status_sets[0]
                        prefix = (f"Farm {farm} event {event['event_id']} "
                                  f"{representation}/{detector}/"
                                  f"{[sorted(s) for s in status_sets]}")
                        try:
                            result = score_loaded_event(
                                df, meta, event, farm, X, mode, n_cov,
                                n_targets, train_mask, detector,
                                primary_statuses)
                            for statuses in status_sets:
                                cloned = result.copy()
                                cloned["normal_statuses"] = "+".join(
                                    map(str, sorted(statuses)))
                                rows.append(cloned)
                            print(f"{prefix}: mean={result['event_mean_percentile']:.3f} "
                                  f"alarm={int(result['alarm'])}")
                        except Exception as exc:
                            for statuses in status_sets:
                                rows.append({
                                    "farm": farm,
                                    "event_id": event["event_id"],
                                    "label": event["event_label"],
                                    "representation": representation,
                                    "detector": detector,
                                    "normal_statuses": "+".join(
                                        map(str, sorted(statuses))),
                                    "error": str(exc)[:300],
                                })
                            print(f"{prefix}: ERROR {exc}")

    args.output.mkdir(parents=True, exist_ok=True)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(args.output / "care_benchmark_events.csv", index=False)
    ok = detailed[detailed.get("error").isna()] \
        if "error" in detailed else detailed
    summary = summarize_events(ok) if len(ok) else pd.DataFrame()
    summary.to_csv(args.output / "care_benchmark_summary.csv", index=False)
    (args.output / "care_benchmark_metadata.json").write_text(
        json.dumps(metadata(args), indent=2), encoding="utf-8")
    print("\n", summary.to_string(index=False))
    print(f"\nSaved benchmark outputs to {args.output}")


if __name__ == "__main__":
    main()

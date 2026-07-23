"""Real-data validation on the CARE-to-Compare dataset (Zenodo 15846963).

Scope: PRELIMINARY anomaly-detection validation only. The synthetic
experiments remain the basis for all cost-policy / PTO-vs-DFL results;
CARE labels (event start + status IDs) support detection validation but
not the cost simulation.

What this script does:
 1. Inspects the unzipped dataset tree (prints + saves structure summary).
 2. Selects Wind Farm A (86 features, EDP-based) by default; picks one
    labeled anomaly dataset and one normal dataset from event_info.csv.
 3. Uses ONLY Avg sensor columns (dataset notes: Min/Max/Std contain
    implausible values) from rows with status_type_id == 0 (definitely
    normal) in the 'train' portion.
 4. Trains a normal-behaviour detector: StandardScaler -> PCA(<=10 comps)
    -> GMM(4); anomaly score = negative log-likelihood, smoothed with a
    1-day rolling median.
 5. Threshold = 99.5% quantile of training scores.
 6. Plots the score over time for both datasets, shading the prediction
    window and the labeled anomaly event -> real_data_anomaly_score.png
 7. Saves real_data_validation_summary.csv

Usage (PowerShell, after downloading + unzipping CARE_To_Compare.zip):
    python real_data_care.py "D:\\path\\to\\CARE_To_Compare"
    python real_data_care.py "D:\\path\\to\\CARE_To_Compare" --anomaly-event 3 --normal-event 12

The script discovers files rather than assuming exact folder names, and
prints clear errors if the layout differs from expectations.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

NORMAL_STATUS = {0}          # status_type_id values treated as "surely normal"
SMOOTH_STEPS = 144           # 1 day of 10-min data


# ---------------------------------------------------------------- discovery

def sniff_sep(path):
    with open(path, "r", errors="ignore") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","


def inspect(root: Path):
    """Walk the tree; find event_info files and dataset csvs. Returns a dict
    farm_dir -> (event_info_df, {event_id: dataset_path}). Also writes
    care_structure_summary.txt."""
    lines, farms = [], {}
    infos = sorted(root.rglob("event_info*.csv"))
    if not infos:
        sys.exit(f"ERROR: no event_info*.csv found under {root}. "
                 "Is this the unzipped CARE_To_Compare folder?")
    for info_path in infos:
        farm_dir = info_path.parent
        ev = pd.read_csv(info_path, sep=sniff_sep(info_path))
        ev.columns = [c.strip().lower() for c in ev.columns]
        csvs = {p.stem: p for p in farm_dir.rglob("*.csv")
                if "event_info" not in p.name}
        farms[farm_dir] = (ev, csvs)
        lines.append(f"farm dir: {farm_dir}")
        lines.append(f"  event_info columns: {list(ev.columns)}")
        if "event_label" in ev.columns:
            lines.append(f"  events: {len(ev)} "
                         f"(anomaly: {(ev.event_label == 'anomaly').sum()}, "
                         f"normal: {(ev.event_label == 'normal').sum()})")
        lines.append(f"  dataset csv files: {len(csvs)}")
    summary = "\n".join(lines)
    print(summary)
    Path("care_structure_summary.txt").write_text(summary)
    return farms


def pick_farm(farms, prefer="A"):
    for d in farms:
        if prefer.lower() in d.name.lower().replace("_", " ").split() \
           or d.name.lower().endswith(prefer.lower()):
            return d
    for d in farms:                       # fallback: farm with fewest columns
        if "a" in d.name.lower():
            return d
    return next(iter(farms))


def find_dataset_file(csvs, event_id):
    """Dataset files are typically named by event id (e.g. '3.csv' or
    'comma_3.csv'); match on the numeric id."""
    for stem, p in csvs.items():
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits == str(event_id):
            return p
    return None


# ---------------------------------------------------------------- pipeline

def load_dataset(path):
    df = pd.read_csv(path, sep=sniff_sep(path))
    df.columns = [c.strip().lower() for c in df.columns]
    meta = {}
    for key, needles in [("time", ["time_stamp", "timestamp", "time"]),
                         ("train", ["train_test", "train"]),
                         ("status", ["status_type_id", "status"]),
                         ("asset", ["asset_id"]), ("rowid", ["id"])]:
        for c in df.columns:
            if any(c == n or c.startswith(n) for n in needles):
                meta[key] = c
                break
    missing = {"time", "train", "status"} - set(meta)
    if missing:
        raise ValueError(
            f"{path} lacks expected metadata columns {missing}; "
            f"found columns {list(df.columns)[:10]}...")
    df[meta["time"]] = pd.to_datetime(df[meta["time"]], errors="coerce")
    return df.sort_values(meta["time"]).reset_index(drop=True), meta


def avg_sensor_columns(df, meta):
    """Avg-only sensor selection per the dataset's Known Data Issues."""
    bad = ("min", "max", "std")
    cols = []
    for c in df.columns:
        if c in meta.values():
            continue
        if any(b in c.split("_") for b in bad) or c.endswith(bad):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


class NBDetector:
    """StandardScaler -> PCA(<=10) -> GMM(4); score = -log p(x)."""

    def __init__(self, n_components=4, max_pca=10):
        self.scaler, self.n_components, self.max_pca = StandardScaler(), n_components, max_pca

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=min(self.max_pca, X.shape[1]))
        Z = self.pca.fit_transform(Xs)
        self.gmm = GaussianMixture(self.n_components, covariance_type="full",
                                   random_state=0, reg_covar=1e-4).fit(Z)
        return self

    def score(self, X):
        Z = self.pca.transform(self.scaler.transform(X))
        return -self.gmm.score_samples(Z)


def prepare_matrix(df, cols):
    X = df[cols].astype(float)
    X = X.interpolate(limit_direction="both")
    keep = [c for c in cols if X[c].std() > 1e-9 and X[c].notna().all()]
    return X[keep].values, keep


def analyse_dataset(path, event_row, detector_cols=None, detector=None):
    df, meta = load_dataset(path)
    cols = avg_sensor_columns(df, meta)
    if detector is None:
        train_mask = (df[meta["train"]].astype(str).str.lower() == "train") \
            & df[meta["status"]].isin(NORMAL_STATUS)
        if train_mask.sum() < 1000:
            sys.exit(f"ERROR: only {train_mask.sum()} normal training rows "
                     f"in {path}; check status/train_test parsing.")
        X_all, keep = prepare_matrix(df, cols)
        det = NBDetector().fit(X_all[train_mask.values])
        detector, detector_cols = det, keep
    X_all, _ = prepare_matrix(df, detector_cols)
    raw = detector.score(X_all)
    smooth = pd.Series(raw).rolling(SMOOTH_STEPS, min_periods=1).median().values
    return df, meta, raw, smooth, detector, detector_cols


def split_covariates_targets(cols):
    """Residual mode: covariates = operating-condition channels (power,
    wind speed, rotor/generator speed); targets = temperature channels if
    identifiable by name, else all remaining sensors (CARE names may be
    anonymised). Returns (covs, targets) or (None, None) if no covariates
    can be identified -> caller falls back to raw mode."""
    covs = [c for c in cols if any(k in c for k in
            ("power", "wind_speed", "rotor", "speed", "rpm"))]
    if not covs:
        return None, None
    rest = [c for c in cols if c not in covs]
    temps = [c for c in rest if "temp" in c]
    return covs, (temps if temps else rest)


def compute_residuals(df, cols, train_mask):
    """Linear normal-behaviour model per target channel (faithful to the
    synthetic Stage 1): fit targets ~ covariates on normal training rows,
    return residual matrix for ALL rows, plus a mode string."""
    from sklearn.linear_model import LinearRegression
    covs, targets = split_covariates_targets(cols)
    if covs is None:
        X_all, keep = prepare_matrix(df, cols)
        return X_all, "raw-fallback(no covariates found)", 0, len(keep)
    C = df[covs].astype(float).interpolate(limit_direction="both")
    T = df[targets].astype(float).interpolate(limit_direction="both")
    keep_c = [c for c in covs if C[c].std() > 1e-9 and C[c].notna().all()]
    keep_t = [c for c in targets if T[c].std() > 1e-9 and T[c].notna().all()]
    if not keep_c or not keep_t:
        X_all, keep = prepare_matrix(df, cols)
        return X_all, "raw-fallback(degenerate columns)", 0, len(keep)
    reg = LinearRegression().fit(C[keep_c].values[train_mask],
                                 T[keep_t].values[train_mask])
    R = T[keep_t].values - reg.predict(C[keep_c].values)
    return R, "residual", len(keep_c), len(keep_t)


def batch_analyse_one(path, row, residual=False):
    """Self-contained analysis of one dataset: train on its own training
    year (status-0, Avg columns), score everything. Returns a result dict;
    on any failure returns a dict with an 'error' field instead of raising."""
    try:
        df, meta = load_dataset(path)
        cols = avg_sensor_columns(df, meta)
        train_mask = (df[meta["train"]].astype(str).str.lower() == "train") \
            & df[meta["status"]].isin(NORMAL_STATUS)
        if train_mask.sum() < 1000:
            return {"event_id": row["_id"], "error":
                    f"only {int(train_mask.sum())} normal training rows"}
        if residual:
            X_all, mode, n_cov, n_tgt = compute_residuals(
                df, cols, train_mask.values)
        else:
            X_all, _ = prepare_matrix(df, cols)
            mode, n_cov, n_tgt = "raw", 0, X_all.shape[1]
        det = NBDetector().fit(X_all[train_mask.values])
        sm = pd.Series(det.score(X_all)).rolling(
            SMOOTH_STEPS, min_periods=1).median().values
        thr = float(np.quantile(sm[train_mask.values], 0.995))

        t = df[meta["time"]]
        pred = (df[meta["train"]].astype(str).str.lower() != "train").values
        e_start = pd.to_datetime(row.get("event_start", None), errors="coerce")
        e_end = pd.to_datetime(row.get("event_end", None), errors="coerce")
        if row["event_label"] == "anomaly" and pd.notna(e_start):
            ev_mask = np.asarray(t >= e_start)
            if pd.notna(e_end):
                ev_mask = ev_mask & np.asarray(t <= e_end)
        else:
            ev_mask = pred        # normals: evaluate the prediction window

        tr_s, ev_s = sm[train_mask.values], sm[ev_mask]
        out = {
            "event_id": row["_id"], "label": row["event_label"],
            "description": row.get("event_description", ""),
            "mode": mode, "n_covariates": n_cov, "n_targets": n_tgt,
            "n_event_points": int(ev_mask.sum()),
            "train_mean": round(float(tr_s.mean()), 2),
            "threshold": round(thr, 2),
        }
        if ev_mask.sum() == 0:
            out["error"] = "empty event window"
            return out
        out.update({
            "event_mean": round(float(ev_s.mean()), 2),
            "elevation": round(float(ev_s.mean() - tr_s.mean()), 2),
            "event_p95": round(float(np.quantile(ev_s, 0.95)), 2),
            "frac_above_thr": round(float(np.mean(ev_s > thr)), 3),
        })
        try:
            y = np.r_[np.zeros(len(tr_s)), np.ones(len(ev_s))]
            out["auc_vs_train"] = round(
                float(roc_auc_score(y, np.r_[tr_s, ev_s])), 3)
        except Exception:
            out["auc_vs_train"] = np.nan
        return out
    except Exception as e:                      # robust: never kill the batch
        return {"event_id": row.get("_id", "?"), "error": str(e)[:120]}


def batch_mode(farm_dir, ev, csvs, residual=False):
    suffix = "_residual" if residual else ""
    idc = "event_id" if "event_id" in ev.columns else ev.columns[0]
    rows = []
    for _, r in ev.iterrows():
        r = r.copy()
        r["_id"] = r[idc]
        path = find_dataset_file(csvs, r[idc])
        if path is None:
            rows.append({"event_id": r[idc], "error": "csv not found"})
            continue
        res = batch_analyse_one(path, r, residual=residual)
        rows.append(res)
        msg = res.get("error")
        if msg:
            print(f"event {r[idc]:>4}  [{r['event_label']:7s}] ERROR: {msg}")
        else:
            desc = res.get("description", "")
            print(f"event {r[idc]:>4}  [{res['label']:7s}] "
                  f"({res.get('mode','raw')[:12]}) "
                  f"elev={res.get('elevation', float('nan')):+6.2f}  "
                  f"AUC={res.get('auc_vs_train', float('nan')):.3f}  "
                  f"frac>thr={res.get('frac_above_thr', float('nan')):.3f}  "
                  f"{desc if isinstance(desc, str) else ''}")

    summ = pd.DataFrame(rows)
    s_name = f"care_farmA_batch_summary{suffix}.csv"
    summ.to_csv(s_name, index=False)

    ok = summ[summ.get("error").isna()] if "error" in summ else summ
    by_type = None
    if "description" in ok.columns and (ok.label == "anomaly").any():
        by_type = (ok[ok.label == "anomaly"]
                   .groupby("description")
                   .agg(n_events=("event_id", "count"),
                        mean_elevation=("elevation", "mean"),
                        mean_auc=("auc_vs_train", "mean"),
                        mean_frac_above=("frac_above_thr", "mean"))
                   .round(3).reset_index())
        by_type.to_csv(f"care_farmA_batch_by_type{suffix}.csv", index=False)

    # overview: AUC per event, anomalies vs normals
    if "auc_vs_train" in ok.columns and len(ok):
        d = ok.sort_values(["label", "auc_vs_train"])
        colors = ["tab:red" if l == "anomaly" else "tab:green"
                  for l in d.label]
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.5))
        ax.barh([f"{i} ({l})" for i, l in zip(d.event_id, d.label)],
                d.auc_vs_train, color=colors, alpha=0.8)
        ax.axvline(0.5, ls="--", color="gray", lw=1)
        ax.set_xlabel("AUC: window scores vs own training scores")
        ax.set_title(f"{farm_dir.name}{' (residual mode)' if residual else ''}: "
                     "anomaly (red, event window) vs normal (green, prediction window)")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(f"care_farmA_batch_overview{suffix}.png", dpi=150)

    # headline aggregates
    an = ok[ok.label == "anomaly"] if "label" in ok else ok.iloc[0:0]
    no = ok[ok.label == "normal"] if "label" in ok else ok.iloc[0:0]
    print(f"\nprocessed: {len(summ)} datasets "
          f"({len(an)} anomaly + {len(no)} normal OK, "
          f"{int(summ['error'].notna().sum()) if 'error' in summ else 0} errors)")
    if len(an):
        print(f"anomaly events : mean AUC={an.auc_vs_train.mean():.3f}, "
              f"AUC>0.5 on {(an.auc_vs_train > 0.5).sum()}/{len(an)}, "
              f"alarm (frac>thr>0) on {(an.frac_above_thr > 0).sum()}/{len(an)}")
    if len(no):
        print(f"normal datasets: mean AUC={no.auc_vs_train.mean():.3f}, "
              f"false alarms (frac>thr>0) on {(no.frac_above_thr > 0).sum()}/{len(no)}")
    if by_type is not None:
        print("\nby fault type:\n", by_type.to_string(index=False))
    print(f"\nsaved: {s_name}"
          + (f", care_farmA_batch_by_type{suffix}.csv" if by_type is not None else "")
          + f", care_farmA_batch_overview{suffix}.png, care_structure_summary.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="path to unzipped CARE_To_Compare folder")
    ap.add_argument("--farm", default="A")
    ap.add_argument("--all", action="store_true",
                    help="batch mode: evaluate every dataset in the farm")
    ap.add_argument("--residual", action="store_true",
                    help="batch mode: score residuals of a normal-behaviour "
                         "regression (targets ~ power/wind/rotor) instead of "
                         "raw Avg features")
    ap.add_argument("--anomaly-event", type=int, default=None)
    ap.add_argument("--normal-event", type=int, default=None)
    args = ap.parse_args()

    farms = inspect(Path(args.root))
    farm_dir = pick_farm(farms, args.farm)
    ev, csvs = farms[farm_dir]
    print(f"\nusing farm dir: {farm_dir}")

    if args.all:
        batch_mode(farm_dir, ev, csvs, residual=args.residual)
        return
    if args.residual:
        sys.exit("--residual currently requires --all (batch mode)")

    idc = "event_id" if "event_id" in ev.columns else ev.columns[0]
    anomalies = ev[ev.event_label == "anomaly"]
    normals = ev[ev.event_label == "normal"]
    if anomalies.empty or normals.empty:
        sys.exit("ERROR: event_info has no anomaly/normal rows as expected.")
    a_row = anomalies[anomalies[idc] == args.anomaly_event].iloc[0] \
        if args.anomaly_event is not None else anomalies.iloc[0]
    n_row = normals[normals[idc] == args.normal_event].iloc[0] \
        if args.normal_event is not None else normals.iloc[0]

    a_path = find_dataset_file(csvs, a_row[idc])
    n_path = find_dataset_file(csvs, n_row[idc])
    if a_path is None or n_path is None:
        sys.exit(f"ERROR: could not locate csv for events "
                 f"{a_row[idc]} / {n_row[idc]} in {farm_dir}")
    print(f"anomaly dataset: {a_path.name}  (event {a_row[idc]}: "
          f"{a_row.get('event_description', 'n/a')})")
    print(f"normal dataset : {n_path.name}  (event {n_row[idc]})")

    # ---- train on the anomaly dataset's own train year; reuse the SAME
    # detector on the normal dataset (same turbine type / farm) ------------
    try:
        adf, ameta, araw, asm, det, dcols = analyse_dataset(a_path, a_row)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    train_mask = (adf[ameta["train"]].astype(str).str.lower() == "train") \
        & adf[ameta["status"]].isin(NORMAL_STATUS)
    thr = np.quantile(asm[train_mask.values], 0.995)

    ndf, nmeta, nraw, nsm, _, _ = analyse_dataset(
        n_path, n_row, detector_cols=dcols, detector=det)

    # event window: labeled [event_start, event_end] if available;
    # fall back to (event_start -> end of data) or the prediction window.
    # NOTE (real data): for CARE anomaly events the event window often
    # coincides with the ENTIRE prediction period, so "prediction before
    # event" can legitimately be empty.
    e_start = pd.to_datetime(a_row.get("event_start",
                             a_row.get("event_start_time", None)),
                             errors="coerce")
    e_end = pd.to_datetime(a_row.get("event_end", None), errors="coerce")
    t_a = adf[ameta["time"]]
    pred_a = adf[ameta["train"]].astype(str).str.lower() != "train"
    if pd.notna(e_start):
        in_event = (t_a >= e_start)
        if pd.notna(e_end):
            in_event &= (t_a <= e_end)
    else:
        in_event = pred_a  # fallback: whole prediction window

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharey=True)
    for ax, (df_, meta_, sm_, title, ev_mask) in zip(axes, [
            (adf, ameta, asm, f"anomaly dataset (event {a_row[idc]})", in_event),
            (ndf, nmeta, nsm, f"normal dataset (event {n_row[idc]})", None)]):
        t = df_[meta_["time"]]
        pred = df_[meta_["train"]].astype(str).str.lower() != "train"
        ax.plot(t, sm_, lw=0.8, color="tab:blue", label="anomaly score (1d median)")
        ax.axhline(thr, ls="--", color="black", lw=1,
                   label="threshold (99.5% of train)")
        if pred.any():
            ax.axvspan(t[pred].iloc[0], t[pred].iloc[-1], color="orange",
                       alpha=0.12, label="prediction window")
        if ev_mask is not None and ev_mask.any():
            ax.axvspan(t[ev_mask].iloc[0], t[ev_mask].iloc[-1], color="red",
                       alpha=0.18, label="labeled anomaly event")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("-log likelihood")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("CARE (Wind Farm A) -- GMM normal-behaviour anomaly score",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("real_data_anomaly_score.png", dpi=150)

    # ---- summary ---------------------------------------------------------
    def region_stats(name, sm_, mask):
        n = int(np.sum(mask))
        if n == 0:
            return {"dataset": name, "n_points": 0, "score_mean": "n/a",
                    "score_p95": "n/a", "frac_above_thr": "n/a"}
        s = sm_[mask]
        return {"dataset": name, "n_points": n,
                "score_mean": round(float(np.mean(s)), 2),
                "score_p95": round(float(np.quantile(s, 0.95)), 2),
                "frac_above_thr": round(float(np.mean(s > thr)), 3)}

    rows = [
        region_stats("anomaly/train(normal)", asm, train_mask.values),
        region_stats("anomaly/pred_before_event", asm,
                     (pred_a & ~in_event).values),
        region_stats("anomaly/event_window", asm, in_event.values),
        region_stats("normal/prediction", nsm,
                     (ndf[nmeta["train"]].astype(str).str.lower() != "train").values),
    ]
    out = pd.DataFrame(rows)
    out.insert(1, "threshold", round(float(thr), 2))
    out.to_csv("real_data_validation_summary.csv", index=False)
    print("\n", out.to_string(index=False))
    print("\nsaved: real_data_anomaly_score.png, "
          "real_data_validation_summary.csv, care_structure_summary.txt")


if __name__ == "__main__":
    main()

"""Farm B batch validation -- standalone, does NOT modify real_data_care.py.

Reuses the tested helper functions from real_data_care.py (loading, Avg
column selection, PCA+GMM detector, residual computation, per-event
analysis) and only changes (a) which farm is selected and (b) the output
filenames (care_farmB_*). Farm A behaviour is therefore untouched.

Usage:
    python real_data_care_farmB.py ".\\CARE_To_Compare"
    python real_data_care_farmB.py ".\\CARE_To_Compare" --residual
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the already-verified Farm A helpers unchanged.
from real_data_care import (inspect, find_dataset_file, batch_analyse_one)


def pick_farm_b(farms):
    """Select the Wind Farm B directory explicitly."""
    for d in farms:
        name = d.name.lower().replace("_", " ")
        if name.endswith(" b") or name.endswith("farm b") or "farm b" in name:
            return d
    # fallback: any dir whose last token is exactly 'b'
    for d in farms:
        if d.name.strip().lower().split()[-1] == "b":
            return d
    sys.exit("ERROR: could not find a 'Wind Farm B' directory under the "
             "given root. Directories found: "
             + ", ".join(d.name for d in farms))


def run_farm_b(root, residual=False):
    suffix = "_residual" if residual else ""
    farms = inspect(Path(root))
    farm_dir = pick_farm_b(farms)
    ev, csvs = farms[farm_dir]
    print(f"\nusing farm dir: {farm_dir}  (residual={residual})")

    idc = "event_id" if "event_id" in ev.columns else ev.columns[0]
    rows, fallback_seen = [], False
    for _, r in ev.iterrows():
        r = r.copy()
        r["_id"] = r[idc]
        path = find_dataset_file(csvs, r[idc])
        if path is None:
            rows.append({"event_id": r[idc], "error": "csv not found"})
            print(f"event {r[idc]:>4}  ERROR: csv not found")
            continue
        res = batch_analyse_one(path, r, residual=residual)
        rows.append(res)
        if res.get("error"):
            print(f"event {r[idc]:>4}  [{r['event_label']:7s}] "
                  f"ERROR: {res['error']}")
            continue
        mode = res.get("mode", "raw")
        if residual and str(mode).startswith("raw-fallback"):
            fallback_seen = True
        desc = res.get("description", "")
        print(f"event {r[idc]:>4}  [{res['label']:7s}] ({str(mode)[:20]}) "
              f"elev={res.get('elevation', float('nan')):+6.2f}  "
              f"AUC={res.get('auc_vs_train', float('nan')):.3f}  "
              f"frac>thr={res.get('frac_above_thr', float('nan')):.3f}  "
              f"{desc if isinstance(desc, str) else ''}")

    if residual and fallback_seen:
        print("\nWARNING: residual column discovery failed for one or more "
              "datasets; those rows fell back to raw Avg features "
              "(see the 'mode' column, value 'raw-fallback...').")

    summ = pd.DataFrame(rows)
    s_name = f"care_farmB_batch_summary{suffix}.csv"
    summ.to_csv(s_name, index=False)

    ok = summ[summ["error"].isna()] if "error" in summ.columns else summ
    saved = [s_name]

    by_type = None
    if "description" in ok.columns and "label" in ok.columns \
            and (ok.label == "anomaly").any():
        by_type = (ok[ok.label == "anomaly"]
                   .groupby("description")
                   .agg(n_events=("event_id", "count"),
                        mean_elevation=("elevation", "mean"),
                        mean_auc=("auc_vs_train", "mean"),
                        mean_frac_above=("frac_above_thr", "mean"))
                   .round(3).reset_index())
        bt_name = f"care_farmB_batch_by_type{suffix}.csv"
        by_type.to_csv(bt_name, index=False)
        saved.append(bt_name)

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
                     "anomaly (red, event window) vs normal (green, prediction)")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        png = f"care_farmB_batch_overview{suffix}.png"
        fig.savefig(png, dpi=150)
        saved.append(png)

    # ---- final summary -------------------------------------------------
    an = ok[ok.label == "anomaly"] if "label" in ok.columns else ok.iloc[0:0]
    no = ok[ok.label == "normal"] if "label" in ok.columns else ok.iloc[0:0]
    n_err = int(summ["error"].notna().sum()) if "error" in summ.columns else 0
    print("\n----- Farm B summary -----")
    print(f"datasets processed : {len(summ)} ({n_err} errors)")
    print(f"anomalies          : {len(an)}")
    print(f"normals            : {len(no)}")
    if len(an):
        print(f"mean anomaly AUC   : {an.auc_vs_train.mean():.3f}")
        print(f"AUC > 0.5          : {(an.auc_vs_train > 0.5).sum()}/{len(an)}")
        print(f"threshold alarms   : {(an.frac_above_thr > 0).sum()}/{len(an)}")
    if len(no):
        print(f"normal false alarms: {(no.frac_above_thr > 0).sum()}/{len(no)}")
    if by_type is not None:
        print("\nby fault type:\n", by_type.to_string(index=False))
    print("\nsaved: " + ", ".join(saved))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="path to unzipped CARE_To_Compare folder")
    ap.add_argument("--residual", action="store_true",
                    help="score residuals of a normal-behaviour regression "
                         "instead of raw Avg features")
    args = ap.parse_args()
    run_farm_b(args.root, residual=args.residual)


if __name__ == "__main__":
    main()

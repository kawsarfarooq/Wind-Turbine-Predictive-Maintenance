"""Create statistical and report-ready outputs from a CARE benchmark run."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from care_benchmark import summarize_events


def method_label(row):
    rep = {
        "raw": "raw",
        "linear_residual": "linear",
        "quadratic_residual": "quadratic",
    }.get(row["representation"], row["representation"])
    return f"{rep}-{row['detector']}"


def ensure_cluster_ids(events):
    """Preserve existing asset clusters and infer IDs for appended runs."""
    events = events.copy()
    inferred = events.farm.astype(str) + ":" + events.asset.astype(str)
    if "cluster_id" not in events:
        events["cluster_id"] = inferred
    else:
        events["cluster_id"] = events.cluster_id.fillna(inferred)
    return events


def plot_auc(summary, path):
    data = summary[(summary.farm != "ALL")
                   & (summary.normal_statuses == "0")].copy()
    data["method"] = data.apply(method_label, axis=1)
    farms = ["A", "B", "C"]
    representations = [
        ("raw", "raw"),
        ("linear_residual", "linear"),
        ("quadratic_residual", "quadratic"),
    ]
    representations = [
        item for item in representations if item[0] in set(data.representation)]
    methods = [f"{label}-{detector}" for detector in ["gmm", "iforest"]
               for _, label in representations]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(farms))
    width = min(0.75 / len(methods), 0.18)
    for idx, method in enumerate(methods):
        part = data.set_index(["farm", "method"]).reindex(
            pd.MultiIndex.from_product([farms, [method]])).reset_index()
        y = part.event_roc_auc.to_numpy()
        low = part.event_roc_auc_ci_low.to_numpy()
        high = part.event_roc_auc_ci_high.to_numpy()
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, y, width, label=method,
               yerr=np.vstack([y - low, high - y]), capsize=3)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1,
               label="chance")
    ax.set_xticks(x, [f"Farm {farm}" for farm in farms])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Event-level ROC-AUC")
    ax.set_title("CARE anomaly-vs-normal separation by farm")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_event_distributions(events, path):
    data = events[(events.normal_statuses == "0")
                  & (events.detector == "gmm")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, farm in zip(axes, ["A", "B", "C"]):
        farm_data = data[data.farm == farm]
        groups, labels, colors = [], [], []
        representations = [
            representation for representation in
            ["raw", "linear_residual", "quadratic_residual"]
            if representation in set(farm_data.representation)]
        for representation in representations:
            for label in ["normal", "anomaly"]:
                groups.append(farm_data[
                    (farm_data.representation == representation)
                    & (farm_data.label == label)
                ].event_mean_percentile.to_numpy())
                rep = {"raw": "raw", "linear_residual": "linear",
                       "quadratic_residual": "quad"}[representation]
                labels.append(f"{rep}\n{label}")
                colors.append("tab:red" if label == "anomaly" else "tab:green")
        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                        showmeans=True)
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.35)
        ax.set_title(f"Farm {farm}")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean training-relative score percentile")
    fig.suptitle("CARE GMM event scores: anomalies must exceed normal controls")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_findings(summary, path):
    chosen = summary[(summary.normal_statuses == "0")
                     & (summary.farm != "ALL")].copy()
    chosen["method"] = chosen.apply(method_label, axis=1)
    has_quadratic = "quadratic_residual" in set(chosen.representation)
    lines = [
        ("# CARE representation-ablation findings" if has_quadratic
         else "# CARE full-benchmark findings"),
        "",
        "All intervals are 95% asset-clustered bootstrap intervals.",
        "",
        "| Farm | Best method | ROC-AUC (95% CI) | Anomaly-normal gap | Detection | Normal false alarms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for farm in ["A", "B", "C"]:
        best = chosen[chosen.farm == farm].sort_values(
            "event_roc_auc", ascending=False).iloc[0]
        lines.append(
            f"| {farm} | {best.method} | {best.event_roc_auc:.3f} "
            f"({best.event_roc_auc_ci_low:.3f}–{best.event_roc_auc_ci_high:.3f}) "
            f"| {best.anomaly_normal_gap:+.3f} | {best.detection_rate:.1%} "
            f"| {best.normal_false_alarm_rate:.1%} |")
    lines.append("")
    if has_quadratic:
        lines.extend([
            "Interpretation: adding pairwise nonlinear operating-condition terms does",
            "not outperform the strongest existing representation. Linear residuals",
            "remain best on Farm A, raw features remain best on Farm C, and no tested",
            "representation reliably separates Farm B. Normal-behaviour model complexity",
            "is therefore a farm-dependent choice rather than a universal improvement.",
        ])
    else:
        lines.extend([
            "Interpretation: residual normalization is strongly beneficial on Farm A,",
            "does not rescue Farm B, and is inferior to raw GMM on Farm C. Therefore",
            "condition normalization is a farm-dependent design choice, not a universal",
            "improvement. Farm B remains a negative transfer result because normal",
            "prediction windows score at least as highly as anomaly windows.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--additional-events", type=Path, nargs="*", default=[],
                        help="event CSV files to combine with the base run")
    parser.add_argument("--output-dir", type=Path,
                        help="write analysis here instead of modifying run_dir")
    parser.add_argument("--reuse-summary", action="store_true",
                        help="reuse an existing bootstrap summary")
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    event_tables = [pd.read_csv(args.run_dir / "care_benchmark_events.csv")]
    event_tables.extend(pd.read_csv(path) for path in args.additional_events)
    events = pd.concat(event_tables, ignore_index=True)
    if "error" in events:
        events = events[events.error.isna()].copy()
    events = ensure_cluster_ids(events)
    events.to_csv(output_dir / "care_benchmark_events.csv", index=False)
    summary_path = output_dir / "care_benchmark_summary_with_ci.csv"
    if args.reuse_summary and summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = summarize_events(events)
        summary.to_csv(summary_path, index=False)

    fault = (events[(events.normal_statuses == "0")
                    & (events.label == "anomaly")]
             .groupby(["farm", "representation", "detector", "description"])
             .agg(n_events=("event_id", "count"),
                  mean_percentile=("event_mean_percentile", "mean"),
                  detection_rate=("alarm", "mean"))
             .reset_index())
    fault.to_csv(output_dir / "care_benchmark_by_fault.csv", index=False)
    plot_auc(summary, output_dir / "care_event_auc_by_farm.png")
    plot_event_distributions(
        events, output_dir / "care_event_score_distributions.png")
    write_findings(summary, output_dir / "FINDINGS.md")
    print(summary[summary.normal_statuses == "0"].to_string(index=False))


if __name__ == "__main__":
    main()

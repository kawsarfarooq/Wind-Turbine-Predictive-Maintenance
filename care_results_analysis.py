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
    rep = "residual" if row["representation"] == "linear_residual" else "raw"
    return f"{rep}-{row['detector']}"


def plot_auc(summary, path):
    data = summary[(summary.farm != "ALL")
                   & (summary.normal_statuses == "0")].copy()
    data["method"] = data.apply(method_label, axis=1)
    farms, methods = ["A", "B", "C"], [
        "raw-gmm", "residual-gmm", "raw-iforest", "residual-iforest"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(farms))
    width = 0.18
    for idx, method in enumerate(methods):
        part = data.set_index(["farm", "method"]).reindex(
            pd.MultiIndex.from_product([farms, [method]])).reset_index()
        y = part.event_roc_auc.to_numpy()
        low = part.event_roc_auc_ci_low.to_numpy()
        high = part.event_roc_auc_ci_high.to_numpy()
        ax.bar(x + (idx - 1.5) * width, y, width, label=method,
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
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, farm in zip(axes, ["A", "B", "C"]):
        farm_data = data[data.farm == farm]
        groups, labels, colors = [], [], []
        for representation in ["raw", "linear_residual"]:
            for label in ["normal", "anomaly"]:
                groups.append(farm_data[
                    (farm_data.representation == representation)
                    & (farm_data.label == label)
                ].event_mean_percentile.to_numpy())
                labels.append(
                    f"{'resid' if representation == 'linear_residual' else 'raw'}\n{label}")
                colors.append("tab:red" if label == "anomaly" else "tab:green")
        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                        showmeans=True)
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.35)
        ax.set_title(f"Farm {farm}")
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
    lines = [
        "# CARE full-benchmark findings",
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
    lines.extend([
        "",
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
    parser.add_argument("--reuse-summary", action="store_true",
                        help="reuse an existing bootstrap summary")
    args = parser.parse_args()
    events = pd.read_csv(args.run_dir / "care_benchmark_events.csv")
    if "error" in events:
        events = events[events.error.isna()].copy()
    if "cluster_id" not in events:
        events["cluster_id"] = (
            events.farm.astype(str) + ":" + events.asset.astype(str))
    summary_path = args.run_dir / "care_benchmark_summary_with_ci.csv"
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
    fault.to_csv(args.run_dir / "care_benchmark_by_fault.csv", index=False)
    plot_auc(summary, args.run_dir / "care_event_auc_by_farm.png")
    plot_event_distributions(
        events, args.run_dir / "care_event_score_distributions.png")
    write_findings(summary, args.run_dir / "FINDINGS.md")
    print(summary[summary.normal_statuses == "0"].to_string(index=False))


if __name__ == "__main__":
    main()

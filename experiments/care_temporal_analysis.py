"""Analyse CARE smoothing-window and threshold-calibration ablations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from care_benchmark import summarize_events


REFERENCE_QUANTILE = 0.995
CONFIG_LABELS = {
    ("A", "linear_residual", "gmm"): "linear-GMM",
    ("B", "quadratic_residual", "iforest"): "quadratic-IF",
    ("C", "raw", "gmm"): "raw-GMM",
}


def config_label(row):
    return CONFIG_LABELS.get(
        (row["farm"], row["representation"], row["detector"]),
        f"{row['representation']}-{row['detector']}")


def load_event_tables(paths):
    events = pd.concat([pd.read_csv(path) for path in paths],
                       ignore_index=True)
    if "error" in events:
        events = events[events.error.isna()].copy()
    events = events[events.normal_statuses.astype(str) == "0"].copy()
    events["smooth_steps"] = pd.to_numeric(events.smooth_steps).astype(int)
    events["threshold_quantile"] = pd.to_numeric(
        events.threshold_quantile)
    return events


def discrimination_summary(events):
    reference = events[np.isclose(
        events.threshold_quantile, REFERENCE_QUANTILE)].copy()
    reference = reference.drop(columns=["threshold_quantile"])
    summary = summarize_events(reference)
    summary = summary[summary.farm != "ALL"].copy()
    summary["smooth_hours"] = summary.smooth_steps / 6.0
    summary["configuration"] = summary.apply(config_label, axis=1)
    return summary


def threshold_summary(events):
    keys = ["farm", "representation", "detector", "smooth_steps",
            "threshold_quantile"]

    def rates(group):
        anomaly = group[group.label == "anomaly"]
        normal = group[group.label == "normal"]
        return pd.Series({
            "n_anomaly": len(anomaly),
            "n_normal": len(normal),
            "detection_rate": anomaly.alarm.mean(),
            "normal_false_alarm_rate": normal.alarm.mean(),
            "mean_anomaly_fraction_above": anomaly.frac_above_threshold.mean(),
            "mean_normal_fraction_above": normal.frac_above_threshold.mean(),
        })

    summary = events.groupby(keys, dropna=False).apply(
        rates, include_groups=False).reset_index()
    summary["smooth_hours"] = summary.smooth_steps / 6.0
    summary["configuration"] = summary.apply(config_label, axis=1)
    summary["detection_minus_false_alarm"] = (
        summary.detection_rate - summary.normal_false_alarm_rate)
    return summary


def exploratory_operating_points(summary, max_false_alarm_rate=0.20):
    """Select descriptive points under a normal-event false-alarm constraint."""
    selected = []
    for farm in ["A", "B", "C"]:
        feasible = summary[
            (summary.farm == farm)
            & (summary.normal_false_alarm_rate <= max_false_alarm_rate)
        ].sort_values(
            ["detection_minus_false_alarm", "detection_rate",
             "normal_false_alarm_rate"],
            ascending=[False, False, True])
        if len(feasible):
            selected.append(feasible.iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def window_label(steps):
    labels = {1: "none (10 min)", 6: "1 h", 36: "6 h",
              144: "24 h", 432: "72 h"}
    return labels.get(int(steps), f"{steps / 6:g} h")


def plot_discrimination(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    order = sorted(summary.smooth_steps.unique())
    x = np.arange(len(order))
    for ax, farm in zip(axes, ["A", "B", "C"]):
        part = summary[summary.farm == farm].set_index(
            "smooth_steps").reindex(order)
        y = part.event_roc_auc.to_numpy()
        low = part.event_roc_auc_ci_low.to_numpy()
        high = part.event_roc_auc_ci_high.to_numpy()
        ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]),
                    marker="o", capsize=4, linewidth=2)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x, [window_label(value) for value in order], rotation=30)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Farm {farm}: {part.configuration.dropna().iloc[0]}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Event-level ROC-AUC (95% clustered CI)")
    fig.suptitle("Effect of causal score smoothing on CARE discrimination")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoffs(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    windows = sorted(summary.smooth_steps.unique())
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(windows)))
    for ax, farm in zip(axes, ["A", "B", "C"]):
        farm_data = summary[summary.farm == farm]
        for steps, color in zip(windows, colors):
            part = farm_data[farm_data.smooth_steps == steps].sort_values(
                "threshold_quantile")
            ax.plot(part.normal_false_alarm_rate, part.detection_rate,
                    marker="o", color=color, label=window_label(steps))
            for _, row in part.iterrows():
                if np.isclose(row.threshold_quantile, 0.995) or np.isclose(
                        row.threshold_quantile, 0.999):
                    ax.annotate(f"{row.threshold_quantile:.3f}",
                                (row.normal_false_alarm_rate,
                                 row.detection_rate),
                                xytext=(3, 3), textcoords="offset points",
                                fontsize=6, color=color)
        ax.axvline(0.2, color="grey", linestyle=":", linewidth=1)
        ax.set_title(f"Farm {farm}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Anomaly-event detection rate")
    axes[1].set_xlabel("Normal-event false-alarm rate")
    axes[0].legend(title="Trailing median", fontsize=7, title_fontsize=8)
    fig.suptitle("Threshold trade-off (labels show q=0.995 and 0.999)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_findings(discrimination, thresholds, operating_points, path):
    lines = [
        "# CARE temporal and threshold ablation findings",
        "",
        "Configurations were fixed from the preceding representation ablation:",
        "linear-GMM for Farm A, exploratory quadratic-IF for Farm B, and raw-GMM",
        "for Farm C. Intervals are 95% asset-clustered bootstrap intervals.",
        "",
        "| Farm | Configuration | Best smoothing | ROC-AUC (95% CI) | 24 h ROC-AUC | Detection / false alarms at 24 h, q=0.995 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for farm in ["A", "B", "C"]:
        farm_data = discrimination[discrimination.farm == farm]
        best = farm_data.sort_values("event_roc_auc", ascending=False).iloc[0]
        default_auc = farm_data[farm_data.smooth_steps == 144].iloc[0]
        default_alarm = thresholds[
            (thresholds.farm == farm)
            & (thresholds.smooth_steps == 144)
            & np.isclose(thresholds.threshold_quantile, REFERENCE_QUANTILE)
        ].iloc[0]
        lines.append(
            f"| {farm} | {best.configuration} | {window_label(best.smooth_steps)} "
            f"| {best.event_roc_auc:.3f} "
            f"({best.event_roc_auc_ci_low:.3f} to "
            f"{best.event_roc_auc_ci_high:.3f}) "
            f"| {default_auc.event_roc_auc:.3f} "
            f"| {default_alarm.detection_rate:.1%} / "
            f"{default_alarm.normal_false_alarm_rate:.1%} |")
    lines.extend([
        "",
        "Exploratory operating points maximizing detection minus false alarms",
        "subject to a normal-event false-alarm rate of at most 20%:",
        "",
        "| Farm | Smoothing | Training quantile | Detection | Normal false alarms |",
        "|---|---:|---:|---:|---:|",
    ])
    for _, point in operating_points.iterrows():
        lines.append(
            f"| {point.farm} | {window_label(point.smooth_steps)} "
            f"| {point.threshold_quantile:.3f} | {point.detection_rate:.1%} "
            f"| {point.normal_false_alarm_rate:.1%} |")
    lines.extend([
        "",
        "Smoothing selection is exploratory because the same labeled CARE events",
        "were used to compare windows; it is not an independently validated tuning",
        "result. Threshold quantiles are estimated from smoothed scores at normal-",
        "training timestamps, while labeled event controls are reserved for evaluation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("event_csvs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-discrimination", action="store_true",
                        help="reuse existing clustered discrimination intervals")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    events = load_event_tables(args.event_csvs)
    discrimination_path = (
        args.output / "temporal_discrimination_with_ci.csv")
    if args.reuse_discrimination and discrimination_path.exists():
        discrimination = pd.read_csv(discrimination_path)
    else:
        discrimination = discrimination_summary(events)
    thresholds = threshold_summary(events)
    operating_points = exploratory_operating_points(thresholds)
    events.to_csv(args.output / "temporal_ablation_events.csv", index=False)
    discrimination.to_csv(discrimination_path, index=False)
    thresholds.to_csv(
        args.output / "threshold_tradeoffs.csv", index=False)
    operating_points.to_csv(
        args.output / "exploratory_operating_points.csv", index=False)
    plot_discrimination(
        discrimination, args.output / "temporal_auc_by_farm.png")
    plot_tradeoffs(thresholds, args.output / "threshold_tradeoffs.png")
    write_findings(discrimination, thresholds, operating_points,
                   args.output / "FINDINGS.md")
    metadata = {
        "event_csvs": [str(path) for path in args.event_csvs],
        "event_rows": len(events),
        "normal_statuses": "0",
        "reference_threshold_quantile": REFERENCE_QUANTILE,
        "smoothing_steps": sorted(events.smooth_steps.unique().tolist()),
        "threshold_quantiles": sorted(
            events.threshold_quantile.unique().tolist()),
        "confidence_intervals": "1000-resample asset-clustered bootstrap",
        "selection_status": "exploratory; same labeled events used for comparison",
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(discrimination.to_string(index=False))
    print("\n", thresholds.to_string(index=False))


if __name__ == "__main__":
    main()

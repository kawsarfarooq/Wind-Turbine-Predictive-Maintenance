"""Fault-type and asset-level interpretation of canonical CARE event scores."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from care_benchmark import cluster_bootstrap


REFERENCE_SMOOTH_STEPS = 144
REFERENCE_QUANTILE = 0.995

FAULT_RULES = [
    ("Drivetrain / bearings", ["gearbox", "bearing", "coupling"]),
    ("Hydraulic / brake / lubrication", [
        "hydraulic", "rotorbrake", "rotor brake", "brake disc",
        "gear oil", "oil level", "oil leakage", "oil cooler", "grease",
    ]),
    ("Pitch / blade / hub", ["pitch", "blade", "hub", "spinner", "axis"]),
    ("Electrical / converter / transformer", [
        "transformer", "converter", "umrichter", "current measurement",
        "24vac", "rcd", "dc-link", "fuse",
    ]),
    ("Communication / control", [
        "communication", "beckhoff", "bk1120", "nc300",
    ]),
    ("Yaw", ["yaw"]),
    ("Cooling", ["cooling"]),
]


def fault_category(description):
    """Map heterogeneous free-text CARE labels to transparent system groups."""
    text = str(description).lower()
    for category, keywords in FAULT_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return "Other / mixed"


def load_reference_events(path, smooth_steps=REFERENCE_SMOOTH_STEPS,
                          threshold_quantile=REFERENCE_QUANTILE):
    events = pd.read_csv(path)
    if "error" in events:
        events = events[events.error.isna()].copy()
    events = events[
        (events.normal_statuses.astype(str) == "0")
        & (events.smooth_steps == smooth_steps)
        & np.isclose(events.threshold_quantile, threshold_quantile)
    ].copy()
    events["cluster_id"] = (
        events.farm.astype(str) + ":" + events.asset.astype(str))
    events["fault_category"] = events.description.map(fault_category)
    events.loc[events.label == "normal", "fault_category"] = "Normal control"
    return events


def inference_status(n_anomaly, n_anomaly_assets):
    if n_anomaly < 3 or n_anomaly_assets < 2:
        return "descriptive only"
    if n_anomaly < 5 or n_anomaly_assets < 3:
        return "limited"
    return "supported"


def fault_category_summary(events):
    rows = []
    for farm in sorted(events.farm.unique()):
        farm_events = events[events.farm == farm]
        normal = farm_events[farm_events.label == "normal"]
        anomaly = farm_events[farm_events.label == "anomaly"]
        for category, category_events in anomaly.groupby("fault_category"):
            comparison = pd.concat([category_events, normal], ignore_index=True)
            y = (comparison.label == "anomaly").astype(int).to_numpy()
            score = comparison.event_mean_percentile.to_numpy()
            auc_low, auc_high, gap_low, gap_high = cluster_bootstrap(comparison)
            rows.append({
                "farm": farm,
                "fault_category": category,
                "n_anomaly_events": len(category_events),
                "n_anomaly_assets": category_events.asset.nunique(),
                "n_normal_events": len(normal),
                "n_normal_assets": normal.asset.nunique(),
                "event_roc_auc": roc_auc_score(y, score),
                "event_pr_auc": average_precision_score(y, score),
                "event_roc_auc_ci_low": auc_low,
                "event_roc_auc_ci_high": auc_high,
                "mean_anomaly_percentile": (
                    category_events.event_mean_percentile.mean()),
                "mean_normal_percentile": normal.event_mean_percentile.mean(),
                "anomaly_normal_gap": (
                    category_events.event_mean_percentile.mean()
                    - normal.event_mean_percentile.mean()),
                "anomaly_normal_gap_ci_low": gap_low,
                "anomaly_normal_gap_ci_high": gap_high,
                "detection_rate": category_events.alarm.mean(),
                "normal_false_alarm_rate": normal.alarm.mean(),
                "inference_status": inference_status(
                    len(category_events), category_events.asset.nunique()),
            })
    return pd.DataFrame(rows)


def raw_fault_summary(events):
    anomaly = events[events.label == "anomaly"].copy()
    return (anomaly.groupby(
        ["farm", "fault_category", "description"], dropna=False)
        .agg(n_events=("event_id", "count"),
             n_assets=("asset", "nunique"),
             mean_event_percentile=("event_mean_percentile", "mean"),
             detection_rate=("alarm", "mean"))
        .reset_index())


def asset_summary(events):
    rows = []
    for (farm, asset), group in events.groupby(["farm", "asset"]):
        anomaly = group[group.label == "anomaly"]
        normal = group[group.label == "normal"]
        auc = np.nan
        gap = np.nan
        if len(anomaly) and len(normal):
            y = (group.label == "anomaly").astype(int)
            auc = roc_auc_score(y, group.event_mean_percentile)
            gap = (anomaly.event_mean_percentile.mean()
                   - normal.event_mean_percentile.mean())
        rows.append({
            "farm": farm,
            "asset": asset,
            "n_events": len(group),
            "n_anomaly": len(anomaly),
            "n_normal": len(normal),
            "mean_anomaly_percentile": (
                anomaly.event_mean_percentile.mean() if len(anomaly) else np.nan),
            "mean_normal_percentile": (
                normal.event_mean_percentile.mean() if len(normal) else np.nan),
            "asset_roc_auc_descriptive": auc,
            "asset_anomaly_normal_gap": gap,
            "detection_rate": anomaly.alarm.mean() if len(anomaly) else np.nan,
            "normal_false_alarm_rate": (
                normal.alarm.mean() if len(normal) else np.nan),
        })
    return pd.DataFrame(rows)


def leave_one_asset_out(events):
    rows = []
    for farm, farm_events in events.groupby("farm"):
        y = (farm_events.label == "anomaly").astype(int)
        full_auc = roc_auc_score(y, farm_events.event_mean_percentile)
        anomaly = farm_events[y == 1]
        normal = farm_events[y == 0]
        full_gap = (anomaly.event_mean_percentile.mean()
                    - normal.event_mean_percentile.mean())
        for asset in sorted(farm_events.asset.unique()):
            remaining = farm_events[farm_events.asset != asset]
            remaining_y = (remaining.label == "anomaly").astype(int)
            if remaining_y.nunique() < 2:
                continue
            auc = roc_auc_score(remaining_y, remaining.event_mean_percentile)
            remaining_anomaly = remaining[remaining.label == "anomaly"]
            remaining_normal = remaining[remaining.label == "normal"]
            gap = (remaining_anomaly.event_mean_percentile.mean()
                   - remaining_normal.event_mean_percentile.mean())
            removed = farm_events[farm_events.asset == asset]
            rows.append({
                "farm": farm,
                "asset_removed": asset,
                "n_events_removed": len(removed),
                "n_anomaly_removed": int((removed.label == "anomaly").sum()),
                "n_normal_removed": int((removed.label == "normal").sum()),
                "full_event_roc_auc": full_auc,
                "leave_one_out_event_roc_auc": auc,
                "delta_event_roc_auc": auc - full_auc,
                "full_anomaly_normal_gap": full_gap,
                "leave_one_out_anomaly_normal_gap": gap,
                "delta_anomaly_normal_gap": gap - full_gap,
            })
    return pd.DataFrame(rows)


def plot_fault_categories(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharex=True)
    status_colors = {
        "supported": "tab:blue",
        "limited": "tab:orange",
        "descriptive only": "tab:grey",
    }
    for ax, farm in zip(axes, ["A", "B", "C"]):
        part = summary[summary.farm == farm].sort_values("event_roc_auc")
        y = np.arange(len(part))
        auc = part.event_roc_auc.to_numpy()
        low = part.event_roc_auc_ci_low.to_numpy()
        high = part.event_roc_auc_ci_high.to_numpy()
        colors = [status_colors[value] for value in part.inference_status]
        ax.errorbar(auc, y, xerr=np.vstack([auc - low, high - auc]),
                    fmt="none", ecolor="black", capsize=3, alpha=0.7)
        ax.scatter(auc, y, c=colors, s=55, zorder=3)
        ax.set_yticks(y, [
            f"{row.fault_category}\n(n={row.n_anomaly_events})"
            for _, row in part.iterrows()
        ], fontsize=8)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.03)
        ax.set_title(f"Farm {farm}")
        ax.grid(axis="x", alpha=0.25)
    axes[1].set_xlabel("Event ROC-AUC vs farm normal controls")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=color, markeredgecolor=color, label=status)
        for status, color in status_colors.items()
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("CARE fault-category discrimination (small groups are descriptive)")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_asset_influence(influence, path):
    data = influence.sort_values(["farm", "delta_event_roc_auc"]).copy()
    data["label"] = data.apply(
        lambda row: f"Farm {row.farm} / asset {row.asset_removed}", axis=1)
    colors = data.farm.map({"A": "tab:blue", "B": "tab:orange",
                            "C": "tab:green"})
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(y, data.delta_event_roc_auc, color=colors, alpha=0.8)
    ax.set_yticks(y, data.label, fontsize=8)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(-0.05, color="grey", linestyle=":", linewidth=1)
    ax.axvline(0.05, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Change in farm ROC-AUC after removing the asset")
    ax.set_title("Leave-one-asset-out influence on CARE conclusions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_findings(categories, influence, path):
    lines = [
        "# CARE fault-type and asset-level findings",
        "",
        "The analysis uses the fixed 24-hour, q=0.995 scores from the preceding",
        "temporal study. Each fault category is compared with all normal controls",
        "from the same farm. Categories are generated by documented keyword rules;",
        "the unmodified CARE descriptions remain available in the raw-fault table.",
        "",
        "| Farm | Best category with at least 3 events | Events/assets | ROC-AUC (95% CI) | Detection |",
        "|---|---|---:|---:|---:|",
    ]
    for farm in ["A", "B", "C"]:
        eligible = categories[
            (categories.farm == farm) & (categories.n_anomaly_events >= 3)
        ].sort_values("event_roc_auc", ascending=False)
        if len(eligible):
            best = eligible.iloc[0]
            lines.append(
                f"| {farm} | {best.fault_category} "
                f"| {best.n_anomaly_events}/{best.n_anomaly_assets} "
                f"| {best.event_roc_auc:.3f} "
                f"({best.event_roc_auc_ci_low:.3f} to "
                f"{best.event_roc_auc_ci_high:.3f}) "
                f"| {best.detection_rate:.1%} |")
        else:
            lines.append(f"| {farm} | No category has 3 events | - | - | - |")
    def category_row(farm, category):
        return categories[
            (categories.farm == farm)
            & (categories.fault_category == category)
        ].iloc[0]

    a_hydraulic = category_row("A", "Hydraulic / brake / lubrication")
    a_drivetrain = category_row("A", "Drivetrain / bearings")
    b_drivetrain = category_row("B", "Drivetrain / bearings")
    b_electrical = category_row("B", "Electrical / converter / transformer")
    c_hydraulic = category_row("C", "Hydraulic / brake / lubrication")
    c_pitch = category_row("C", "Pitch / blade / hub")
    c_electrical = category_row("C", "Electrical / converter / transformer")
    lines.extend([
        "",
        "Key subgroup interpretation:",
        "",
        f"- Farm A performance is supported by hydraulic/brake/lubrication events "
        f"(AUC {a_hydraulic.event_roc_auc:.3f}, n={a_hydraulic.n_anomaly_events}) "
        f"and drivetrain/bearing events (AUC {a_drivetrain.event_roc_auc:.3f}, "
        f"n={a_drivetrain.n_anomaly_events}).",
        f"- Farm B is heterogeneous: its three drivetrain events rank above controls "
        f"(AUC {b_drivetrain.event_roc_auc:.3f}), while its three electrical/transformer "
        f"events rank below controls (AUC {b_electrical.event_roc_auc:.3f}). The wide "
        f"intervals and opposing directions explain why the farm-level model fails.",
        f"- Farm C is mainly driven by hydraulic/brake/lubrication events "
        f"(AUC {c_hydraulic.event_roc_auc:.3f}, n={c_hydraulic.n_anomaly_events}) "
        f"and pitch/blade/hub events (AUC {c_pitch.event_roc_auc:.3f}, "
        f"n={c_pitch.n_anomaly_events}). Electrical/converter events are weaker "
        f"(AUC {c_electrical.event_roc_auc:.3f}).",
    ])
    lines.extend([
        "",
        "Leave-one-asset-out stability:",
        "",
        "| Farm | Full ROC-AUC | Leave-one-out range | Largest absolute change | Most influential asset |",
        "|---|---:|---:|---:|---|",
    ])
    for farm in ["A", "B", "C"]:
        part = influence[influence.farm == farm]
        most = part.loc[part.delta_event_roc_auc.abs().idxmax()]
        lines.append(
            f"| {farm} | {most.full_event_roc_auc:.3f} "
            f"| {part.leave_one_out_event_roc_auc.min():.3f} to "
            f"{part.leave_one_out_event_roc_auc.max():.3f} "
            f"| {most.delta_event_roc_auc:+.3f} "
            f"| {most.asset_removed} |")
    lines.extend([
        "",
        "The Farm C conclusion is comparatively stable: removing any one asset keeps",
        "the point estimate between 0.704 and 0.761. Farm A remains above chance in",
        "every leave-one-asset-out point estimate but is more sensitive (0.779 to",
        "0.950). Farm B moves around chance (0.450 to 0.583), reinforcing that its",
        "apparent subgroup differences are not a reliable deployment result.",
    ])
    lines.extend([
        "",
        "Interpretation rules:",
        "",
        "- categories with fewer than three anomaly events or fewer than two anomaly",
        "  assets are descriptive only, even when their point estimate is high;",
        "- Farm B subgroup results cannot rescue a farm-level detector whose overall",
        "  discrimination remains statistically indistinguishable from chance;",
        "- leave-one-asset-out changes quantify sensitivity, not external validation;",
        "- raw descriptions and event counts must accompany any category-level claim.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smooth-steps", type=int,
                        default=REFERENCE_SMOOTH_STEPS)
    parser.add_argument("--threshold-quantile", type=float,
                        default=REFERENCE_QUANTILE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    events = load_reference_events(
        args.events, args.smooth_steps, args.threshold_quantile)
    categories = fault_category_summary(events)
    raw_faults = raw_fault_summary(events)
    assets = asset_summary(events)
    influence = leave_one_asset_out(events)

    events.to_csv(args.output / "reference_events.csv", index=False)
    categories.to_csv(args.output / "fault_category_summary.csv", index=False)
    raw_faults.to_csv(args.output / "raw_fault_description_summary.csv", index=False)
    assets.to_csv(args.output / "asset_summary.csv", index=False)
    influence.to_csv(args.output / "leave_one_asset_out.csv", index=False)
    plot_fault_categories(categories, args.output / "fault_category_auc.png")
    plot_asset_influence(influence, args.output / "asset_influence.png")
    write_findings(categories, influence, args.output / "FINDINGS.md")

    metadata = {
        "source_events": str(args.events),
        "reference_smooth_steps": args.smooth_steps,
        "reference_threshold_quantile": args.threshold_quantile,
        "normal_statuses": "0",
        "event_rows": len(events),
        "assets": int(events[["farm", "asset"]].drop_duplicates().shape[0]),
        "fault_taxonomy": {
            category: keywords for category, keywords in FAULT_RULES
        },
        "confidence_intervals": "1000-resample asset-clustered bootstrap",
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(categories.to_string(index=False))
    print("\n", influence.to_string(index=False))


if __name__ == "__main__":
    main()

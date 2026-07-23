"""Report figure: one run-to-failure episode, clean vs noisy observation.

The paired-noise design means the SAME failure occurs at the SAME time at
every noise level; only the observed bearing temperature changes. This
figure shows the last 80 days before one failure on a held-out turbine at
noise_level=0.0 (clean, monotone ramp) and noise_level=1.5 (sensor noise,
false spikes, weakened ramp).

Run:  python episode_trace.py    ->  episode_trace.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_data import generate_farm

TURBINE, EPISODE, WINDOW_DAYS = 3, 1, 80


def episode_slice(noise):
    df = generate_farm(n_turbines=5, n_days=730, seed=0, noise_level=noise)
    dt = df[(df.turbine == TURBINE) & (df.episode == EPISODE)]
    fail_t = int(dt.t[dt.failed_now].iloc[0])
    lo = fail_t - WINDOW_DAYS * 24
    win = df[(df.turbine == TURBINE) & (df.t >= lo) & (df.t <= fail_t)]
    days_to_fail = (win.t.values - fail_t) / 24.0
    return days_to_fail, win.bearing_temp.values


def main():
    x0, y0 = episode_slice(0.0)
    x1, y1 = episode_slice(1.5)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, sharey=True)
    for ax, (x, y, lvl, c) in zip(axes, [(x0, y0, 0.0, "tab:blue"),
                                         (x1, y1, 1.5, "tab:red")]):
        ax.plot(x, y, lw=0.6, color=c)
        # daily mean to guide the eye
        n = len(y) // 24
        ax.plot(x[:n * 24].reshape(n, 24).mean(1),
                y[:n * 24].reshape(n, 24).mean(1),
                lw=2.0, color="black", alpha=0.7, label="daily mean")
        ax.axvline(0, ls="--", color="gray")
        ax.set_title(f"noise_level = {lvl}", fontsize=10)
        ax.set_ylabel("bearing temp (°C)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].annotate("failure", xy=(0, axes[0].get_ylim()[1]),
                     xytext=(-8, -2), textcoords="offset points",
                     ha="right", va="top", color="gray", fontsize=9)
    axes[1].set_xlabel("days to failure")
    fig.suptitle(f"Same failure (turbine {TURBINE}, episode {EPISODE}), "
                 "two observability levels", fontsize=11)
    fig.tight_layout()
    fig.savefig("episode_trace.png", dpi=150)
    print("saved: episode_trace.png")


if __name__ == "__main__":
    main()

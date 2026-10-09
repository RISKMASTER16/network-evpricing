"""
Plot the London private-vehicle charging start-time distribution
(tri-modal Gaussian, Wei et al. Eq. 3). Parameters are a heuristic
estimate; see docs/inputs/charging_start_time.md.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import PROCESSED, INPUT_FIGURES


def trimodal_pdf(t, weights, means, stds):
    """Tri-modal Gaussian mixture, wrapped around 0-24 h."""
    t = np.asarray(t, dtype=float)
    total = np.zeros_like(t)
    for w, mu, sigma in zip(weights, means, stds):
        for k in (-24, 0, 24):
            total += w / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((t - (mu + k)) ** 2) / (2 * sigma ** 2))
    return total


def bin_midpoints(intervals: pd.Series) -> np.ndarray:
    return intervals.str.split("-").str[0].astype(float).values + 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=PROCESSED / "london_private_trimodal_params.csv")
    ap.add_argument("--hourly", default=PROCESSED / "london_private_hourly_distribution.csv")
    ap.add_argument("--label", default="London private vehicle (heuristic estimate)")
    ap.add_argument("--out", default=INPUT_FIGURES / "start_time_distribution.png")
    args = ap.parse_args()

    params = pd.read_csv(args.params)
    hourly = pd.read_csv(args.hourly)
    w, mu, sd = params["weight_w"].values, params["mean_mu_h"].values, params["std_sigma_h"].values

    mids = bin_midpoints(hourly["Interval"])
    empirical = hourly["Percentage"].values / 100.0
    t = np.linspace(0, 24, 481)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mids, empirical, "x", color="tab:blue", markersize=7, label="Hourly points (heuristic table)")
    ax.plot(t, trimodal_pdf(t, w, mu, sd), "r--", linewidth=1.5)
    ax.plot(mids, trimodal_pdf(mids, w, mu, sd), "o", color="red", markersize=5,
            linestyle="none", label="Tri-modal curve (Eq. 3 form)")
    ax.set_xlabel("Time of day / h")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 24)
    ax.set_title(f"Charging start-time distribution -- {args.label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

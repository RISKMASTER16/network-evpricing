"""
Plot the London private-vehicle initial-SOC distribution (lognormal,
Wei et al. Eq. 2). Parameters are a heuristic estimate; see
docs/inputs/initial_soc.md.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from paths import PROCESSED, INPUT_FIGURES

BIN_WIDTH = 10.0


def bin_midpoints(ranges: pd.Series) -> np.ndarray:
    parts = ranges.str.split("-")
    return (parts.str[0].astype(float) + parts.str[1].astype(float)) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=PROCESSED / "london_private_soc_lognormal_params.csv")
    ap.add_argument("--bins", default=PROCESSED / "london_private_soc_bin_distribution.csv")
    ap.add_argument("--label", default="London private vehicle (heuristic estimate)")
    ap.add_argument("--out", default=INPUT_FIGURES / "initial_soc_distribution.png")
    args = ap.parse_args()

    params = pd.read_csv(args.params)
    bins = pd.read_csv(args.bins)
    dist = lognorm(s=float(params["sigma_log"].iloc[0]), scale=np.exp(float(params["mu_log"].iloc[0])))

    mids = bin_midpoints(bins["SOC_range_pct"])
    empirical = (bins["Percentage"].values / 100.0) / BIN_WIDTH
    soc = np.linspace(0.01, 100, 500)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mids, empirical, "o", color="tab:blue", markersize=7, label="Original data points (heuristic bins)")
    ax.plot(soc, dist.pdf(soc), "r--", linewidth=1.5, marker="x", markevery=15,
            markersize=5, label="Fitted curve (lognormal)")
    ax.set_xlabel("Initial SOC / %")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 90)
    ax.set_title(f"Initial SOC distribution -- {args.label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

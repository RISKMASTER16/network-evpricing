"""
Fit the London private-BEV daily-mileage distribution.

Follows the lognormal form of Wei et al. (2026), Eq. 1, fitted to DfT
National Travel Survey data (England, 2024):

  NTS0904  annual mileage bands, all cars  -> lognormal shape (sigma)
  NTS0901a annual mileage by fuel type     -> BEV mean, used to re-anchor mu

NTS0904 is not split by fuel type, so the BEV distribution borrows the
all-car shape and is re-centred on the BEV mean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import lognorm

from paths import RAW, PROCESSED, INPUT_FIGURES

MILES_TO_KM = 1.60934
RNG = np.random.default_rng(42)
YEAR = 2024

TOP_BAND_ASSUMED_CAP_MILES = 45_000  # upper edge for the open "30,000+" band

LONDON_TOTAL_LICENSED_CARS = 2_560_000  # TfL Technical Note 12
LONDON_BEV_SHARE = 0.054                # GLA "Car Emissions", Q1 2025
LONDON_BEV_FLEET = int(LONDON_TOTAL_LICENSED_CARS * LONDON_BEV_SHARE)

BAND_DEFS = [
    ("0 to 499 miles (%)", 0, 499),
    ("500 to 999 miles (%)", 500, 999),
    ("1,000 to 1,999 miles (%)", 1000, 1999),
    ("2,000 to 2,999 miles (%)", 2000, 2999),
    ("3,000 to 3,999 miles (%)", 3000, 3999),
    ("4,000 to 4,999 miles (%)", 4000, 4999),
    ("5,000 to 6,999 miles (%)", 5000, 6999),
    ("7,000 to 8,999 miles (%)", 7000, 8999),
    ("9,000 to 11,999 miles (%)", 9000, 11999),
    ("12,000 to 14,999 miles (%)", 12000, 14999),
    ("15,000 to 17,999 miles (%)", 15000, 17999),
    ("18,000 to 20,999 miles (%)", 18000, 20999),
    ("21,000 to 29,999 miles (%)", 21000, 29999),
    ("30,000 and over miles (%)", 30000, TOP_BAND_ASSUMED_CAP_MILES),
]

FUEL_COLS = {
    "Petrol": "Petrol",
    "Diesel": "Diesel",
    "Hybrid": "Hybrid [notes 4, 5]",
    "Plug-in hybrid": "Plug-in hybrid [notes 4, 6]",
    "Battery electric": "Battery electric [notes 4, 7]",
    "All cars": "All cars [note 8]",
}


def lognormal_pdf(s, mu, sigma):
    """Wei et al. Eq. 1."""
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = (1.0 / (s * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(s) - mu) ** 2) / (2 * sigma ** 2))
    return np.where(s > 0, pdf, 0.0)


def annual_to_daily(mu_annual, sigma_annual):
    """X ~ LN(mu, sigma)  =>  X/365 ~ LN(mu - ln 365, sigma)."""
    return mu_annual - np.log(365), sigma_annual


def load_mileage_bands(csv_path, year=YEAR):
    row = pd.read_csv(csv_path).query("Year == @year").iloc[0]
    records = []
    for col, lo, hi in BAND_DEFS:
        records.append({
            "band_label": col.replace(" (%)", ""),
            "band_low_miles": lo,
            "band_high_miles": hi,
            "pct_of_cars": float(row[col]),
            "mid_km": (lo + hi) / 2 * MILES_TO_KM,
            "width_km": (hi - lo + 1) * MILES_TO_KM,
        })
    table = pd.DataFrame(records)
    table["density_per_km"] = (table["pct_of_cars"] / 100.0) / table["width_km"]
    sample_col = "Unweighted sample size: all cars (number)"
    table["sample_size_all_cars"] = int(row[sample_col]) if sample_col in row.index else None
    return table


def load_fuel_type_mileage(csv_path, year=YEAR):
    row = pd.read_csv(csv_path).query("Year == @year").iloc[0]
    records = []
    for label, col in FUEL_COLS.items():
        raw = row[col]
        flagged = isinstance(raw, str) and raw.strip().startswith("[")
        miles = None if flagged else float(raw)
        records.append({
            "fuel_type": label,
            "annual_mileage_miles": miles,
            "annual_mileage_km": None if miles is None else miles * MILES_TO_KM,
            "daily_mileage_km": None if miles is None else miles * MILES_TO_KM / 365,
            "flag": raw if flagged else "",
        })
    table = pd.DataFrame(records)
    table["sample_size_all_cars"] = int(row["Unweighted sample size: all cars"])
    return table


def fit_all_cars_lognormal(bands):
    x = bands["mid_km"].values
    y = bands["density_per_km"].values
    w = bands["pct_of_cars"].values / 100.0

    mean0 = np.sum(w * x)
    var0 = np.sum(w * (x - mean0) ** 2)
    sigma0 = np.sqrt(np.log(1 + var0 / mean0 ** 2))
    mu0 = np.log(mean0) - sigma0 ** 2 / 2

    (mu, sigma), _ = curve_fit(lognormal_pdf, x, y, p0=[mu0, sigma0], maxfev=10000)
    return mu, sigma


def build_ev_daily_distribution(fuel, sigma_all):
    bev_mean_km = fuel.loc[fuel["fuel_type"] == "Battery electric", "daily_mileage_km"].iloc[0]
    mu_ev = np.log(bev_mean_km) - sigma_all ** 2 / 2
    return mu_ev, sigma_all, bev_mean_km


def build_km_band_table(mu, sigma, top_band_cap_km=200.0):
    """Percentage of vehicles per 10 km band, in the layout of Wei et al. Table 1."""
    dist = lognorm(s=sigma, scale=np.exp(mu))
    edges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60),
             (60, 70), (70, 80), (80, 90), (90, 100), (100, top_band_cap_km)]
    labels = ["<10", "10-20", "20-30", "30-40", "40-50", "50-60",
              "60-70", "70-80", "80-90", "90-100", "above 100"]
    return pd.DataFrame([
        {"band_label": lab, "band_low_km": lo, "band_high_km": hi,
         "pct_of_vehicles": round((dist.cdf(hi) - dist.cdf(lo)) * 100.0, 2)}
        for (lo, hi), lab in zip(edges, labels)
    ])


def save_lognormal_params(mu, sigma, path):
    params = pd.DataFrame([{
        "distribution": "lognormal",
        "mu_log": round(mu, 4),
        "sigma_log": round(sigma, 4),
        "median_km": round(np.exp(mu), 2),
        "mean_km": round(np.exp(mu + sigma ** 2 / 2), 2),
    }])
    params.to_csv(path, index=False)
    return params


def plot_fits(bands, mu_all, sigma_all, mu_ev, sigma_ev, bev_mean_km, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    x = np.linspace(200, bands["mid_km"].max() * 1.3, 800)
    ax.plot(x, lognormal_pdf(x, mu_all, sigma_all), color="crimson", lw=1.8,
            label="Fitted curve (log-normal)")
    ax.scatter(bands["mid_km"], bands["density_per_km"], color="steelblue", zorder=5,
               label="Original data points (NTS0904)")
    ax.set_title(f"(a) All cars — annual mileage, England {YEAR}\n"
                 f"mu={mu_all:.3f}, sigma={sigma_all:.3f}")
    ax.set_xlabel("Annual mileage / km")
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.linspace(1, 200, 800)
    y = lognormal_pdf(x, mu_ev, sigma_ev)
    ax.plot(x, y, color="darkgreen", lw=2,
            label="EV daily-mileage log-normal\n(shape from (a), mean anchored to BEV data)")
    ax.axvline(bev_mean_km, color="black", ls="--", lw=1.2,
               label=f"BEV mean = {bev_mean_km:.1f} km/day\n(NTS0901a, {YEAR})")
    ax.fill_between(x, y, color="darkgreen", alpha=0.1)
    ax.set_title(f"(b) Battery-electric cars — daily mileage, England {YEAR}\n"
                 f"mu={mu_ev:.3f}, sigma={sigma_ev:.3f}")
    ax.set_xlabel("Daily mileage / km")
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=8)

    fig.suptitle("Log-normal daily-mileage fits (Eq. 1 form), "
                 "England National Travel Survey data, 2024", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fleet_total(mu, sigma, out_path, n_samples=200_000):
    """Fleet-total daily mileage via a CLT approximation to the sum of per-vehicle draws."""
    mean_v = np.exp(mu + sigma ** 2 / 2)
    var_v = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
    samples = RNG.normal(LONDON_BEV_FLEET * mean_v, np.sqrt(LONDON_BEV_FLEET * var_v), size=n_samples)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(samples / 1000, bins=80, color="darkorange", alpha=0.85, density=True)
    ax.set_xlabel("Total London BEV fleet daily mileage / '000 km per day")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Estimated London BEV fleet daily mileage, {YEAR}\n"
                 f"(~{LONDON_BEV_FLEET:,} BEVs, CLT approximation of summed per-vehicle draws)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return samples


def main():
    bands = load_mileage_bands(RAW / "nts0904_mileage_bands.csv")
    fuel = load_fuel_type_mileage(RAW / "nts0901a_fuel_type_mileage.csv")

    print(f"NTS0904 annual mileage bands, all cars, England {YEAR}")
    print(bands[["band_label", "pct_of_cars", "mid_km", "density_per_km"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nNTS0901a annual mileage by fuel type, England {YEAR}")
    print(fuel[["fuel_type", "annual_mileage_miles", "annual_mileage_km", "daily_mileage_km", "flag"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    mu_all, sigma_all = fit_all_cars_lognormal(bands)
    mu_d, sigma_d = annual_to_daily(mu_all, sigma_all)
    print(f"\nAll-cars fit (annual, km): mu={mu_all:.4f}, sigma={sigma_all:.4f}")
    print(f"All-cars daily mean: {np.exp(mu_d + sigma_d ** 2 / 2):.2f} km/day")

    mu_ev, sigma_ev, bev_mean_km = build_ev_daily_distribution(fuel, sigma_all)
    print(f"\nEV daily distribution: mu={mu_ev:.4f}, sigma={sigma_ev:.4f}")
    print(f"EV daily mean (NTS0901a anchor): {bev_mean_km:.2f} km/day")

    plot_fits(bands, mu_all, sigma_all, mu_ev, sigma_ev, bev_mean_km,
              INPUT_FIGURES / "daily_mileage_fits.png")

    fleet = plot_fleet_total(mu_ev, sigma_ev, INPUT_FIGURES / "fleet_daily_mileage.png")
    print(f"\nLondon BEV fleet: {LONDON_TOTAL_LICENSED_CARS:,} cars x {LONDON_BEV_SHARE:.1%} "
          f"= ~{LONDON_BEV_FLEET:,} BEVs")
    print(f"Fleet daily mileage: mean {fleet.mean() / 1000:,.0f}k km, "
          f"p5-p95 {np.percentile(fleet, 5) / 1000:,.0f}-{np.percentile(fleet, 95) / 1000:,.0f}k km")

    bands.to_csv(PROCESSED / "table1_mileage_bands_2024.csv", index=False)
    fuel.to_csv(PROCESSED / "table2_fuel_type_mileage_2024.csv", index=False)

    km_bands = build_km_band_table(mu_ev, sigma_ev)
    km_bands.to_csv(PROCESSED / "london_private_mileage_km_bands.csv", index=False)
    print("\nLondon private BEV daily mileage by 10 km band")
    print(km_bands.to_string(index=False))

    params = save_lognormal_params(mu_ev, sigma_ev, PROCESSED / "london_private_mileage_lognormal_params.csv")
    print("\n" + params.to_string(index=False))


if __name__ == "__main__":
    main()

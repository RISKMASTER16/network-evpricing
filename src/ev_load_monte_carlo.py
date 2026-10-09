"""
Step 1: London private-BEV charging load by Monte Carlo simulation.

Method: Wei et al. (2026), Global Energy Interconnection 9, 131-142,
Eqs. 1-10. Each session replenishes ~1/k days of driving (Eq. 6,
Ec = Ed / k), so only N*k vehicles charge on a given day.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paths import PROCESSED, FIGURES, LOAD_PROFILE_CSV

rng = np.random.default_rng(seed=42)

KM_BANDS_CSV = PROCESSED / "london_private_mileage_km_bands.csv"
PLOT_PNG = FIGURES / "london_ev_load_profile.png"

# Fitted London distributions (see data/processed and docs/inputs)
MILEAGE_MU, MILEAGE_SIGMA = 3.4048, 0.7279  # lognormal, km/day
SOC_MU, SOC_SIGMA = 3.37, 0.48              # lognormal, % of capacity
TRIMODAL = pd.DataFrame({
    "weight": [0.04, 0.30, 0.66],
    "mean_h": [0.5, 10.0, 18.5],
    "std_h": [0.4, 2.4, 3.6],
})

N_VEHICLES = 138_240  # 2.56M London cars (TfL TN12) x 5.4% BEV share (GLA, Q1 2025)
W100 = 18.0           # kWh/100 km
K_PER_DAY = 0.214     # sessions/vehicle/day, ~1.5 per week (DfT EV driver survey)
FAST_SHARE = 0.40     # share of rapid/public sessions (~40% of Londoners lack home charging)
P_FAST_KW = 50.0
P_SLOW_KW = 7.0
ETA = 0.92

DT_HOURS = 1 / 60
N_MC_RUNS = 1000


def sample_daily_mileage(n):
    return np.clip(rng.lognormal(mean=MILEAGE_MU, sigma=MILEAGE_SIGMA, size=n), 0.5, 400.0)


def sample_initial_soc(n):
    return np.clip(rng.lognormal(mean=SOC_MU, sigma=SOC_SIGMA, size=n), 1.0, 100.0)


def sample_start_time(n):
    comp = rng.choice(len(TRIMODAL), size=n, p=TRIMODAL["weight"].values)
    t = rng.normal(loc=TRIMODAL["mean_h"].values[comp], scale=TRIMODAL["std_h"].values[comp])
    return np.mod(t, 24.0)


def simulate_one_day():
    n_sessions = rng.poisson(N_VEHICLES * K_PER_DAY)

    mileage = sample_daily_mileage(n_sessions)
    soc = sample_initial_soc(n_sessions) / 100.0
    start_h = sample_start_time(n_sessions)

    e_d = (mileage / 100.0) * W100            # Eq. 5
    e_c = e_d / K_PER_DAY                     # Eq. 6
    e_n = np.clip((1.0 - soc) * e_c, 0.2, None)  # Eq. 7

    is_fast = rng.random(n_sessions) < FAST_SHARE  # Eq. 8
    power_kw = np.where(is_fast, P_FAST_KW, P_SLOW_KW)
    duration_h = e_n / (power_kw * ETA)       # Eq. 9

    return pd.DataFrame({"start_h": start_h, "duration_h": duration_h, "power_kw": power_kw})


def aggregate_load(events, dt_hours=DT_HOURS):
    """Eq. 10, with wraparound at midnight."""
    n_steps = int(round(24 / dt_hours))
    load_mw = np.zeros(n_steps)

    start_idx = np.floor(events["start_h"].values / dt_hours).astype(int)
    n_slots = np.clip(np.ceil(events["duration_h"].values / dt_hours).astype(int), 1, n_steps)
    power_mw = events["power_kw"].values / 1000.0

    for s_idx, n_slot, p in zip(start_idx, n_slots, power_mw):
        load_mw[(s_idx + np.arange(n_slot)) % n_steps] += p
    return load_mw


def run_monte_carlo(n_runs=N_MC_RUNS):
    n_steps = int(round(24 / DT_HOURS))
    runs = np.zeros((n_runs, n_steps))
    session_counts = np.zeros(n_runs)
    for r in range(n_runs):
        events = simulate_one_day()
        session_counts[r] = len(events)
        runs[r] = aggregate_load(events)
    return runs, session_counts


def validate_mileage_fit(km_bands_csv, n_check=200_000):
    bands = pd.read_csv(km_bands_csv)
    sim = sample_daily_mileage(n_check)
    print("Mileage distribution check (survey vs simulated %):")
    for _, row in bands.iterrows():
        pct = 100 * np.mean((sim >= row["band_low_km"]) & (sim < row["band_high_km"]))
        print(f"  {row['band_label']:>10}: survey {row['pct_of_vehicles']:5.2f}% | simulated {pct:5.2f}%")


def expected_daily_energy_mwh():
    mean_mileage = np.exp(MILEAGE_MU + MILEAGE_SIGMA ** 2 / 2)
    mean_soc = np.exp(SOC_MU + SOC_SIGMA ** 2 / 2)
    e_d = mean_mileage / 100 * W100
    return N_VEHICLES * e_d * (1 - mean_soc / 100) / ETA / 1000


def main():
    validate_mileage_fit(KM_BANDS_CSV)

    print(f"\nFleet: {N_VEHICLES:,} BEVs | k={K_PER_DAY:.3f} -> ~{N_VEHICLES * K_PER_DAY:,.0f} sessions/day")
    print(f"Running {N_MC_RUNS} Monte Carlo days...")
    runs, session_counts = run_monte_carlo()

    mean_profile = runs.mean(axis=0)
    p10 = np.percentile(runs, 10, axis=0)
    p90 = np.percentile(runs, 90, axis=0)
    hours = np.arange(len(mean_profile)) * DT_HOURS

    peak, trough = mean_profile.max(), mean_profile.min()
    print(f"\nSessions per day: {session_counts.mean():.0f}")
    print(f"Peak load: {peak:.1f} MW at {hours[np.argmax(mean_profile)]:.2f} h")
    print(f"Trough load: {trough:.1f} MW")
    print(f"Peak-to-trough spread: {peak - trough:.1f} MW")

    total = mean_profile.sum() * DT_HOURS
    expected = expected_daily_energy_mwh()
    print(f"Daily grid energy: {total:.0f} MWh (expected {expected:.0f} MWh, "
          f"diff {100 * (total - expected) / expected:+.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(hours, p10, p90, color="tab:red", alpha=0.15, label="10th-90th pct")
    ax.plot(hours, mean_profile, color="tab:red", lw=1.8, label="Mean simulated load")
    ax.set_xlabel("Time of day (h)")
    ax.set_ylabel("Charging load (MW)")
    ax.set_title("London Private BEV Charging Load — Monte Carlo Simulation\n"
                 "(methodology: Wei et al. 2026; London 2024 parameters)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)

    pd.DataFrame({"hour": hours, "mean_load_mw": mean_profile, "p10_mw": p10, "p90_mw": p90}) \
        .to_csv(LOAD_PROFILE_CSV, index=False)
    print(f"\nSaved {PLOT_PNG} and {LOAD_PROFILE_CSV}")


if __name__ == "__main__":
    main()

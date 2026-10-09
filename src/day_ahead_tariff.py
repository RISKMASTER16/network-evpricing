"""
Step 2: build one day-ahead DToU + CPP tariff for a London LV feeder,
then simulate a realised day with the RTP settlement adjustment.

Method: Khan & Aziz (2022), eqs. 1-8 (day-ahead) and 9-14 (real-time),
adapted from a centralised charging hub to distributed home charging.
See docs/METHODOLOGY.md, section 3.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import feeder_model as fm
import pricing as pr
from feeder_model import SP, DT_H, HOURS
from paths import LOAD_PROFILE_CSV, RESULTS, FIGURES

SUBSTATION_KVA = 500.0
FEEDERS_PER_SUB = 4
HOUSEHOLDS_PER_FEEDER = 75
POWER_FACTOR = 0.95
PLANNING_UTILISATION = 0.85   # 1.00 = nameplate, 0.85 = constrained case

CHARGER_KW = 7.4
EV_PENETRATION = 0.60
SMART_FRACTION = 0.80
SEASONAL_FACTOR = 1.35        # 1.35 winter peak week, 1.10 winter weekday

PROFIT_MARGIN = 1.10
CPP_LOW = 0.80
CPP_HIGH = 1.00
CPP_PENALTY_MULTIPLE = 1.0
CPP_QUANTILE = 0.90
RTP_N = 0.50


def build_tariff(verbose=True, seed=1):
    rng = np.random.default_rng(seed)

    feeder = fm.Feeder(n_households=HOUSEHOLDS_PER_FEEDER,
                       substation_kva=SUBSTATION_KVA,
                       feeders_per_substation=FEEDERS_PER_SUB,
                       power_factor=POWER_FACTOR,
                       planning_utilisation=PLANNING_UTILISATION)
    limit = feeder.firm_limit_kw

    params, _, _, _ = fm.calibrate_to_step1(LOAD_PROFILE_CSV, verbose=verbose)
    params.charger_kw = CHARGER_KW
    n_ev = fm.feeder_ev_count(feeder, EV_PENETRATION)
    background = fm.background_profile_kw(feeder, SEASONAL_FACTOR)

    cfg = pr.DToUConfig(profit_margin=PROFIT_MARGIN, cpp_low=CPP_LOW, cpp_high=CPP_HIGH,
                        cpp_penalty_multiple=CPP_PENALTY_MULTIPLE,
                        cpp_quantile=CPP_QUANTILE, rtp_n=RTP_N)

    def sampler():
        return fm.sample_sessions(params, n_ev, rng, SMART_FRACTION)

    da = pr.dtou_equilibrium(background, sampler, limit, CHARGER_KW, rng, cfg=cfg)

    if verbose:
        print(f"\nFeeder: {feeder.n_households} households, "
              f"{feeder.rating_kva:.0f} kVA, firm limit {limit:.1f} kW")
        print(f"Background peak {background.max():.1f} kW ({background.max() / limit:.0%} of limit)")
        print(f"EVs on feeder {n_ev:.0f} at {EV_PENETRATION:.0%} penetration")
        print(f"DToU coefficient C1 = {da['C1']:.4g}")
        print(f"Converged in {len(da['convergence'])} iterations "
              f"(final delta {da['convergence'][-1]:.4g} kW)")

    return dict(feeder=feeder, params=params, limit=limit,
                background=background, da=da, cfg=cfg, rng=rng, n_ev=n_ev)


def real_time_day(ctx, rt_mode="stochastic", scenario="p90", seed=None):
    """
    One realised day and its RTP-adjusted settlement price.

    rt_mode "stochastic": background drawn from the session model
            "percentile": background scaled by the step-1 P10/P90 band
    """
    rng = ctx["rng"] if seed is None else np.random.default_rng(seed)
    feeder, params, limit = ctx["feeder"], ctx["params"], ctx["limit"]
    da, cfg = ctx["da"], ctx["cfg"]

    if rt_mode == "percentile":
        raw = pd.read_csv(LOAD_PROFILE_CSV)
        slot = np.clip((raw["hour"].astype(float) * 2).astype(int), 0, SP - 1)
        col = "p10_mw" if scenario == "p10" else "p90_mw"
        band = raw.groupby(slot)[col].mean().values
        mean = raw.groupby(slot)["mean_load_mw"].mean().values
        bg_rt = ctx["background"] * (band / mean)
    else:
        bg_rt = fm.background_scenarios(feeder, 1, rng, SEASONAL_FACTOR)[0]

    sessions = fm.sample_sessions(params, ctx["n_ev"], rng, SMART_FRACTION)
    ev = pr.schedule_day(sessions, da["price"], rng)
    total = bg_rt + ev

    _, _, n_acs_rt = pr.surplus_and_capacity(bg_rt, limit, params.charger_kw)
    rtp = pr.rtp_adjust(da["T_dtou"], da["n_acs"], n_acs_rt, n=cfg.rtp_n)
    final = np.clip(da["price"] + rtp, 0.0, None)

    return pd.DataFrame({
        "slot": np.arange(SP),
        "hour": HOURS,
        "duos_band": [pr.duos_band(h) for h in HOURS],
        "background_forecast_kW": ctx["background"],
        "background_realtime_kW": bg_rt,
        "headroom_kW": da["headroom_kw"],
        "N_ACs_dayahead": da["n_acs"],
        "N_ACs_realtime": n_acs_rt,
        "forecast_EV_kW": da["ev_forecast_kw"],
        "forecast_EV_p90_kW": da["ev_tail_forecast_kw"],
        "headroom_utilisation": da["utilisation"],
        "T_DToU_p_per_kWh": da["T_dtou"],
        "CPP_p_per_kWh": da["CPP"],
        "CPP_status": da["status"],
        "dayahead_price_p_per_kWh": da["price"],
        "RTP_component_p_per_kWh": rtp,
        "final_settlement_p_per_kWh": final,
        "realised_EV_kW": ev,
        "realised_total_kW": total,
    })


def plot(ctx, rt, path):
    da, limit = ctx["da"], ctx["limit"]
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.1]})

    ax[0].fill_between(HOURS, 0, rt["background_forecast_kW"],
                       color="#c9c9c9", label="Non-EV background (forecast)")
    ax[0].fill_between(HOURS, rt["background_forecast_kW"],
                       rt["background_forecast_kW"] + da["ev_forecast_kw"],
                       color="#2ca02c", alpha=0.6, label="Forecast EV (converged equilibrium)")
    ax[0].plot(HOURS, rt["realised_total_kW"], color="k", lw=1.3, label="Realised total (one day)")
    ax[0].axhline(limit, color="crimson", ls="--", lw=1.4, label=f"Feeder firm limit {limit:.0f} kW")
    ax[0].set_ylabel("kW")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("Adapted Khan & Aziz day-ahead scheme on a London LV feeder")

    ax[1].step(HOURS, pr.static_tou_price(), where="post", color="#1f77b4",
               ls="--", lw=1.2, label="Static ToU reference (Octopus Go)")
    ax[1].step(HOURS, da["T_dtou"], where="post", color="#2ca02c", lw=1.5,
               label="$T_{DToU}$  (eqs. 5-7)")
    ax[1].step(HOURS, da["price"], where="post", color="k", lw=1.8,
               label="$T_{DToU}$ + CPP, published day-ahead (eq. 8)")
    ax[1].step(HOURS, rt["final_settlement_p_per_kWh"], where="post", color="#d62728", lw=1.2,
               label="Final settlement incl. RTP (eqs. 12-14)")
    on = da["CPP"] > 0
    if on.any():
        ax[1].scatter(HOURS[on], da["price"][on], color="crimson", s=26, zorder=5, label="CPP tier active")
    ax[1].set_ylabel("p/kWh")
    ax[1].set_xlabel("Hour of day")
    ax[1].set_xticks(range(0, 25, 2))
    ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rt-mode", choices=["stochastic", "percentile"], default="stochastic")
    ap.add_argument("--scenario", choices=["p10", "p90"], default="p90")
    args = ap.parse_args()

    ctx = build_tariff()
    rt = real_time_day(ctx, rt_mode=args.rt_mode, scenario=args.scenario)

    show = ["hour", "duos_band", "background_forecast_kW", "headroom_kW",
            "N_ACs_dayahead", "forecast_EV_kW", "headroom_utilisation",
            "T_DToU_p_per_kWh", "CPP_p_per_kWh", "dayahead_price_p_per_kWh"]
    print("\nDay-ahead DToU + CPP tariff")
    print(rt[show].round(3).to_string(index=False))

    show_rt = ["hour", "background_realtime_kW", "N_ACs_dayahead", "N_ACs_realtime",
               "dayahead_price_p_per_kWh", "RTP_component_p_per_kWh",
               "final_settlement_p_per_kWh", "realised_total_kW"]
    print(f"\nReal-time adjustment ({args.rt_mode})")
    print(rt[show_rt].round(3).to_string(index=False))

    out_csv = RESULTS / "day_ahead_tariff.csv"
    rt.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    e = rt["realised_EV_kW"] * DT_H
    peak = rt["realised_total_kW"].max()
    print(f"\nEV energy delivered   : {e.sum():.1f} kWh")
    print(f"Mean settlement price : {(rt['final_settlement_p_per_kWh'] * e).sum() / max(e.sum(), 1e-9):.2f} p/kWh")
    print(f"Realised peak         : {peak:.1f} kW ({peak / ctx['limit']:.0%} of limit)")
    print(f"Slots with CPP active : {int((ctx['da']['CPP'] > 0).sum())}")
    print(f"Slots curtailed       : {int((ctx['da']['status'] == 'curtailed').sum())}")

    plot(ctx, rt, FIGURES / "day_ahead_tariff.png")


if __name__ == "__main__":
    main()

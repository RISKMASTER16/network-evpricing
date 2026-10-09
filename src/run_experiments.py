"""
Step 3: Monte Carlo comparison of pricing schemes on a London LV feeder.

Network cases
  base         winter weekday, feeder at nameplate (118.8 kW)
  constrained  winter peak week, 0.85 planning derate (101.0 kW)

Operator margin is reported as published and revenue-neutral (all
schemes credited at the Ofgem cap rate), so the latter isolates the
effect of when energy is bought.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import feeder_model as fm
import pricing as pr
from feeder_model import SP, DT_H
from paths import LOAD_PROFILE_CSV, RESULTS

CSV_PATH = LOAD_PROFILE_CSV
N_DAYS = 2000
PENETRATIONS = [0.03, 0.20, 0.40, 0.60, 0.80, 1.00]
SCHEMES = ["uncontrolled", "static_tou", "agile", "proposed", "proposed_tail"]
SCHEME_LABEL = {
    "uncontrolled": "Uncontrolled (flat, Ofgem cap)",
    "static_tou": "Static ToU (Octopus Go)",
    "agile": "Dynamic RTP (Octopus Agile)",
    "proposed": "Proposed DToU + CPP + RTP",
    "proposed_tail": "Proposed, tail-aware CPP (q=0.90)",
    "cvar_opt": "CVaR-constrained optimal",
}

NETWORK_CASES = {
    "base": dict(seasonal=1.10, utilisation=1.00),
    "constrained": dict(seasonal=1.35, utilisation=0.85),
}


def var_cvar(x: np.ndarray, q: float):
    v = float(np.quantile(x, q))
    tail = x[x >= v]
    return v, float(tail.mean()) if tail.size else v


def run_case(case_name: str, smart_fraction=0.80, timer_fraction=0.75,
             seed=20260827, n_days=N_DAYS, penetrations=None, verbose=True):
    cfgc = NETWORK_CASES[case_name]
    rng = np.random.default_rng(seed)
    feeder = fm.Feeder(planning_utilisation=cfgc["utilisation"])
    penetrations = penetrations or PENETRATIONS
    params, _, _, diag = fm.calibrate_to_step1(CSV_PATH, verbose=verbose)

    limit = feeder.firm_limit_kw
    season = cfgc["seasonal"]
    bg_days = fm.background_scenarios(feeder, n_days, rng, season)
    bg_forecast = fm.background_profile_kw(feeder, season)
    wholesale = pr.sample_wholesale(n_days, rng)
    agile_days = pr.agile_from_wholesale(wholesale)

    price_flat = pr.flat_price()
    price_tou = pr.static_tou_price()

    rows, profiles, tariffs = [], {}, {}

    if verbose:
        print(f"\nNetwork case '{case_name}': limit {limit:.1f} kW, "
              f"background peak {bg_forecast.max():.1f} kW "
              f"({bg_forecast.max() / limit:.0%} of limit), "
              f"{bg_forecast.max() / feeder.n_households:.2f} kW/household")

    for pen in penetrations:
        n_ev = fm.feeder_ev_count(feeder, pen)

        def sampler(_n=n_ev):
            return fm.sample_sessions(params, _n, rng, smart_fraction)

        da = pr.dtou_equilibrium(bg_forecast, sampler, limit, params.charger_kw, rng,
                                 cfg=pr.DToUConfig(cpp_quantile=0.50),
                                 timer_fraction=timer_fraction)
        da_t = pr.dtou_equilibrium(bg_forecast, sampler, limit, params.charger_kw, rng,
                                   cfg=pr.DToUConfig(cpp_quantile=0.90),
                                   timer_fraction=timer_fraction)
        n_acs_da = da["n_acs"]
        tariffs[pen] = {"point": da, "tail": da_t}
        scheme_price = {"uncontrolled": lambda d: price_flat,
                        "static_tou": lambda d: price_tou,
                        "agile": lambda d: agile_days[d],
                        "proposed": lambda d: da["price"],
                        "proposed_tail": lambda d: da_t["price"]}

        for scheme in SCHEMES:
            peaks = np.zeros(n_days)
            energies = np.zeros(n_days)
            ccost = np.zeros(n_days)
            margin = np.zeros(n_days)
            scost = np.zeros(n_days)
            exceed = np.zeros(n_days)
            evpk = np.zeros(n_days)
            nsess = np.zeros(n_days)
            mean_prof = np.zeros(SP)
            mean_ev = np.zeros(SP)

            for d in range(n_days):
                sf = 0.0 if scheme == "uncontrolled" else smart_fraction
                s = fm.sample_sessions(params, n_ev, rng, sf)
                nsess[d] = len(s)
                price = scheme_price[scheme](d)

                ev = pr.schedule_day(s, price, rng, timer_fraction)
                total = bg_days[d] + ev

                if scheme.startswith("proposed"):
                    _, _, n_acs_rt = pr.surplus_and_capacity(bg_days[d], limit, params.charger_kw)
                    rtp = pr.rtp_adjust(da["T_dtou"], n_acs_da, n_acs_rt)
                    bill_price = np.clip(price + rtp, 0.0, None)
                else:
                    bill_price = price

                res = pr.settle(ev, bill_price, wholesale[d])
                peaks[d] = total.max()
                evpk[d] = ev.max()
                energies[d] = res["energy_kwh"]
                ccost[d] = res["consumer_cost_p"]
                margin[d] = res["operator_margin_p"]
                scost[d] = res["supply_cost_p"]
                exceed[d] = float(np.clip(total - limit, 0, None).sum() * DT_H)
                mean_prof += total
                mean_ev += ev

            mean_prof /= n_days
            mean_ev /= n_days
            v95, c95 = var_cvar(peaks, 0.95)
            v99, c99 = var_cvar(peaks, 0.99)
            ok = energies > 1e-6
            p_per_kwh = np.where(ok, ccost / np.maximum(energies, 1e-9), np.nan)
            margin_rn = pr.OFGEM_CAP_P_PER_KWH * energies - scost

            rows.append(dict(
                case=case_name, penetration=pen, scheme=scheme,
                label=SCHEME_LABEL[scheme], n_ev=n_ev,
                limit_kw=limit,
                mean_sessions=nsess.mean(),
                mean_peak_kw=peaks.mean(), max_peak_kw=peaks.max(),
                mean_load_kw=mean_prof.mean(),
                PAR=mean_prof.max() / mean_prof.mean(),
                ev_peak_kw=evpk.mean(),
                p_overload=float((peaks > limit).mean()),
                VaR95_kw=v95, VaR99_kw=v99, CVaR95_kw=c95, CVaR99_kw=c99,
                CVaR95_util=c95 / limit,
                exceed_kwh_mean=exceed.mean(),
                exceed_kwh_p99=float(np.quantile(exceed, 0.99)),
                energy_kwh_day=energies.mean(),
                consumer_p_per_kwh=float(np.nanmean(p_per_kwh)),
                consumer_p_per_kwh_sd=float(np.nanstd(p_per_kwh)),
                consumer_p_per_kwh_p95=float(np.nanquantile(p_per_kwh, 0.95)),
                consumer_cost_p_day=ccost.mean(),
                consumer_cost_sd_p_day=ccost.std(),
                margin_p_day=margin.mean(), margin_sd_p_day=margin.std(),
                margin_sharpe=(margin.mean() / margin.std() if margin.std() > 0 else np.nan),
                margin_rn_p_day=margin_rn.mean(),
                margin_rn_sd_p_day=margin_rn.std(),
                margin_rn_sharpe=(margin_rn.mean() / margin_rn.std() if margin_rn.std() > 0 else np.nan),
                supply_cost_p_per_kwh=float(scost.sum() / max(energies.sum(), 1e-9)),
            ))
            profiles[(case_name, pen, scheme)] = (mean_prof, mean_ev)
            if verbose:
                print(f"  pen={pen:>4.0%} {scheme:<13s} "
                      f"peak {peaks.mean():6.1f}  CVaR95 {c95:6.1f} "
                      f"({c95 / limit:5.0%})  P(ovl) {(peaks > limit).mean():6.2%}  "
                      f"cons {np.nanmean(p_per_kwh):5.1f}p  "
                      f"buy {scost.sum() / max(energies.sum(), 1e-9):5.1f}p")

    return dict(df=pd.DataFrame(rows), profiles=profiles, tariffs=tariffs, feeder=feeder,
                params=params, diag=diag, wholesale=wholesale,
                agile=agile_days, bg_forecast=bg_forecast, limit=limit,
                bg_days=bg_days, prices=dict(flat=price_flat, static_tou=price_tou))


def run_all(verbose=True):
    outs, frames = {}, []
    for case in NETWORK_CASES:
        outs[case] = run_case(case, verbose=verbose)
        frames.append(outs[case]["df"])
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "metrics_by_scheme.csv", index=False)
    outs["all"] = df
    return outs


if __name__ == "__main__":
    df = run_all()["all"]
    print("\nBase case, 60% penetration")
    cols = ["scheme", "mean_peak_kw", "CVaR95_kw", "p_overload",
            "consumer_p_per_kwh", "margin_rn_p_day", "margin_rn_sharpe"]
    print(df[(df.case == "base") & (df.penetration == 0.60)][cols].round(3).to_string(index=False))

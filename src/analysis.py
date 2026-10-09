"""
Efficient frontier, CVaR-constrained optimal pricing, and a 365-day
seasonal backtest.

The optimisation searches a family that nests the compared schemes:

    price_i = (1 - w) * cost_signal_i + w * network_signal_i + CPP(tau, lam)_i

w = 0 is pure cost-following (Agile-like), w = 1 pure demand-following
(the proposed DToU).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import feeder_model as fm
import pricing as pr
import run_experiments as rx
from feeder_model import SP, DT_H
from paths import RESULTS

MONTH_SEASONAL = np.array([1.35, 1.32, 1.22, 1.08, 0.95, 0.86,
                           0.83, 0.84, 0.93, 1.08, 1.22, 1.33])
MONTH_WHOLESALE = np.array([10.5, 10.0, 9.0, 7.6, 6.6, 6.0,
                            5.9, 6.2, 7.2, 8.6, 9.8, 10.6])
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def efficient_frontier(df: pd.DataFrame, case: str, penetration: float,
                       baseline: str = "uncontrolled") -> pd.DataFrame:
    d = df[(df.case == case) & (df.penetration == penetration)].set_index("scheme")
    b = d.loc[baseline]
    out = []
    for s, r in d.iterrows():
        out.append(dict(
            scheme=s, label=r["label"],
            CVaR95_kw=r.CVaR95_kw,
            risk_reduction_pct=100 * (b.CVaR95_kw - r.CVaR95_kw) / b.CVaR95_kw,
            overload_reduction_pct=(100 * (b.p_overload - r.p_overload) / b.p_overload
                                    if b.p_overload > 0 else np.nan),
            consumer_cost_change_pct=100 * (r.consumer_p_per_kwh - b.consumer_p_per_kwh)
                                     / b.consumer_p_per_kwh,
            consumer_p95_change_pct=100 * (r.consumer_p_per_kwh_p95 - b.consumer_p_per_kwh_p95)
                                    / b.consumer_p_per_kwh_p95,
            operator_margin_change_pct=100 * (r.margin_rn_p_day - b.margin_rn_p_day)
                                       / b.margin_rn_p_day,
            margin_sharpe=r.margin_rn_sharpe,
            consumer_cost_sd=r.consumer_p_per_kwh_sd,
        ))
    f = pd.DataFrame(out)

    pareto = []
    for i, r in f.iterrows():
        dominated = False
        for j, q in f.iterrows():
            if j == i:
                continue
            weakly_better = (q.CVaR95_kw <= r.CVaR95_kw and
                             q.consumer_cost_change_pct <= r.consumer_cost_change_pct)
            strictly_better = (q.CVaR95_kw < r.CVaR95_kw or
                               q.consumer_cost_change_pct < r.consumer_cost_change_pct)
            if weakly_better and strictly_better:
                dominated = True
                break
        pareto.append(not dominated)
    f["pareto_efficient"] = pareto
    return f


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)


def blended_price(cost_signal, network_signal, w, utilisation, tau, lam,
                  peak_static=pr.GO_PEAK_P, curtail_p=200.0) -> np.ndarray:
    base = ((1 - w) * _normalise(cost_signal) + w * _normalise(network_signal)) * peak_static
    cpp = np.zeros(SP)
    cpp[(utilisation >= tau) & (utilisation < 1.0)] = lam * peak_static
    cpp[utilisation >= 1.0] = curtail_p
    return base + cpp


def evaluate_policy(w, tau, lam, *, params, feeder, limit, bg_forecast,
                    bg_days, wholesale, n_ev, rng, n_days,
                    smart_fraction=0.80, timer_fraction=0.75,
                    n_iter=18, n_scen=15):
    """Solve the equilibrium price for (w, tau, lam), then Monte Carlo it."""
    surplus, _, _ = pr.surplus_and_capacity(bg_forecast, limit, params.charger_kw)
    headroom = np.clip(surplus, 1e-6, None)
    cost_signal = wholesale.mean(axis=0) + pr.DUOS_VECTOR

    ev_forecast = np.zeros(SP)
    for _ in range(n_iter):
        price = blended_price(cost_signal, bg_forecast, w, ev_forecast / headroom, tau, lam)
        induced = np.zeros(SP)
        for _ in range(n_scen):
            s = fm.sample_sessions(params, n_ev, rng, smart_fraction)
            induced += pr.schedule_day(s, price, rng, timer_fraction)
        induced /= n_scen
        ev_forecast = 0.65 * ev_forecast + 0.35 * induced

    price = blended_price(cost_signal, bg_forecast, w, ev_forecast / headroom, tau, lam)

    peaks = np.zeros(n_days)
    margins = np.zeros(n_days)
    energies = np.zeros(n_days)
    ccost = np.zeros(n_days)
    for d in range(n_days):
        s = fm.sample_sessions(params, n_ev, rng, smart_fraction)
        ev = pr.schedule_day(s, price, rng, timer_fraction)
        peaks[d] = (bg_days[d] + ev).max()
        e = ev * DT_H
        energies[d] = e.sum()
        sc = float((pr.supply_cost_p_per_kwh(wholesale[d]) * e).sum())
        margins[d] = pr.OFGEM_CAP_P_PER_KWH * energies[d] - sc
        ccost[d] = float((price * e).sum())

    _, cvar = rx.var_cvar(peaks, 0.95)
    return dict(w=w, tau=tau, lam=lam,
                margin=float(margins.mean()),
                margin_sd=float(margins.std()),
                sharpe=float(margins.mean() / margins.std()) if margins.std() > 0 else np.nan,
                cvar95=cvar,
                p_overload=float((peaks > limit).mean()),
                mean_peak=float(peaks.mean()),
                consumer_p_per_kwh=float(ccost.sum() / max(energies.sum(), 1e-9)),
                price=price, ev_forecast=ev_forecast)


def optimise_cvar(case="constrained", penetration=0.60, n_days=500, seed=7, verbose=True):
    cfgc = rx.NETWORK_CASES[case]
    rng = np.random.default_rng(seed)
    feeder = fm.Feeder(planning_utilisation=cfgc["utilisation"])
    params, *_ = fm.calibrate_to_step1(rx.CSV_PATH, verbose=False)
    limit = feeder.firm_limit_kw
    bg_forecast = fm.background_profile_kw(feeder, cfgc["seasonal"])
    bg_days = fm.background_scenarios(feeder, n_days, rng, cfgc["seasonal"])
    wholesale = pr.sample_wholesale(n_days, rng)
    n_ev = fm.feeder_ev_count(feeder, penetration)

    grid_w = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grid_tau = [0.60, 0.75, 0.90, 1.10]
    grid_lam = [0.0, 1.0, 2.0]

    rows = []
    for w, tau, lam in itertools.product(grid_w, grid_tau, grid_lam):
        r = evaluate_policy(w, tau, lam, params=params, feeder=feeder,
                            limit=limit, bg_forecast=bg_forecast,
                            bg_days=bg_days, wholesale=wholesale,
                            n_ev=n_ev, rng=rng, n_days=n_days)
        r.pop("price")
        r.pop("ev_forecast")
        rows.append(r)
    g = pd.DataFrame(rows)
    g["limit_kw"] = limit
    g["feasible"] = g.cvar95 <= limit

    unconstrained = g.loc[g.margin.idxmax()]
    feas = g[g.feasible]
    constrained = feas.loc[feas.margin.idxmax()] if len(feas) else None
    feas110 = g[g.cvar95 <= 1.10 * limit]  # short-time thermal overload tolerance
    constrained110 = feas110.loc[feas110.margin.idxmax()] if len(feas110) else None

    g.to_csv(RESULTS / "cvar_policy_grid.csv", index=False)
    if verbose:
        def fmt(r):
            return (f"w={r.w:.1f} tau={r.tau:.2f} lam={r.lam:.1f}  "
                    f"margin {r.margin / 100:.2f} GBP/day  CVaR95 {r.cvar95:.1f} kW")

        print(f"\nCVaR-constrained optimisation ({case}, {penetration:.0%} penetration, "
              f"limit {limit:.1f} kW)")
        print(f"  unconstrained optimum : {fmt(unconstrained)} ({unconstrained.cvar95 / limit:.0%})")
        if constrained is not None:
            coc = 100 * (unconstrained.margin - constrained.margin) / unconstrained.margin
            print(f"  CVaR95 <= rating      : {fmt(constrained)}")
            print(f"  cost of constraint    : {coc:.1f}% of operator margin")
        else:
            print("  CVaR95 <= rating      : infeasible at this penetration")
        if constrained110 is not None:
            coc = 100 * (unconstrained.margin - constrained110.margin) / unconstrained.margin
            print(f"  CVaR95 <= 110% rating : {fmt(constrained110)}")
            print(f"  cost of constraint    : {coc:.1f}% of operator margin")
    return dict(grid=g, unconstrained=unconstrained, constrained=constrained,
                constrained110=constrained110, limit=limit)


def backtest(case="constrained", penetration=0.60, seed=99, verbose=True):
    """365 days with seasonal background and wholesale; tariff republished monthly."""
    cfgc = rx.NETWORK_CASES[case]
    rng = np.random.default_rng(seed)
    feeder = fm.Feeder(planning_utilisation=cfgc["utilisation"])
    params, *_ = fm.calibrate_to_step1(rx.CSV_PATH, verbose=False)
    limit = feeder.firm_limit_kw
    n_ev = fm.feeder_ev_count(feeder, penetration)
    price_flat, price_tou = pr.flat_price(), pr.static_tou_price()

    recs = []
    for m in range(12):
        nd = int(MONTH_DAYS[m])
        season = MONTH_SEASONAL[m]
        bg_forecast = fm.background_profile_kw(feeder, season)
        bg_days = fm.background_scenarios(feeder, nd, rng, season)
        ws = pr.sample_wholesale(nd, rng, mean_p_per_kwh=MONTH_WHOLESALE[m])
        agile = pr.agile_from_wholesale(ws)

        def sampler(_n=n_ev):
            return fm.sample_sessions(params, _n, rng, 0.80)

        da = pr.dtou_equilibrium(bg_forecast, sampler, limit, params.charger_kw, rng,
                                 n_iter=20, n_scenarios=15)
        da_t = pr.dtou_equilibrium(bg_forecast, sampler, limit, params.charger_kw, rng,
                                   cfg=pr.DToUConfig(cpp_quantile=0.90),
                                   n_iter=20, n_scenarios=15)
        price_prop, n_acs_da = da["price"], da["n_acs"]

        for d in range(nd):
            for scheme in rx.SCHEMES:
                sf = 0.0 if scheme == "uncontrolled" else 0.80
                s = fm.sample_sessions(params, n_ev, rng, sf)
                price = {"uncontrolled": price_flat,
                         "static_tou": price_tou,
                         "agile": agile[d],
                         "proposed": price_prop,
                         "proposed_tail": da_t["price"]}[scheme]
                ev = pr.schedule_day(s, price, rng)
                peak = float((bg_days[d] + ev).max())
                e = ev * DT_H
                sc = float((pr.supply_cost_p_per_kwh(ws[d]) * e).sum())
                if scheme.startswith("proposed"):
                    _, _, n_rt = pr.surplus_and_capacity(bg_days[d], limit, params.charger_kw)
                    bill = np.clip(price + pr.rtp_adjust(da["T_dtou"], n_acs_da, n_rt), 0, None)
                else:
                    bill = price
                recs.append(dict(month=m + 1, scheme=scheme, peak_kw=peak,
                                 energy_kwh=float(e.sum()),
                                 consumer_p=float((bill * e).sum()),
                                 margin_rn_p=pr.OFGEM_CAP_P_PER_KWH * float(e.sum()) - sc,
                                 margin_pub_p=float((bill * e).sum()) - sc,
                                 overload=peak > limit))
    bt = pd.DataFrame(recs)

    summ = []
    for scheme, g in bt.groupby("scheme"):
        cum = g["margin_rn_p"].cumsum().values
        drawdown = float((np.maximum.accumulate(cum) - cum).max())
        _, cvar = rx.var_cvar(g["peak_kw"].values, 0.95)
        sharpe = g["margin_rn_p"].mean() / g["margin_rn_p"].std()
        summ.append(dict(
            scheme=scheme, label=rx.SCHEME_LABEL[scheme],
            annual_margin_gbp=g["margin_rn_p"].sum() / 100,
            margin_mean_p_day=g["margin_rn_p"].mean(),
            margin_sd_p_day=g["margin_rn_p"].std(),
            sharpe=sharpe,
            annualised_sharpe=sharpe * np.sqrt(365),
            max_drawdown_gbp=drawdown / 100,
            overload_days=int(g["overload"].sum()),
            CVaR95_peak_kw=cvar,
            consumer_gbp_year=g["consumer_p"].sum() / 100,
            consumer_p_per_kwh=g["consumer_p"].sum() / g["energy_kwh"].sum(),
        ))
    s = pd.DataFrame(summ).set_index("scheme").loc[rx.SCHEMES].reset_index()
    bt.to_csv(RESULTS / "backtest_daily.csv", index=False)
    s.to_csv(RESULTS / "backtest_summary.csv", index=False)
    if verbose:
        print(f"\n365-day seasonal backtest ({case}, {penetration:.0%} penetration)")
        print(s[["scheme", "annual_margin_gbp", "sharpe", "annualised_sharpe",
                 "max_drawdown_gbp", "overload_days", "CVaR95_peak_kw",
                 "consumer_gbp_year"]].round(3).to_string(index=False))
    return dict(daily=bt, summary=s, limit=limit)


def write_frontiers(df: pd.DataFrame, verbose=True):
    for case in ["base", "constrained"]:
        f = efficient_frontier(df, case, 0.60)
        f.to_csv(RESULTS / f"frontier_{case}_p60.csv", index=False)
        if verbose:
            print(f"\nEfficient frontier ({case}, 60% penetration, baseline = uncontrolled)")
            print(f[["scheme", "risk_reduction_pct", "overload_reduction_pct",
                     "consumer_cost_change_pct", "consumer_p95_change_pct",
                     "operator_margin_change_pct", "pareto_efficient"]]
                  .round(2).to_string(index=False))


if __name__ == "__main__":
    write_frontiers(pd.read_csv(RESULTS / "metrics_by_scheme.csv"))
    optimise_cvar()
    backtest()

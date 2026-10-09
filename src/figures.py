"""
Run the full step-3 pipeline and write every figure in docs/ANALYSIS.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import feeder_model as fm
import pricing as pr
import run_experiments as rx
import analysis as an
from feeder_model import HOURS
from paths import FIGURES, RESULTS

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})

COLOR = {"uncontrolled": "#8c8c8c", "static_tou": "#1f77b4",
         "agile": "#d62728", "proposed": "#2ca02c",
         "proposed_tail": "#17557a", "cvar_opt": "#9467bd"}
SHORT = {"uncontrolled": "Uncontrolled", "static_tou": "Static ToU (Go)",
         "agile": "Agile (RTP)", "proposed": "Proposed DToU+CPP+RTP",
         "proposed_tail": "Proposed, tail-aware CPP"}


def fig1_calibration(diag):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(HOURS, diag["target_shape"] * 100, "o-", ms=3, color="k",
            label="Step-1 Monte Carlo output (Wei et al. model)")
    ax.plot(HOURS, diag["fitted_shape"] * 100, "--", color="#d62728",
            label=f"Fitted session model (R$^2$ = {diag['r2_shape']:.4f})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of daily EV energy")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(fontsize=8)
    ax.set_title("Fig 1  Session-level model reproduces the step-1 London EV load shape")
    fig.savefig(FIGURES / "fig1_calibration.png")
    plt.close(fig)


def fig2_profiles(out, case="constrained", pen=0.60):
    prof, limit, bg = out[case]["profiles"], out[case]["limit"], out[case]["bg_forecast"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for scheme in rx.SCHEMES:
        tot, ev = prof[(case, pen, scheme)]
        axes[0].plot(HOURS, tot, color=COLOR[scheme], label=SHORT[scheme], lw=1.6)
        axes[1].plot(HOURS, ev, color=COLOR[scheme], label=SHORT[scheme], lw=1.6)
    axes[0].plot(HOURS, bg, color="k", ls=":", lw=1.2, label="Non-EV background")
    axes[0].axhline(limit, color="crimson", ls="--", lw=1.2, label=f"Feeder limit {limit:.0f} kW")
    axes[0].set_title("Mean total feeder load")
    axes[1].set_title("Mean EV charging load only")
    for a in axes:
        a.set_xlabel("Hour of day")
        a.set_ylabel("kW")
        a.set_xticks(range(0, 25, 3))
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Fig 2  Mean load profiles, {case} feeder, {pen:.0%} EV penetration",
                 y=1.02, fontsize=10)
    fig.savefig(FIGURES / "fig2_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig3_peak_distribution(case="constrained", pen=0.60, n_days=4000, seed=515):
    """Re-simulate one case to get the full daily-peak distribution."""
    cfgc = rx.NETWORK_CASES[case]
    rng = np.random.default_rng(seed)
    feeder = fm.Feeder(planning_utilisation=cfgc["utilisation"])
    params, *_ = fm.calibrate_to_step1(rx.CSV_PATH, verbose=False)
    limit = feeder.firm_limit_kw
    bgf = fm.background_profile_kw(feeder, cfgc["seasonal"])
    bgd = fm.background_scenarios(feeder, n_days, rng, cfgc["seasonal"])
    agile = pr.agile_from_wholesale(pr.sample_wholesale(n_days, rng))
    n_ev = fm.feeder_ev_count(feeder, pen)

    def sampler(_n=n_ev):
        return fm.sample_sessions(params, _n, rng, 0.80)

    da = pr.dtou_equilibrium(bgf, sampler, limit, params.charger_kw, rng,
                             cfg=pr.DToUConfig(cpp_quantile=0.50))
    da_t = pr.dtou_equilibrium(bgf, sampler, limit, params.charger_kw, rng,
                               cfg=pr.DToUConfig(cpp_quantile=0.90))

    peaks = {}
    for scheme in rx.SCHEMES:
        p = np.zeros(n_days)
        for d in range(n_days):
            sf = 0.0 if scheme == "uncontrolled" else 0.80
            s = fm.sample_sessions(params, n_ev, rng, sf)
            price = {"uncontrolled": pr.flat_price(),
                     "static_tou": pr.static_tou_price(),
                     "agile": agile[d], "proposed": da["price"],
                     "proposed_tail": da_t["price"]}[scheme]
            p[d] = (bgd[d] + pr.schedule_day(s, price, rng)).max()
        peaks[scheme] = p

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bins = np.linspace(min(v.min() for v in peaks.values()),
                       max(v.max() for v in peaks.values()), 70)
    for scheme, p in peaks.items():
        ax.hist(p, bins=bins, histtype="step", lw=1.6, density=True,
                color=COLOR[scheme], label=SHORT[scheme])
        _, c = rx.var_cvar(p, 0.95)
        ax.axvline(c, color=COLOR[scheme], ls=":", lw=1.1, alpha=0.8)
    ax.axvline(limit, color="crimson", ls="--", lw=1.5, label=f"Feeder limit {limit:.0f} kW")
    ax.set_xlabel("Daily peak feeder load (kW)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title(f"Fig 3  Distribution of daily peak load ({case}, {pen:.0%} penetration)\n"
                 "dotted lines = CVaR$_{95}$", fontsize=10)
    fig.savefig(FIGURES / "fig3_peak_distribution.png")
    plt.close(fig)
    np.savez(RESULTS / "peak_samples.npz", **peaks)
    return peaks


def fig4_risk_vs_penetration(df):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=True)
    for j, case in enumerate(["base", "constrained"]):
        d = df[df.case == case]
        lim = d.limit_kw.iloc[0]
        for scheme in rx.SCHEMES:
            s = d[d.scheme == scheme].sort_values("penetration")
            axes[0, j].plot(s.penetration * 100, s.CVaR95_kw, "o-", ms=3.5,
                            color=COLOR[scheme], label=SHORT[scheme])
            axes[1, j].plot(s.penetration * 100, s.p_overload * 100, "o-", ms=3.5,
                            color=COLOR[scheme], label=SHORT[scheme])
        axes[0, j].axhline(lim, color="crimson", ls="--", lw=1.2, label=f"limit {lim:.0f} kW")
        axes[0, j].set_title(f"{case} feeder")
        axes[0, j].set_ylabel("CVaR$_{95}$ of daily peak (kW)")
        axes[1, j].set_ylabel("P(peak > rating)  [%]")
        axes[1, j].set_xlabel("Household EV penetration (%)")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Fig 4  Network tail risk against EV penetration", y=1.01, fontsize=11)
    fig.savefig(FIGURES / "fig4_risk_vs_penetration.png", bbox_inches="tight")
    plt.close(fig)


def fig5_frontier(df, case="constrained", pen=0.60):
    f = an.efficient_frontier(df, case, pen)
    d = df[(df.case == case) & (df.penetration == pen)].set_index("scheme")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for _, r in f.iterrows():
        s = r.scheme
        m = "o" if r.pareto_efficient else "s"
        points = [(r.consumer_cost_change_pct, r.risk_reduction_pct),
                  (d.loc[s, "consumer_p_per_kwh"], d.loc[s, "consumer_p_per_kwh_sd"])]
        for ax, xy in zip(axes, points):
            ax.scatter(*xy, s=95, marker=m, color=COLOR[s], zorder=3, edgecolor="k", linewidth=0.6)
            ax.annotate(SHORT[s], xy, textcoords="offset points", xytext=(6, 5), fontsize=7.5)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_xlabel("Change in mean consumer cost vs uncontrolled (%)")
    axes[0].set_ylabel("Reduction in CVaR$_{95}$ of peak (%)")
    axes[0].set_title("Risk-cost frontier\n(circles = Pareto-efficient)", fontsize=9.5)
    axes[1].set_xlabel("Mean consumer cost (p/kWh)")
    axes[1].set_ylabel("Day-to-day s.d. of consumer cost (p/kWh)")
    axes[1].set_title("Cost level vs cost predictability", fontsize=9.5)
    fig.suptitle(f"Fig 5  Pareto efficiency ({case} feeder, {pen:.0%} penetration)", y=1.03, fontsize=11)
    fig.savefig(FIGURES / "fig5_frontier.png", bbox_inches="tight")
    plt.close(fig)
    return f


def fig6_tariff_anatomy(out, case="constrained", pen=0.60):
    da = out[case]["tariffs"][pen]["tail"]
    bg, limit = out[case]["bg_forecast"], out[case]["limit"]
    charger_kw = out[case]["params"].charger_kw
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    axes[0].fill_between(HOURS, 0, bg, color="#bbbbbb", label="Non-EV background")
    axes[0].fill_between(HOURS, bg, bg + da["ev_forecast_kw"], color="#2ca02c",
                         alpha=0.65, label="Forecast EV (equilibrium)")
    axes[0].axhline(limit, color="crimson", ls="--", label="Feeder limit")
    axes[0].set_ylabel("kW")
    axes[0].set_title("Headroom and forecast load", fontsize=9.5)
    axes[0].legend(fontsize=7)

    axes[1].step(HOURS, da["n_acs"], where="post", color="#1f77b4", lw=1.6,
                 label="$N_{ACs}(i)$ = max simultaneous households")
    axes[1].step(HOURS, da["ev_forecast_kw"] / charger_kw, where="post", color="#2ca02c",
                 lw=1.6, label="Forecast simultaneous households")
    axes[1].set_ylabel("households")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Adapted $N_{AC}$ (eqs. 2-3)", fontsize=9.5)

    axes[2].step(HOURS, da["T_dtou"], where="post", color="#2ca02c", lw=1.6, label="$T_{DToU}$ (eqs. 5-7)")
    axes[2].step(HOURS, da["price"], where="post", color="k", lw=1.4, label="$T_{DToU}$ + CPP (eq. 8)")
    axes[2].step(HOURS, pr.static_tou_price(), where="post", color="#1f77b4",
                 ls="--", lw=1.2, label="Static ToU reference")
    cpp_on = da["CPP"] > 0
    if cpp_on.any():
        axes[2].scatter(HOURS[cpp_on], da["price"][cpp_on], color="crimson",
                        s=18, zorder=4, label="CPP tier active")
    axes[2].set_ylabel("p/kWh")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Published day-ahead tariff", fontsize=9.5)
    axes[2].set_ylim(0, min(80, max(40, da["price"].max() * 1.15)))
    for a in axes:
        a.set_xlabel("Hour")
        a.set_xticks(range(0, 25, 6))
    fig.suptitle(f"Fig 6  Anatomy of the adapted DToU+CPP tariff ({case}, {pen:.0%} penetration)",
                 y=1.04, fontsize=11)
    fig.savefig(FIGURES / "fig6_tariff_anatomy.png", bbox_inches="tight")
    plt.close(fig)


def fig7_cvar_frontier(grid, limit):
    front, best = [], -1e18
    for _, r in grid.sort_values("cvar95").iterrows():
        if r.margin > best:
            front.append(r)
            best = r.margin
    f = pd.DataFrame(front)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sc = ax.scatter(grid.cvar95, grid.margin / 100, c=grid.w, cmap="viridis",
                    s=28, alpha=0.85, edgecolor="none")
    ax.plot(f.cvar95, f.margin / 100, "k-", lw=1.4, zorder=2, label="efficient frontier")
    ax.axvline(limit, color="crimson", ls="--", lw=1.4, label=f"feeder rating {limit:.0f} kW")
    ax.axvline(1.10 * limit, color="darkorange", ls=":", lw=1.4, label="110% of rating")
    plt.colorbar(sc, ax=ax, label="w  (0 = cost-following, 1 = network-following)")
    ax.set_xlabel("CVaR$_{95}$ of daily peak load (kW)")
    ax.set_ylabel("Expected operator margin (GBP/day, revenue-neutral)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Fig 7  Profit vs network-safety frontier over the blended price family", fontsize=10)
    fig.savefig(FIGURES / "fig7_cvar_frontier.png")
    plt.close(fig)


def fig8_backtest(bt):
    daily = bt["daily"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for scheme in rx.SCHEMES:
        g = daily[daily.scheme == scheme]
        days = np.arange(len(g))
        axes[0].plot(days, g["margin_rn_p"].cumsum() / 100, color=COLOR[scheme], lw=1.5, label=SHORT[scheme])
        axes[1].plot(days, g["consumer_p"].cumsum() / 100, color=COLOR[scheme], lw=1.5, label=SHORT[scheme])
    axes[0].set_ylabel("Cumulative operator margin (GBP)")
    axes[1].set_ylabel("Cumulative consumer spend (GBP)")
    for a in axes:
        a.set_xlabel("Day of year")
        a.legend(fontsize=7.5)
    fig.suptitle("Fig 8  365-day seasonal backtest, one representative feeder", y=1.03, fontsize=11)
    fig.savefig(FIGURES / "fig8_backtest.png", bbox_inches="tight")
    plt.close(fig)


def fig9_breakeven(df, case="constrained"):
    d = df[df.case == case]
    pens, be = [], []
    for pen in sorted(d.penetration.unique()):
        a = d[(d.penetration == pen) & (d.scheme == "agile")].iloc[0]
        p = d[(d.penetration == pen) & (d.scheme == "proposed")].iloc[0]
        saved = a.exceed_kwh_mean - p.exceed_kwh_mean
        if saved <= 1e-6:
            continue
        extra = (p.consumer_p_per_kwh - a.consumer_p_per_kwh) * p.energy_kwh_day / 100
        pens.append(pen * 100)
        be.append(extra / saved)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(pens, be, "o-", color="#2ca02c", lw=1.8, ms=5)
    ax.axhspan(6.0, 25.3, color="#1f77b4", alpha=0.13,
               label="Range of GB VoLL estimates\n(£6/kWh admin - £25.3/kWh ENWL)")
    ax.axhline(16.94, color="#1f77b4", ls="--", lw=1.3, label="Ofgem headline VoLL £16.94/kWh")
    ax.set_yscale("log")
    ax.set_xlabel("Household EV penetration (%)")
    ax.set_ylabel("Break-even shadow price of overload energy (GBP/kWh)")
    ax.legend(fontsize=8)
    ax.set_title("Fig 9  Shadow price at which the proposed scheme overtakes Agile\n"
                 "(below the VoLL band = proposed scheme is welfare-superior)", fontsize=10)
    fig.savefig(FIGURES / "fig9_breakeven.png")
    plt.close(fig)
    return pd.DataFrame({"penetration_pct": pens, "breakeven_gbp_per_kwh": be})


def main():
    outs = rx.run_all(verbose=False)
    df = outs["all"]

    fig1_calibration(outs["base"]["diag"])
    fig2_profiles(outs)
    fig3_peak_distribution()
    fig4_risk_vs_penetration(df)
    fig5_frontier(df)
    fig6_tariff_anatomy(outs)
    opt = an.optimise_cvar(verbose=True)
    fig7_cvar_frontier(opt["grid"], opt["limit"])
    fig8_backtest(an.backtest(verbose=True))
    fig9_breakeven(df).to_csv(RESULTS / "breakeven_shadow_price.csv", index=False)
    an.write_frontiers(df, verbose=False)
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()

"""
Tariffs, the charging-behaviour model, and settlement.

Schemes
  uncontrolled   Ofgem cap flat rate, no price response
  static_tou     Octopus Go, cheap window 00:30-05:30
  agile          Octopus Agile, from the published algorithm on GB wholesale
  proposed       Khan & Aziz (2022) DToU + CPP + RTP, adapted to a London
                 LV feeder and iterated to a fixed point

Sources
  [P1] Ofgem price cap, 1 Jul - 30 Sep 2026: 26.11 p/kWh
  [P2] Octopus Go: 7.5 p/kWh 00:30-05:30, ~25 p/kWh otherwise
  [P3] Octopus Agile: 2.2 x wholesale - 3.5 p/kWh, +12 p/kWh 16:00-19:00, cap 100 p/kWh
  [P4] GB wholesale: weekly average 6-10 p/kWh through 2025 (Ofgem)
  [P5] UKPN London Power Networks LC14 DUoS, effective 1 April 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feeder_model import SP, DT_H, HOURS, Sessions

OFGEM_CAP_P_PER_KWH = 26.11                                       # [P1]
DUOS_P_PER_KWH = {"red": 3.440, "amber": 0.262, "green": 0.008}  # [P5]
OTHER_SUPPLY_COST_P_PER_KWH = 5.0  # TNUoS + BSUoS + policy + supplier opex

GO_CHEAP_P = 7.5                   # [P2]
GO_PEAK_P = 25.0
GO_WINDOW = (0.5, 5.5)

AGILE_MULT = 2.2                   # [P3]
AGILE_OFFSET = -3.5
AGILE_PEAK_UPLIFT = 12.0
AGILE_PEAK_WINDOW = (16.0, 19.0)
AGILE_CAP = 100.0


def duos_band(hour: float) -> str:
    if 16.0 <= hour < 19.0:
        return "red"
    if 7.0 <= hour < 16.0 or 19.0 <= hour < 23.0:
        return "amber"
    return "green"


DUOS_VECTOR = np.array([DUOS_P_PER_KWH[duos_band(h)] for h in HOURS])

# Normalised GB day-ahead wholesale shape (relative to daily mean)
_WHOLESALE_SHAPE = np.array([
    0.72, 0.70, 0.68, 0.66, 0.64, 0.63, 0.62, 0.62,
    0.62, 0.63, 0.65, 0.69, 0.75, 0.83, 0.92, 0.98,
    1.02, 1.03, 1.01, 0.97, 0.92, 0.88, 0.85, 0.83,
    0.82, 0.82, 0.83, 0.85, 0.88, 0.92, 0.97, 1.03,
    1.12, 1.24, 1.38, 1.52, 1.62, 1.66, 1.62, 1.52,
    1.38, 1.25, 1.13, 1.03, 0.95, 0.88, 0.82, 0.77,
])


def sample_wholesale(n_days: int, rng: np.random.Generator,
                     mean_p_per_kwh: float = 8.0,
                     day_cv: float = 0.35,
                     slot_cv: float = 0.28,
                     spike_prob: float = 0.02,
                     spike_mult: float = 2.6) -> np.ndarray:
    """
    Half-hourly day-ahead wholesale (p/kWh), shape (n_days, 48), with
    overnight wind-driven dips (sometimes negative) and evening spikes. [P4]
    """
    day_level = mean_p_per_kwh * rng.lognormal(-0.5 * day_cv ** 2, day_cv, size=(n_days, 1))
    noise = rng.lognormal(-0.5 * slot_cv ** 2, slot_cv, size=(n_days, SP))
    w = day_level * _WHOLESALE_SHAPE[None, :] * noise

    windy = rng.random(n_days) < 0.18
    for d in np.where(windy)[0]:
        start = rng.integers(0, 20)
        length = rng.integers(6, 20)
        idx = np.arange(start, start + length) % SP
        w[d, idx] -= rng.uniform(4.0, 11.0)

    spikes = rng.random((n_days, SP)) < spike_prob
    spikes &= ((HOURS >= 16.0) & (HOURS < 20.0))[None, :]
    return np.where(spikes, w * spike_mult, w)


def agile_from_wholesale(wholesale: np.ndarray) -> np.ndarray:
    p = AGILE_MULT * wholesale + AGILE_OFFSET
    uplift = np.where((HOURS >= AGILE_PEAK_WINDOW[0]) & (HOURS < AGILE_PEAK_WINDOW[1]),
                      AGILE_PEAK_UPLIFT, 0.0)
    return np.minimum(p + uplift[None, :], AGILE_CAP)


def flat_price() -> np.ndarray:
    return np.full(SP, OFGEM_CAP_P_PER_KWH)


def static_tou_price() -> np.ndarray:
    cheap = (HOURS >= GO_WINDOW[0]) & (HOURS < GO_WINDOW[1])
    return np.where(cheap, GO_CHEAP_P, GO_PEAK_P)


@dataclass
class DToUConfig:
    profit_margin: float = 1.10         # eq. (7)
    cpp_low: float = 0.80               # eq. (8) penalty threshold
    cpp_high: float = 1.00              # eq. (8) curtailment threshold
    cpp_penalty_multiple: float = 1.0   # CPP_T = multiple x peak static rate
    curtail_price_p: float = 200.0      # stands in for eq. (8)'s infinity
    rtp_n: float = 0.5                  # eqs. (12)-(13)
    reference_tariff: str = "static_tou"
    cpp_quantile: float = 0.50          # 0.5 = paper (point forecast); 0.9 = tail-aware


def surplus_and_capacity(background_kw: np.ndarray, firm_limit_kw: float, charger_kw: float):
    """
    Khan & Aziz eqs. (1)-(3). N_ACs(i) is the maximum number of households
    that may charge simultaneously in slot i without breaching headroom.
    """
    surplus = firm_limit_kw - background_kw
    n_ac = np.clip(surplus, 0.0, None) / charger_kw
    n_acs = np.floor(n_ac).astype(int)
    return surplus, n_ac, n_acs


def day_ahead_dtou(background_kw: np.ndarray,
                   forecast_ev_kw: np.ndarray,
                   firm_limit_kw: float,
                   charger_kw: float,
                   cfg: DToUConfig = DToUConfig(),
                   cpp_load_kw: np.ndarray | None = None):
    """
    Day-ahead DToU + CPP tariff (p/kWh), Khan & Aziz eqs. (4)-(8).

    CPP fires on headroom utilisation u = EV load / headroom:
    penalty for cpp_low <= u < cpp_high, curtailment for u >= cpp_high.
    cpp_load_kw defaults to the point forecast.
    """
    surplus, n_ac, n_acs = surplus_and_capacity(background_kw, firm_limit_kw, charger_kw)

    ref = static_tou_price() if cfg.reference_tariff == "static_tou" else flat_price()
    energy_ref_kwh = n_acs * charger_kw * DT_H
    bill_static = float((ref * energy_ref_kwh).sum())

    denom = float((np.abs(background_kw) * charger_kw * n_acs).sum())
    c1 = 0.0 if denom <= 0 else cfg.profit_margin * bill_static / denom
    t_dtou = c1 * np.abs(background_kw)

    headroom = np.clip(surplus, 1e-6, None)
    cpp_load = forecast_ev_kw if cpp_load_kw is None else cpp_load_kw
    utilisation = cpp_load / headroom
    peak_static = float(ref.max())
    cpp = np.zeros(SP)
    status = np.array(["ok"] * SP, dtype=object)

    band_cpp = (utilisation >= cfg.cpp_low) & (utilisation < cfg.cpp_high)
    band_cut = utilisation >= cfg.cpp_high
    cpp[band_cpp] = cfg.cpp_penalty_multiple * peak_static
    cpp[band_cut] = cfg.curtail_price_p
    status[band_cpp] = "CPP"
    status[band_cut] = "curtailed"

    return {
        "surplus_kw": surplus,
        "n_ac": n_ac,
        "n_acs": n_acs,
        "headroom_kw": headroom,
        "utilisation": utilisation,
        "T_dtou": t_dtou,
        "CPP": cpp,
        "status": status,
        "price": t_dtou + cpp,
        "C1": c1,
        "bill_static_reference_p": bill_static,
    }


def rtp_adjust(t_dtou: np.ndarray, n_acs_dayahead: np.ndarray,
               n_acs_realtime: np.ndarray, n: float = 0.5) -> np.ndarray:
    """
    Khan & Aziz eqs. (12)-(13). A fraction n of each slot's surplus/deficit
    is settled in that slot, the rest spread over the remaining slots.
    """
    rtp = np.zeros(SP)
    for i in range(SP):
        remaining = SP - i - 1
        if n_acs_realtime[i] > n_acs_dayahead[i]:
            rtp[i] -= n * t_dtou[i]
            if remaining:
                rtp[i + 1:] -= (1 - n) * t_dtou[i] / remaining
        elif n_acs_realtime[i] < n_acs_dayahead[i]:
            rtp[i] += n * t_dtou[i]
            if remaining:
                rtp[i + 1:] += (1 - n) * t_dtou[i] / remaining
    return rtp


def schedule_day(sessions: Sessions, price: np.ndarray, rng: np.random.Generator,
                 timer_fraction: float = 0.75) -> np.ndarray:
    """
    Realised EV load (48,) under a price vector.

    Non-smart sessions charge at full power from plug-in. Smart sessions
    fill the cheapest slots in their dwell window. Among tied prices, a
    fraction timer_fraction take the earliest slot (timer behaviour) and
    the rest pick at random.
    """
    load = np.zeros(SP)
    if len(sessions) == 0:
        return load

    for k in range(len(sessions)):
        a, d = sessions.arrive_h[k], sessions.depart_h[k]
        e, p = sessions.energy_kwh[k], sessions.power_kw[k]
        n_slots_needed = int(np.ceil(e / (p * DT_H)))
        window = np.arange(int(np.floor(a / DT_H)), int(np.ceil(d / DT_H)))

        if not sessions.smart[k]:
            chosen = window[:n_slots_needed]
        else:
            wp = price[window % SP]
            if rng.random() < timer_fraction:
                order = np.argsort(wp, kind="stable")
            else:
                order = np.lexsort((rng.permutation(len(wp)), wp))
            chosen = window[order[:n_slots_needed]]

        remaining = e
        for s in chosen:
            step = min(p * DT_H, remaining)
            load[s % SP] += step / DT_H
            remaining -= step
            if remaining <= 1e-9:
                break
    return load


def dtou_equilibrium(background_kw: np.ndarray,
                     session_sampler,
                     firm_limit_kw: float,
                     charger_kw: float,
                     rng: np.random.Generator,
                     cfg: DToUConfig = DToUConfig(),
                     n_iter: int = 30,
                     n_scenarios: int = 25,
                     damping: float = 0.35,
                     timer_fraction: float = 0.75,
                     tol: float = 1e-3):
    """
    Iterate price -> scheduled response -> forecast until the published
    DToU + CPP price is consistent with the demand it induces.

    session_sampler is a zero-argument callable returning a Sessions object.
    """
    ev_forecast = np.zeros(SP)
    ev_tail = np.zeros(SP)
    history = []
    for _ in range(n_iter):
        da = day_ahead_dtou(background_kw, ev_forecast, firm_limit_kw,
                            charger_kw, cfg, cpp_load_kw=ev_tail)
        draws = np.empty((n_scenarios, SP))
        for j in range(n_scenarios):
            draws[j] = schedule_day(session_sampler(), da["price"], rng, timer_fraction=timer_fraction)
        induced = draws.mean(axis=0)
        induced_tail = np.quantile(draws, cfg.cpp_quantile, axis=0)
        new = (1 - damping) * ev_forecast + damping * induced
        new_tail = (1 - damping) * ev_tail + damping * induced_tail
        delta = float(np.abs(new - ev_forecast).max())
        ev_forecast, ev_tail = new, new_tail
        history.append(delta)
        if delta < tol:
            break
    da = day_ahead_dtou(background_kw, ev_forecast, firm_limit_kw,
                        charger_kw, cfg, cpp_load_kw=ev_tail)
    da["ev_forecast_kw"] = ev_forecast
    da["ev_tail_forecast_kw"] = ev_tail
    da["convergence"] = history
    return da


def supply_cost_p_per_kwh(wholesale: np.ndarray) -> np.ndarray:
    return wholesale + DUOS_VECTOR + OTHER_SUPPLY_COST_P_PER_KWH


def settle(ev_load_kw: np.ndarray, retail_price: np.ndarray, wholesale: np.ndarray):
    """Consumer cost and operator margin for one day (pence)."""
    energy = ev_load_kw * DT_H
    revenue = float((retail_price * energy).sum())
    cost = float((supply_cost_p_per_kwh(wholesale) * energy).sum())
    return {
        "energy_kwh": float(energy.sum()),
        "consumer_cost_p": revenue,
        "operator_margin_p": revenue - cost,
        "supply_cost_p": cost,
    }

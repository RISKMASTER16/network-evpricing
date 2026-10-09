"""
Physical layer: a representative London LV radial feeder, its non-EV
background demand, and an EV charging-session generator calibrated to
the step-1 London load profile.

All profiles use the GB 48 x 30-minute settlement grid.

Sources
  [D1] DESNZ Subnational electricity consumption 2005-2024, London 2024:
       3,729.075k domestic meters, 12,084.705 GWh -> 3,240.5 kWh/meter/yr
  [D2] SP Energy Networks ESDD-02-012: max 75 customers per radial LV feeder
  [D3] Typical GB distribution substation: 500 kVA across ~4 LV feeders
  [D4] Wei et al. (2026) Table 6: private BEV mileage ~ LN(3.35, 0.70) km,
       6.3 sessions/month, tri-modal plug-in time
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm, norm

SP = 48
DT_H = 0.5
HOURS = np.arange(SP) * DT_H


@dataclass
class Feeder:
    n_households: int = 75             # [D2]
    substation_kva: float = 500.0      # [D3]
    feeders_per_substation: int = 4    # [D3]
    power_factor: float = 0.95
    kwh_per_household_year: float = 3240.5  # [D1]
    planning_utilisation: float = 1.0  # fraction of thermal rating the DNO plans to

    @property
    def rating_kva(self) -> float:
        return self.substation_kva / self.feeders_per_substation

    @property
    def rating_kw(self) -> float:
        return self.rating_kva * self.power_factor

    @property
    def firm_limit_kw(self) -> float:
        """P_max in Khan & Aziz eq. (1)."""
        return self.rating_kw * self.planning_utilisation


# Stylised GB domestic (Elexon Profile Class 1-like) weekday shape.
# Only the shape matters; the level is pinned by [D1].
_PC1_SHAPE = np.array([
    0.55, 0.51, 0.48, 0.45, 0.43, 0.41, 0.40, 0.39,   # 00:00-03:30
    0.39, 0.39, 0.40, 0.42, 0.46, 0.54, 0.66, 0.79,   # 04:00-07:30
    0.86, 0.86, 0.82, 0.76, 0.72, 0.70, 0.70, 0.71,   # 08:00-11:30
    0.74, 0.76, 0.77, 0.77, 0.77, 0.78, 0.80, 0.84,   # 12:00-15:30
    0.90, 1.00, 1.14, 1.30, 1.44, 1.52, 1.55, 1.52,   # 16:00-19:30
    1.44, 1.33, 1.21, 1.10, 0.99, 0.86, 0.74, 0.64,   # 20:00-23:30
])


def background_profile_kw(feeder: Feeder, seasonal_factor: float = 1.0) -> np.ndarray:
    """Deterministic non-EV feeder demand (kW). seasonal_factor ~1.2 winter, ~0.85 summer."""
    daily_kwh_per_hh = feeder.kwh_per_household_year / 365.0
    shape = _PC1_SHAPE / _PC1_SHAPE.mean()
    return shape * (daily_kwh_per_hh / 24.0) * seasonal_factor * feeder.n_households


def background_scenarios(feeder: Feeder, n_days: int, rng: np.random.Generator,
                         seasonal_factor: float = 1.0, cv: float = 0.10) -> np.ndarray:
    """
    Stochastic background demand, shape (n_days, 48): a lognormal day
    factor times half-hourly lognormal noise scaled by 1/sqrt(households).
    """
    base = background_profile_kw(feeder, seasonal_factor)
    day_factor = rng.lognormal(-0.5 * cv ** 2, cv, size=(n_days, 1))
    hh_cv = cv / np.sqrt(feeder.n_households)
    hh_noise = rng.lognormal(-0.5 * hh_cv ** 2, hh_cv, size=(n_days, SP))
    return base[None, :] * day_factor * hh_noise


@dataclass
class EVFleetParams:
    charger_kw: float = 7.4            # 32 A single-phase home charger
    kwh_per_km: float = 0.19
    mileage_mu: float = 3.35           # [D4]
    mileage_sigma: float = 0.70        # [D4]
    days_per_session: float = 30.0 / 6.3
    battery_kwh: float = 60.0
    min_session_kwh: float = 2.0
    arr_w: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.25, 0.60]))
    arr_mu: np.ndarray = field(default_factory=lambda: np.array([0.0, 9.42, 20.0]))
    arr_sd: np.ndarray = field(default_factory=lambda: np.array([0.59, 2.62, 4.15]))
    depart_mu: float = 7.5
    depart_sd: float = 1.2
    min_dwell_h: float = 2.0

    @property
    def sessions_per_ev_day(self) -> float:
        return 6.3 / 30.0

    def energy_dist(self):
        """Lognormal kWh required per session."""
        scale_km = np.exp(self.mileage_mu) * self.days_per_session
        return lognorm(s=self.mileage_sigma, scale=scale_km * self.kwh_per_km)

    def mean_session_kwh(self) -> float:
        d = self.energy_dist()
        xs = np.linspace(0.01, self.battery_kwh * 3, 4000)
        pdf = d.pdf(xs)
        e = np.clip(xs, self.min_session_kwh, self.battery_kwh)
        return float(np.trapezoid(e * pdf, xs) / np.trapezoid(pdf, xs))


def _arrival_pdf(params: EVFleetParams, t: np.ndarray) -> np.ndarray:
    """Tri-modal Gaussian plug-in density, wrapped at midnight."""
    pdf = np.zeros_like(t, dtype=float)
    w = params.arr_w / params.arr_w.sum()
    for wi, mu, sd in zip(w, params.arr_mu, params.arr_sd):
        for shift in (-24.0, 0.0, 24.0):
            pdf += wi * norm.pdf(t, mu + shift, sd)
    return pdf


def uncontrolled_profile_shape(params: EVFleetParams) -> np.ndarray:
    """
    Expected uncontrolled charging power per daily session (kW), 48 slots:

        L(t) = P * sum_a f_arr(a) * P(D > t - a) * da,   D = E / P
    """
    fine = np.linspace(0, 24, 24 * 12, endpoint=False)
    da = fine[1] - fine[0]
    f_arr = _arrival_pdf(params, fine)
    f_arr = f_arr / (f_arr.sum() * da)

    dist = params.energy_dist()
    lag = fine[:, None] - fine[None, :]
    lag = np.where(lag < 0, lag + 24.0, lag)
    still_charging = 1.0 - dist.cdf(np.clip(lag, 0, None) * params.charger_kw)
    load_fine = params.charger_kw * (f_arr[None, :] * still_charging).sum(axis=1) * da
    return load_fine.reshape(SP, -1).mean(axis=1)


def calibrate_to_step1(csv_path, params: EVFleetParams | None = None, verbose: bool = True):
    """
    Fit the plug-in mixture so the session model's uncontrolled profile
    matches the shape of the step-1 London profile.

    Returns (fitted_params, target_shape_48, n_ev_london, diagnostics).
    """
    params = params or EVFleetParams()
    raw = pd.read_csv(csv_path)
    slot = np.clip((raw["hour"].astype(float) * 2).astype(int), 0, SP - 1)
    target_mw = raw.groupby(slot)["mean_load_mw"].mean().reindex(range(SP)).values
    target_kw = target_mw * 1000.0
    daily_mwh = target_mw.mean() * 24.0
    tgt_shape = target_kw / target_kw.sum()

    def unpack(x):
        w = np.exp(x[:3])
        w /= w.sum()
        mu = np.array([x[3] % 24, x[4] % 24, x[5] % 24])
        sd = np.exp(x[6:9])
        return w, mu, sd

    def obj(x):
        w, mu, sd = unpack(x)
        p = EVFleetParams(**{**params.__dict__, "arr_w": w, "arr_mu": mu, "arr_sd": sd})
        prof = uncontrolled_profile_shape(p)
        return float(((prof / prof.sum() - tgt_shape) ** 2).sum() * 1e4)

    x0 = np.concatenate([np.log(params.arr_w), params.arr_mu, np.log(params.arr_sd)])
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": 8000, "fatol": 1e-10, "xatol": 1e-8})
    w, mu, sd = unpack(res.x)
    fitted = EVFleetParams(**{**params.__dict__, "arr_w": w, "arr_mu": mu, "arr_sd": sd})

    prof = uncontrolled_profile_shape(fitted)
    fit_shape = prof / prof.sum()
    r2 = 1 - ((fit_shape - tgt_shape) ** 2).sum() / ((tgt_shape - tgt_shape.mean()) ** 2).sum()

    n_sessions_london = target_kw.sum() / prof.sum()
    n_ev_london = n_sessions_london / fitted.sessions_per_ev_day

    diag = {
        "r2_shape": r2,
        "london_daily_mwh": daily_mwh,
        "london_peak_mw": target_mw.max(),
        "n_sessions_per_day": n_sessions_london,
        "n_ev_london": n_ev_london,
        "mean_session_kwh": fitted.mean_session_kwh(),
        "target_shape": tgt_shape,
        "fitted_shape": fit_shape,
        "target_kw": target_kw,
    }
    if verbose:
        print("Calibration of session model to step-1 profile")
        print(f"  shape R^2                : {r2:0.4f}")
        print(f"  London EV energy         : {daily_mwh:,.0f} MWh/day")
        print(f"  London EV peak           : {target_mw.max():,.1f} MW")
        print(f"  implied sessions/day     : {n_sessions_london:,.0f}")
        print(f"  implied mean kWh/session : {diag['mean_session_kwh']:,.1f}")
        print(f"  implied London BEV fleet : {n_ev_london:,.0f}")
        print(f"  fitted plug-in weights   : {np.round(w, 3)}")
        print(f"  fitted plug-in means (h) : {np.round(mu, 2)}")
        print(f"  fitted plug-in sds  (h)  : {np.round(sd, 2)}")
    return fitted, tgt_shape, n_ev_london, diag


@dataclass
class Sessions:
    arrive_h: np.ndarray
    depart_h: np.ndarray    # may exceed 24
    energy_kwh: np.ndarray
    power_kw: np.ndarray
    smart: np.ndarray       # responds to price signal

    def __len__(self):
        return len(self.arrive_h)


def sample_sessions(params: EVFleetParams, n_ev: float, rng: np.random.Generator,
                    smart_fraction: float = 0.8) -> Sessions:
    """One day of sessions on a feeder with n_ev home-charging EVs (Poisson count)."""
    n = rng.poisson(n_ev * params.sessions_per_ev_day)
    if n == 0:
        z = np.zeros(0)
        return Sessions(z, z, z, z, z.astype(bool))

    w = params.arr_w / params.arr_w.sum()
    comp = rng.choice(3, size=n, p=w)
    arrive = rng.normal(params.arr_mu[comp], params.arr_sd[comp]) % 24.0

    energy = params.energy_dist().rvs(size=n, random_state=rng)
    energy = np.clip(energy, params.min_session_kwh, params.battery_kwh)

    depart = rng.normal(params.depart_mu, params.depart_sd, size=n)
    depart = np.where(depart <= arrive, depart + 24.0, depart)
    need_h = energy / params.charger_kw
    depart = np.maximum(depart, arrive + np.maximum(need_h, params.min_dwell_h))

    power = np.full(n, params.charger_kw)
    smart = rng.random(n) < smart_fraction
    return Sessions(arrive, depart, energy, power, smart)


def feeder_ev_count(feeder: Feeder, penetration: float) -> float:
    return feeder.n_households * penetration

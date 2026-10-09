# London private-vehicle charging start-time: heuristic derivation

**Status: expert-judgment estimate, not a data fit.** No London private-vehicle
telemetry was used to produce these numbers. They were derived by reasoning
from (a) the China private-vehicle tri-modal Gaussian in Wei et al. (2026),
(b) the observed China→London shift in the *ride-hailing* tri-modal Gaussian
(which **was** fit to real London ride-hailing session data), and (c)
independent UK charging-behaviour literature. Treat this as a starting
assumption to be replaced by a real fit as soon as usable London
private-vehicle session data is available.

## 1. Recovering the real China private-vehicle weights

Table 6 of Wei et al. gives only (mean, std) for the three private-vehicle
modes — no mixture weights. Table 4 gives the actual hourly percentage
breakdown for private vehicles. We recovered the implied weights by
least-squares fitting a 3-component Gaussian mixture (means/stds fixed from
Table 6, weights free, wrapped at midnight) against the Table 4 row:

| Component | Meaning | w | μ (h) | σ (h) |
|---|---|---|---|---|
| 1 | narrow overnight/tariff spike | 0.012 | 0.0 | 0.59 |
| 2 | midday/errand charging | 0.308 | 9.42 | 2.62 |
| 3 | evening return-home charging (dominant) | 0.680 | 20.0 | 4.15 |

Notably the sharp midnight spike in the raw data is mostly explained by the
*tail* of the broad evening mode wrapping past 24:00, not by a large
standalone midnight behaviour — so its own weight is small (1.2%).

## 2. Why private vehicles get a smaller shift than ride-hailing

Comparing China and London ride-hailing (fit to real London session data)
gave large shifts: +6.7 h, +1.0 h, and −2.4 h for the three modes. Those
shifts are ride-hailing-specific — dispatch algorithms, tourism/demand
patterns, driver shift structures — and don't transfer to private ownership,
which is anchored to near-universal behaviours (commute, home charging).
Mileage distributions for private vehicles were reported to come out nearly
identical between China and comparable Western datasets, reinforcing that
private charging timing should move only modestly, not proportionally to
ride-hailing's shift.

## 3. Per-mode reasoning

**Evening peak (dominant mode, μ 20.0 h → 18.5 h, σ 4.15 h → 3.6 h).**
UK/National Grid ESO and academic sources place the unmanaged "arrive home
and plug in" peak around 17:00–20:00, generally earlier than China's 20:00 —
consistent with shorter average UK commutes and earlier typical office
finish times. Std tightened slightly to reflect more standardised commuting
hours than assumed for the China sample.

**Midday/errand peak (μ 9.42 h → 10.0 h, σ 2.62 h → 2.4 h).**
Later UK shop-opening and school-run/office-start norms nudge this mode
slightly later; std tightened a little for the same standardisation reason.

**Overnight/tariff peak (μ 0.0 h → 0.5 h, w 0.012 → 0.04, σ 0.59 h → 0.40 h).**
This is the most speculative adjustment. UK EV-specific time-of-use tariffs
(Octopus Go, Intelligent Octopus, Economy 7, etc.) consistently define
off-peak windows starting between 23:30 and 01:00, and app-scheduled smart
charging is comparatively mainstream in the UK. We judged this narrow,
tariff-driven mode deserves a larger weight and a tighter std in London than
in the China sample — but this is an assumption, not a measurement, and is
the single number in this set most worth checking against real data first.

## 4. Resulting parameters (London private vehicle, estimate)

| Component | Meaning | w | μ (h) | σ (h) |
|---|---|---|---|---|
| 1 | overnight/tariff-scheduled | 0.04 | 0.5 | 0.40 |
| 2 | midday/errand/workplace | 0.30 | 10.0 | 2.40 |
| 3 | evening return-home (dominant) | 0.66 | 18.5 | 3.60 |

See `data/processed/london_private_trimodal_params.csv` and
`data/processed/london_private_hourly_distribution.csv` for the
machine-readable versions, and `src/plot_start_time.py` to regenerate the chart.

## 5. What would change this

Any of the following would supersede this heuristic and should be preferred
the moment it's available:

- A genuine tri-modal fit against real London private-vehicle
  session data (home-charger telemetry, smart-charging app exports, DNO
  trial data such as Low Carbon London's residential trial).
- Confirmation/adjustment of the tariff-peak weight (§3, mode 1) against
  actual smart-tariff adoption and scheduling data for private EV owners in
  London specifically (adoption rate, distribution of scheduled start
  times).
- Seasonal effects (this estimate does not account for daylight-saving
  shifts in the evening peak, which UK literature suggests exist).

# London private-vehicle initial SOC: heuristic derivation

**Status: expert-judgment estimate, not a data fit.** As with the
charging-start-time estimate, no London private-vehicle telemetry was used
here. Unlike that estimate, though, one real piece of UK survey evidence was
available and used to inform the adjustment (see §2) — so this is slightly
better grounded than a pure reasoning-by-analogy exercise, but it is still
not a fit to raw session data and should be replaced by one as soon as
that's available.

## 1. China baseline

Table 6 gives a lognormal fit, Initial SOC (%) ~ Lognormal(μ=3.56, σ=0.44)
(natural-log parameterisation: median = e^μ, values in %). Checking this
against Table 3's actual bin percentages confirms the general shape (peak
around 20–40%, right-skewed tail out past 80%), though the reported (μ, σ)
don't reproduce Table 3 exactly bin-for-bin — consistent with normal fit
error, not a red flag.

- Implied median: e^3.56 ≈ **35.2%**
- Implied mean: **38.7%**

## 2. Why we did *not* just carry this over unchanged

Initial SOC should be far more universal than charging
start *time* — it's driven by battery physics, daily mileage vs. range, and
personal "top-up" habit, not by local infrastructure timing or tariff
schedules. That's the default assumption here, and it's why this estimate is
much closer to the China baseline than the charging-start-time one was.

However, one real UK data point argues for a modest downward adjustment: the
DfT's national EV driver survey (n=848) asked respondents with home charging
access their typical battery percentage *before* charging at home, and found
a **median of 24%** — noticeably lower than China's implied 35%. Plausible
reasons this isn't just noise: UK home-charging habits skew toward
"top up when it's needed" rather than frequent opportunistic top-ups
(cheaper petrol-equivalent running costs and widespread overnight
scheduling reduce the anxiety-driven urge to plug in early), and ~20% of UK
EV owners lack home charging entirely and rely on costlier public charging,
which independently pushes people to wait longer before charging.

This single data point is UK-wide, not London-specific, is a self-reported
median with no distribution shape given, and only covers *home* charging
(the China Table 3 data isn't broken out by location, so it likely blends
home/work/public). Given that mismatch in scope, we did not adopt it
outright — instead we blended it 50/50 with the China baseline in log-space,
the same damping approach used for the charging-start-time heuristic:

```
mu_london = 0.5 * mu_china + 0.5 * ln(24)
          = 0.5 * 3.56 + 0.5 * 3.178
          = 3.37
```

σ was widened slightly (0.44 → 0.48) to reflect London's more heterogeneous
charging-access mix (home charger owners, on-street/public-charger-reliant
households, and workplace chargers all likely pull toward somewhat different
typical SOC), rather than any specific data point.

## 3. Resulting parameters (London private vehicle, estimate)

| Distribution | μ (log) | σ (log) | Median SOC | Mean SOC |
|---|---|---|---|---|
| Lognormal | 3.37 | 0.48 | ≈29.1% | ≈32.6% |

Binned form (matching Table 3's style):

| SOC range | % |
|---|---|
| 0–10 | 1.31 |
| 10–20 | 20.47 |
| 20–30 | 30.81 |
| 30–40 | 22.09 |
| 40–50 | 12.38 |
| 50–60 | 6.38 |
| 60–70 | 3.20 |
| 70–80 | 1.61 |
| 80–90 | 0.82 |
| 90–100 | 0.43 |

See `data/processed/london_private_soc_lognormal_params.csv` and
`data/processed/london_private_soc_bin_distribution.csv` for the
machine-readable versions, and `src/plot_initial_soc.py` to regenerate the chart.

## 4. What would change this

- A genuine fit of initial-SOC values from real London private-vehicle
  session data (home wallbox exports, smart-charging app data) would
  supersede this immediately.
- A London-specific (rather than UK-wide) version of the DfT-style survey
  question would let us drop the 50/50 blend and use real data directly.
- Splitting the "typical SOC before charging" question by
  location (home vs. work vs. public) in the DfT survey's underlying data
  would clarify whether the 24% median is representative of the mixed
  population Table 3 describes, or specific to home charging only — which
  would change how much weight it deserves in this blend.

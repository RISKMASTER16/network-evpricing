# Pricing scheme choice and the tail risk of EV charging demand on a London LV feeder

**Research question.** How does the choice of electricity pricing scheme affect both the central tendency and the tail risk of aggregate EV charging demand on a representative London low-voltage feeder, and what is the trade-off between network risk reduction and economic cost/predictability for consumers and operators?

**Short answer.** Pricing scheme choice barely moves the mean and moves the tail enormously. Across every scheme tested the mean feeder peak varies by about 12%, while the probability of exceeding the feeder rating varies by a factor of four. The two schemes that look best on average cost, Octopus Agile and Octopus Go, are the two that concentrate demand hardest and therefore carry the fattest tails. The proposed DToU + CPP + RTP scheme is the only one that reduces the tail while also reducing consumer cost, because it is the only scheme whose price is a function of demand and therefore has a fixed point. It is not the cheapest scheme on average, and it should not claim to be.

---

## 1. What was built

Three layers, each calibrated to something external rather than assumed.

**The feeder.** A representative London LV radial feeder carrying 75 households, the design maximum for a radial LV feeder in SP Energy Networks ESDD-02-012, fed from a 500 kVA ground-mounted distribution substation split across four feeders. At 0.95 power factor that gives a firm limit of 118.8 kW. Two network cases are carried through the whole study: a *base* case at nameplate rating on a winter weekday, and a *constrained* case at a 0.85 planning derate in a winter peak week, giving 100.9 kW. The ranking of tariffs is not invariant across the two, and that dependence is itself one of the results.

**The exogenous load.** The non-EV domestic background on that feeder is anchored to DESNZ *Subnational electricity consumption statistics 2005-2024*, sheet 2024, London (E12000007): 3,729,075 domestic meters consuming 12,084.705 GWh, giving 3,240.5 kWh per household per year. That energy is distributed over a stylised GB domestic (Elexon Profile Class 1) half-hourly shape. In the constrained case the resulting feeder peak is 72.0 kW, 0.96 kW per household after diversity, which sits below the 1.5 to 3 kVA per dwelling that DNOs use for *planning* but is consistent with measured behaviour on a gas-heated London street.

**The controllable load.** The step-1 Monte Carlo output was not used directly as a fixed profile. Instead a session-level model was fitted to it: plug-in time as a tri-modal Gaussian, energy per session from the Wei et al. lognormal daily-mileage distribution, 7.4 kW single-phase home charging. The fit reproduces the shape of `london_ev_load_profile.csv` with R² = 0.9913 (Figure 1) and back-implies 103,557 home-charging BEVs in London at 720 MWh/day and 29.6 kWh per session. That is consistent with DfT's reported 5.4% BEV share of the London car fleet at Q1 2025. The reason for going to session level is given in section 3.

Everything runs on a 48 × 30-minute grid, which is GB settlement resolution and the resolution Agile prices are published at. Khan & Aziz use 24 hourly slots; their equations are resolution-agnostic so this is a strict refinement.

---

## 2. Adapting Khan & Aziz from a Dhaka hub to a London street

The step-1 load curve cannot substitute for the `P_surplus`, `P_max`, `P_demand` system, and the reason matters. In equation (1), `P_demand` is the load *already on the network*, and the EV charging is the controllable load that gets fitted into the surplus underneath it. The step-1 curve is the controllable load. Feeding it in as `P_demand` prices EV demand against itself and the surplus curve stops meaning anything. That is why the DESNZ background had to be brought in.

The full mapping actually implemented:

| Khan & Aziz (Dhaka, centralised hub) | This study (London, distributed) |
|---|---|
| `P_max` = station supply capacity | LV feeder firm thermal limit, 118.8 kW (100.9 kW derated) |
| `P_demand` = residential feeder load | Non-EV domestic background, DESNZ-anchored |
| `P_charger` = 50 kW DC rapid | 7.4 kW single-phase AC home charger |
| `N_AC(i)` = chargers fitting under surplus | **Maximum households permitted to charge simultaneously in slot *i*** |
| `CPP` fires when the marginal charger has 80-100% of its power | `CPP` fires on **headroom utilisation** `u(i) = forecast simultaneous EV demand / headroom`; penalty for 0.8 ≤ u < 1, curtailment for u ≥ 1 |
| `RTP` re-forecasts 10 min ahead | Unchanged in structure; day-ahead forecast versus stochastic realisation |
| `T_DToU = C1·\|P_demand\|` computed once | Iterated to a **fixed point** before publication |

Two of these deserve elaboration.

**The CPP generalisation.** Equation (8) originally tests the fractional part of `N_AC`, which is a question about one marginal 50 kW charger. With a hundred small chargers that question has no content. The replacement keeps the economics exactly as the paper intended, that a slight overload is tolerated at a punitive price and anything beyond it is blocked, but expresses the trigger as a utilisation ratio rather than a rounding remainder. Figure 6 shows the resulting tariff, the adapted `N_ACs(i)` curve and the slots where CPP fires.

**The fixed point.** This is the change that does most of the work, and it is not in the original paper. `T_DToU` is an increasing function of demand, so the published price alters the demand it was computed from. Publishing the one-shot price sends every smart charger to the same cheapest slot. Iterating price → scheduled response → new demand → price until it stops moving (convergence in 12 to 20 iterations) produces a tariff that is self-limiting: filling a cheap slot raises its price, which pushes the marginal vehicle elsewhere. An exogenous price (flat, Go, Agile) has no such fixed point. Figure 2 shows the consequence directly: the proposed scheme's mean EV load is spread across roughly midnight to 06:00, whereas Go's is a spike at 00:30.

**Behavioural model.** Sessions have a plug-in time, an energy requirement, a departure deadline and a smart flag (80% smart by default). Smart sessions take the cheapest half-hours inside their dwell window. Where slots tie on price, which is exactly what a two-rate tariff produces across its whole cheap window, 75% behave as fixed timers and take the earliest tied slot. This is the mechanism that produces the documented timer peak, and it is applied identically to every tariff so no scheme is advantaged by it.

---

## 3. Why P10/P90 could not serve as the real-time scenarios

In `london_ev_load_profile.csv` the P10 to P90 band is ±3% of the mean. That is the Monte Carlo standard error of the mean of a London-wide fleet of a hundred thousand vehicles, not the day-to-day variability of a feeder carrying two to seventy-five of them. Used as RTP inputs it makes the real-time re-evaluation of `N_ACs` almost never differ from the day-ahead value, so RTP fires rarely, and every tail-risk metric in the study collapses to a point mass. Since tail risk is the entire subject of the research question, this had to be replaced.

The session-level model restores the variance that aggregation had averaged away: Poisson session counts, arrival coincidence, and lognormal energy per session. On a 75-household feeder at 60% penetration the daily peak has a standard deviation of roughly 12 kW and a visible right tail (Figure 3). The original percentile mode is retained behind `--rt-mode percentile` in `day_ahead_tariff.py` for continuity.

---

## 4. Sub-question 1: probability and severity of overload

Constrained feeder, 60% household penetration, 2,000 Monte Carlo days. Limit 100.9 kW.

| Scheme | Mean peak | PAR | VaR₉₅ | VaR₉₉ | CVaR₉₅ | P(overload) | Exceedance |
|---|---|---|---|---|---|---|---|
| Uncontrolled (flat) | 94.7 kW | 1.86 | 119.7 | 128.5 | 125.6 | 30.2% | 4.52 kWh/day |
| Static ToU (Go) | 85.2 kW | 1.59 | 107.8 | 122.3 | 118.0 | 9.9% | 1.17 |
| Agile (RTP) | 86.3 kW | 1.55 | 114.6 | 129.7 | 123.9 | 13.8% | 1.68 |
| **Proposed DToU+CPP+RTP** | **82.5 kW** | **1.55** | 106.4 | 123.7 | 116.8 | 7.8% | 1.28 |
| Proposed, tail-aware CPP | 82.5 | 1.55 | **105.5** | **120.5** | **115.2** | **7.5%** | **1.06** |

The proposed scheme has the lowest mean peak, the lowest VaR₉₅, the lowest VaR₉₉, the lowest CVaR₉₅ and the lowest overload probability. Relative to uncontrolled charging it removes 75% of overload days; relative to Octopus Go, 24%; relative to Agile, 46%.

The mechanism is visible in the spread of the metrics rather than their level. Static ToU has a *lower mean peak* than Agile but a comparable tail, because its cheap window is fixed and every timer fires at the same instant. Agile has the *lowest PAR* of the exogenous tariffs because its half-hourly prices are all distinct, but the worst tail, because on any given day every optimiser identifies the same cheapest half hours and converges on them. The proposed scheme achieves a PAR equal to Agile's and a tail better than either, and it does so by construction rather than luck.

Figure 4 traces this against penetration. Three regimes emerge. Below roughly 20% penetration nothing matters; the feeder absorbs everything and all schemes are indistinguishable on risk. Between 20% and 80% the ranking above holds and the gaps widen. Above 80% every scheme fails: at 100% penetration on the constrained feeder the overload probability is 56% under the best scheme and 68% under the worst. Pricing has run out of room and the answer is reinforcement or direct load control.

**An honest counter-result.** On the *base* (non-derated, winter-weekday) feeder above 80% penetration, uncontrolled charging has a lower CVaR₉₅ than any price-responsive scheme (123.6 kW against 134.7 to 141.5 kW). Natural arrival diversity beats any synchronising price signal when there is enough headroom to absorb the evening peak. This is a real finding and should be reported, not buried: price-based demand-side management is only network-beneficial on feeders that are already tight. It is also the reason the constrained case, not the base case, is the right headline scenario for this research question.

---

## 5. Sub-question 2: what risk reduction costs, and which schemes are Pareto-efficient

Constrained feeder, 60% penetration, all changes relative to uncontrolled charging.

| Scheme | CVaR₉₅ reduction | Overload reduction | Mean consumer cost | Consumer p95 cost | Operator margin | Cost s.d. | Pareto |
|---|---|---|---|---|---|---|---|
| Static ToU (Go) | 6.0% | 67.2% | −48.0% | −31.6% | +23.9% | 2.56 | no |
| Agile | 1.3% | 54.1% | **−70.9%** | −33.2% | **+35.3%** | 6.16 | yes |
| Proposed | 7.0% | 74.3% | −65.9% | −40.0% | +27.7% | 3.39 | yes |
| Proposed, tail-aware | **8.3%** | **75.2%** | −65.3% | **−39.8%** | +27.8% | 3.55 | yes |

Static ToU and uncontrolled charging are strictly Pareto-dominated by the proposed scheme: worse on risk *and* worse on consumer cost. Agile is not dominated. It and the proposed scheme sit at two different points on the same frontier, and the trade-off between them is the substantive answer to the second sub-question.

In level terms, at 60% penetration the proposed scheme costs consumers 8.92 p/kWh against Agile's 7.61 p/kWh, a 17% premium. In exchange it removes 7.2 kW of CVaR₉₅ and cuts overload probability from 13.8% to 7.8%. Two further columns matter and are easy to miss:

On **predictability**, which the research question asks about explicitly, the proposed scheme's day-to-day standard deviation of consumer cost is 3.39 p/kWh against Agile's 6.16 p/kWh. Agile is 82% more volatile. The day-ahead DToU price is deterministic given the background forecast, and only the RTP settlement component varies; Agile inherits the full variance of the wholesale market.

On **tail cost to consumers**, the proposed scheme is best outright: 15.67 p/kWh at the 95th percentile against Agile's 17.43, Go's 17.87 and 26.11 flat. The scheme that best protects the network from its tail also best protects consumers from theirs. That is not a coincidence; both tails come from the same source, which is everyone wanting energy at the same moment.

**Monetising the trade-off.** Because the axes are physical on one side and financial on the other, the cleanest comparison is a break-even shadow price: how much must a kWh of over-rating energy be worth before the proposed scheme's extra consumer cost is justified? Figure 9 gives it.

| Penetration | Break-even shadow price |
|---|---|
| 20% | £308.76 /kWh |
| 40% | £20.42 /kWh |
| 60% | £9.02 /kWh |
| 80% | £3.10 /kWh |

The GB reference values are £6,000/MWh (the administrative VoLL in the Balancing and Settlement Code), £16,940/MWh (the headline Ofgem/London Economics 2013 figure) and £25,301/MWh (Electricity North West's updated estimate). Against the headline £16.94/kWh, the proposed scheme becomes welfare-superior to Agile somewhere between 40% and 60% household penetration, and is comfortably superior thereafter. Below 40% Agile wins on welfare, because the risk it creates is not yet worth paying to avoid.

**This crossover is the headline result.** It converts "which tariff is best" into "which tariff is best *given how many EVs are on this feeder*", which is a question a DNO can actually act on.

---

## 6. Sub-question 3: risk-adjusted performance over a seasonal year

365 consecutive days with seasonal background level (1.35 in January falling to 0.83 in July) and seasonal wholesale level (10.6 p/kWh December, 5.9 p/kWh July), with the tariff republished monthly. Agile prices are constructed from Octopus's own published algorithm, `2.2 × W − 3.5 p/kWh` with a 12 p/kWh peak uplift between 16:00 and 19:00 and a 100 p/kWh cap, applied to a wholesale process calibrated to Ofgem's reported 6 to 10 p/kWh weekly range with overnight wind-driven negative excursions.

| Scheme | Operator margin | Sharpe | Annualised Sharpe | Overload days | CVaR₉₅ peak | Consumer spend |
|---|---|---|---|---|---|---|
| Uncontrolled | £11,749 | 1.87 | 35.8 | 52 | 114.8 kW | £26,406 |
| Static ToU (Go) | £14,595 | 2.41 | 46.1 | 21 | 107.2 kW | £13,772 |
| Agile | **£16,136** | 2.48 | 47.3 | 27 | 114.3 kW | **£8,336** |
| Proposed | £15,069 | 2.42 | 46.2 | **16** | 109.0 kW | £8,738 |
| Proposed, tail-aware | £15,559 | **2.52** | **48.2** | 17 | **106.9 kW** | £9,190 |

Over a full year on one feeder the proposed scheme delivers the fewest overload days, 16 against Agile's 27 and uncontrolled's 52, at a consumer cost £402 above Agile and £5,034 below Octopus Go. The tail-aware variant achieves the best risk-adjusted operator performance in the set (Sharpe 2.52) *and* the lowest annual CVaR, which is the cleanest single result in the study.

Operator margin Sharpe ratios for the three responsive schemes lie between 2.41 and 2.52 and should be treated as a statistical tie. The honest statement is that the proposed scheme costs the operator nothing in risk-adjusted terms relative to Agile while removing 40% of its overload days. It does give up about £1,070 a year in absolute margin, because Agile follows the procurement cost signal more closely and therefore buys energy 0.7 p/kWh cheaper.

---

## 7. Sub-question 4: the CVaR-constrained optimal price schedule

To ask what price schedule is optimal rather than merely comparing four fixed ones, the schemes were embedded in a three-parameter family that nests them all:

```
price_i = (1 − w)·cost_signal_i + w·network_signal_i + CPP(τ, λ)_i
```

with `w = 0` a pure cost-following tariff (Agile-like, buy when wholesale plus DUoS is cheapest), `w = 1` a pure demand-following tariff (the proposed DToU), and `(τ, λ)` the headroom-utilisation threshold and penalty. The optimiser is choosing how much of the price signal should point at the operator's procurement cost versus the network's spare capacity. Seventy-two configurations were evaluated at 500 Monte Carlo days each; Figure 7 shows the resulting frontier.

At 60% penetration on the constrained feeder:

The **unconstrained profit optimum** is `w = 0`, that is, pure cost-following, delivering £44.91/day at CVaR₉₅ = 117.1 kW, which is 116% of the feeder rating. Profit maximisation with no network term drives the tariff to Agile-like behaviour and an unsafe tail. That is the formal version of what Section 4 showed empirically.

The constraint **CVaR₉₅ ≤ rating is infeasible.** No price schedule in the family keeps the tail under 100.9 kW at 60% penetration. This is worth stating plainly: at this loading, pricing alone cannot make the feeder safe. Some combination of reinforcement, storage or direct curtailment is required, and a tariff study that claims otherwise is overselling.

Relaxing to **CVaR₉₅ ≤ 110% of rating**, a level a distribution transformer tolerates thermally for short periods, is feasible at `w = 0.4`, giving £43.83/day. The **cost of the safety constraint is 2.4% of operator margin.** Along the whole frontier, moving from CVaR₉₅ = 117.1 kW to 109.8 kW costs 4.6% of margin.

The policy reading is that the optimum is a blend, not a corner. A tariff that is 40% cost-following and 60% network-following captures nearly all the procurement value while removing most of the excess tail, and the fact that the optimum sits at an interior `w` is the strongest argument for the proposed scheme's architecture: it is the only one of the four benchmarks that has a network term at all.

---

## 8. Conclusions

**1. Pricing scheme choice is a tail-risk instrument, not a mean-load instrument.** Mean feeder peak varies by 12% across schemes; overload probability varies by a factor of four. Any evaluation that reports only average load or average PAR will conclude the schemes are near-equivalent and will be wrong.

**2. Exogenous dynamic pricing trades network risk for consumer savings, and the trade is not disclosed to anyone.** Octopus Agile delivers the lowest mean consumer cost (7.61 p/kWh) and the highest operator margin, and simultaneously the second-worst tail of any scheme tested. Octopus Go is Pareto-dominated outright. Neither price contains any information about the state of the feeder, so neither can do anything but synchronise the vehicles that respond to it.

**3. The proposed scheme is the best risk instrument in the set, and it is Pareto-efficient rather than dominant.** It has the lowest mean peak, VaR₉₅, VaR₉₉, CVaR₉₅ and overload probability, the lowest 95th-percentile consumer cost, and consumer-cost volatility 45% below Agile's. It costs 17% more than Agile on mean consumer cost and about £1,070 a year in operator margin. Claiming it is best on every axis would be false; claiming it is the only scheme that reduces network tail risk *and* consumer tail cost *and* consumer cost volatility simultaneously is true and is the stronger claim.

**4. The decisive mechanism is the fixed point, not the CPP tier.** At 60% penetration the CPP tier assessed on the point forecast fires in zero slots; nearly all the risk reduction comes from `T_DToU` being an increasing function of demand and being iterated to convergence before publication. Any demand-proportional tariff computed one-shot will herd exactly like Agile does. This is the single most important implementation detail and it is absent from the source paper.

**5. Assessing CPP on the median forecast systematically under-fires it.** The constraint CPP exists to protect is a tail property of the peak distribution, so testing it against a point forecast is a category error. Evaluating utilisation at the 90th percentile of forecast demand makes the tier bind (five slots at 100% penetration against two), and over the seasonal year it improves CVaR₉₅ from 109.0 to 106.9 kW and Sharpe from 2.42 to 2.52. At single penetration levels the improvement is small and partly within Monte Carlo noise, so this is offered as a recommended refinement rather than a demonstrated large effect.

**6. There is a penetration threshold below which the proposed scheme is not worth deploying.** Against Ofgem's headline VoLL of £16.94/kWh, the break-even crossover with Agile lies between 40% and 60% household penetration. Below it, Agile is welfare-superior. Above 80%, no tariff is sufficient and the question becomes a reinforcement question.

**7. Price-based DSM only helps networks that are already tight.** On the non-derated base feeder above 80% penetration, uncontrolled charging has the *lowest* CVaR of any scheme, because arrival diversity beats a synchronising signal when headroom is abundant. The benefit of the proposed scheme is conditional on the constrained case, and the study should say so.

**8. The safety constraint is cheap where it is feasible at all.** Holding CVaR₉₅ to 110% of rating costs 2.4% of operator margin. The optimum is an interior blend of cost-following and network-following signals, not either corner.

---

## 9. Limitations, and what would strengthen this

The RTP recursion in equations (12) and (13) is unbounded. Because each slot's adjustment is spread over all remaining slots, a day whose background forecast is consistently wrong in one direction accumulates a large drift. On the worked day in `day_ahead_tariff.py`, RTP moved the mean settled price from roughly 8.9 to 6.61 p/kWh, a 26% revenue swing from forecast error alone. A cap of ±20% of `T_DToU`, or a daily-total reconciliation, should be added before this is deployed anywhere. RTP also only adjusts settlement, not behaviour, faithfully to the paper; a version where RTP is published in advance of the slot would change the load results and is worth testing.

The Agile price series is constructed from Octopus's published algorithm applied to a wholesale process calibrated to Ofgem's reported ranges, not to a downloaded half-hourly Agile history. Re-running Section 6 against real region-C Agile data from the Octopus public API (`AGILE-24-10-01`, tariff `E-1R-AGILE-24-10-01-C`) would remove the last modelled input from the backtest and is the highest-value next step.

The background domestic profile shape is a stylised Profile Class 1 curve scaled to the DESNZ energy total. Substituting measured half-hourly data from UK Power Networks' Low Carbon London household trial (5,567 London households, available on the London Datastore) would replace the one remaining shape assumption with London-specific measurement, and would also let the study be parametrised by borough, since the DESNZ table already carries borough-level domestic consumption and meter counts.

Charging behaviour assumes 80% smart chargers and 75% timer behaviour within tied price bands. Both are plausible but neither is measured. The timer fraction in particular drives the static ToU result and deserves a sensitivity sweep.

Finally, the whole study is single-feeder. Aggregating to a substation, a primary and a GSP group would show diversity progressively suppressing the tails, and would show at what network level pricing stops being a risk instrument at all.

---

## Files

| File | Contents |
|---|---|
| `src/feeder_model.py` | Feeder definition, DESNZ-anchored background demand, EV session model and calibration to step 1 |
| `src/pricing.py` | All five tariffs, the adapted Khan & Aziz equations, the fixed-point solver, charging-behaviour model, settlement |
| `src/day_ahead_tariff.py` | Single-day tariff construction and RTP settlement, standalone |
| `src/run_experiments.py` | Monte Carlo across schemes, penetrations and network cases |
| `src/analysis.py` | Efficient frontier, CVaR-constrained optimisation, 365-day backtest |
| `src/figures.py` | Full step-3 pipeline and all nine figures |
| `results/metrics_by_scheme.csv` | Full metric table, both cases, six penetrations, five schemes |
| `results/frontier_*.csv`, `cvar_policy_grid.csv`, `backtest_*.csv`, `breakeven_shadow_price.csv` | Per-section outputs |

**Sources.** DESNZ Subnational electricity consumption statistics 2005-2024. SP Energy Networks ESDD-02-012 LV design framework. UK Power Networks London Power Networks LC14 Use of System Charging Statement, effective 1 April 2026 (DUoS red/amber/green). Ofgem energy price cap, 1 July to 30 September 2026 (26.11 p/kWh electricity unit rate). Octopus Energy published Agile pricing algorithm and Octopus Go rates. Ofgem / London Economics, *The Value of Lost Load for Electricity in GB* (2013). DfT vehicle licensing statistics and GLA London EV data. Wei et al., *Global Energy Interconnection* 9 (2026) 131-142. Khan and Aziz, GECOST 2022.

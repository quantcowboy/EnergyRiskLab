# Model card: historical-simulation VaR and Expected Shortfall

| | |
|---|---|
| Model | Historical-simulation VaR / ES, tenor-matched, with a delta-normal benchmark |
| Code | `erl.risk.scenarios`, `erl.risk.var`, `erl.risk.backtest` |
| Version | 0.1.0 |
| Owner | Pierre Brunelle |
| Status | Development (synthetic data) |

## 1. Purpose and scope

Measures the market risk of a linear natural gas book (fixed-price and basis swaps)
over 1-day and 5-day horizons, at 95% and 99% confidence, per desk and in total.
Out of scope in this version: options (non-linear), the current delivery month,
credit and liquidity risk.

## 2. Method

1. **Exposure.** Each trade is split into delivery months. A fixed-price swap gives
   volume x days at its hub; a basis swap gives the same at its hub and the opposite
   at Henry Hub. Only months after the as-of month are included.
2. **Tenor mapping.** Each exposure is labelled with its tenor on the as-of date
   (0 = prompt month).
3. **Scenarios.** For every historical day, the price change of each contract over the
   horizon is measured on the same contract and labelled with that contract's tenor
   on the end date. Multi-day horizons use overlapping windows.
4. **Scenario P&L** = sum over hubs and tenors of exposure x change.
5. **VaR** = minus the (1 - alpha) quantile of scenario P&L (linear interpolation).
   **ES** = minus the mean of the P&L at or below that quantile.
6. **Benchmark.** Delta-normal VaR = z x sigma, ES = sigma x phi(z) / (1 - alpha), with
   sigma from the sample covariance of the same scenarios and a zero mean.

## 3. Assumptions

- Past price changes at a tenor are representative of future changes at that tenor.
- The book is static over the horizon (no trading, no roll-off).
- P&L is linear in price: no gamma, no discounting, no volume optionality.
- Overlapping 5-day windows are used as if independent for the quantile; this does not
  bias the quantile but reduces its effective sample size.

## 4. Limitations

- **Short history.** 500 scenarios give a 99% VaR from about 5 observations; ES at
  99% rests on those same few days.
- **Regime dependence.** A calm window understates risk after a structural change;
  a window containing an extreme event overstates it until the event rolls off.
- **Back tenors.** A tenor with no history raises an error rather than being given
  zero risk; on roll days the newest back contract has no change and that day is
  dropped for any book holding it.
- **Normal benchmark** understates tail risk for spiky basis (fat tails) and
  overstates it when one extreme day inflates the variance; the two methods are shown
  side by side for that reason.

## 5. Testing

| Test | Result |
|---|---|
| Historical VaR and ES of 400,000 standard normal draws vs closed form (95%, 99%) | Within 1% |
| ES >= VaR on fat-tailed (t, 3 d.f.) P&L | Passes |
| Delta-normal sigma and VaR of a known two-asset covariance | Within 1% |
| Tenor changes on a hand-built market (same contract, horizon 1 and 2) | Exact |
| Scenario P&L on a hand-built market; missing history raises | Exact |
| Kupiec: 10 exceptions in 1,000 days at 99% gives p = 1; 0 or 30 reject | Passes |
| Christoffersen: evenly spread exceptions accepted, a cluster of 10 rejected | Passes |
| Rolling backtest on normal P&L: exception rate 1% +/- 0.5% | Passes |
| Demo backtest (seed 7): 11 exceptions in 792 days vs 7.9 expected; Kupiec p = 0.30, Christoffersen p = 0.58 | Not rejected |

## 6. Change log

| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2026-11-22 | First version |

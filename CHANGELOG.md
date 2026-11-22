# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org).

## [0.2.0] - 2026-11-22

### Added
- Historical stress by calendar month (`erl.risk.stress`): for the as-of month and each
  of the next 12, the book as it will stand then (delivered months rolled off), shifted
  back by whole years so each position keeps its distance to delivery, run through every
  5-day window ending in that calendar month in prior years. Reports the worst net window
  with its long and short legs from the same window, dates, main losing hubs, P5 and
  median, per desk and in total.
- Hypothetical shocks (`erl.risk.shocks`) defined in a small table: Henry Hub parallel
  and prompt-only moves, basis blowouts and collapses by hub (Algonquin, TETCO M3,
  Chicago and SoCal in winter; Waha over the first year), and a winter-summer spread
  shock; P&L per desk and by hub.
- Model card `docs/model_cards/stress_testing.md`; 8 tests on hand-built examples.
- Demo script prints the stress tables and writes them to `outputs/`.

### Changed
- README: project name written as EnergyRiskLab; stress testing documented.

## [0.1.0] - 2026-11-22

### Added
- Synthetic forward curves: two-factor Henry Hub with seasonality and winter volatility,
  seven regional hubs with basis winter premia and correlated shocks, a cold-snap event.
- Fictional trade blotter (fixed-price and basis swaps, three desks).
- Exposure by hub and delivery month; mark-to-market.
- Tenor-matched historical scenarios (1-day and overlapping h-day).
- Historical and delta-normal VaR and Expected Shortfall.
- Rolling VaR backtest with Kupiec and Christoffersen tests.
- Demo script, 18 tests, CI, model card for historical VaR / ES.

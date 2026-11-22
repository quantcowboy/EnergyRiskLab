# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org).

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

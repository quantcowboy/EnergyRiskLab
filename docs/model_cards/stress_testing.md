# Model card: stress testing

| | |
|---|---|
| Model | Historical stress by calendar month; hypothetical price shocks |
| Code | `erl.risk.stress`, `erl.risk.shocks` |
| Version | 0.2.0 |
| Owner | Pierre Brunelle |
| Status | Development (synthetic data) |

## 1. Purpose and scope

VaR describes an ordinary bad day or week from recent history. Stress testing asks
two other questions about the same linear book (fixed-price and basis swaps):

- **Historical:** what would the worst 5-day window seen in a given calendar month
  (a January, a November) do to the book as it will stand in that month?
- **Hypothetical:** what do named, hand-picked moves (Henry Hub up $1, a winter
  Algonquin basis blowout) do to today's book?

Both are reported per desk and for the total book. Out of scope in this version:
options, the current delivery month, liquidity (time to exit), and any
probability attached to a scenario.

## 2. Method

### 2.1 Historical stress by calendar month

For the as-of month and each of the next 12 months (the *stress month* M):

1. **Book in M.** Today's exposures (desk, hub, delivery month, MMBtu), static: no
   new trades, and every delivery month up to and including M has rolled off.
   Exposure is netted per (hub, delivery month) within the book being stressed.
2. **Shift.** For each prior year with history (M minus 12, 24, ... months), each
   position is moved back by the same whole number of months. A position k months
   from delivery in M sits on the contract k months from delivery in the same month
   of that year, so it is shocked with how contracts at that distance moved, in that
   season.
3. **Windows.** Every 5-business-day window whose end date falls in that calendar
   month (the start may fall in the month before). The price change is measured on
   the same contract from the window's start to its end.
4. **P&L** of a window = sum of exposure x change. It is split into the **long leg**
   (positions with positive net exposure) and the **short leg** (negative), from that
   same window, and into P&L by hub.
5. **Report** per book and stress month: number of windows and years, the worst net
   window with its long and short legs, its dates and its three largest losing hubs
   (*main hubs*), and the 5th percentile (P5, linear interpolation) and median of all
   window P&Ls.

Only history up to the as-of date is used, and only years before the stress month's
year, so the as-of month is never stressed with its own (partial) data.

### 2.2 Hypothetical shocks

A shock table (`erl.risk.shocks.DEFAULT_SHOCKS`) has one row per price move:
scenario name, hub, tenor range on the as-of date, delivery calendar months, and a
shift in $/MMBtu. Rows sharing a scenario name are applied together. The hub is
either `all hubs` (a Henry Hub move that every regional hub follows, basis
unchanged) or one hub (a basis move; Henry Hub unchanged). P&L = sum of exposure x
shift. A basis swap is therefore flat to a Henry Hub parallel move and exposed to
its own hub's basis move.

Default scenarios:

| Scenario | Move |
|---|---|
| HH parallel +/- $1.00 | All hubs, all tenors |
| HH prompt +$2.00 / -$1.50 | All hubs, prompt month (tenor 0) only |
| Algonquin basis +$10 (winter) | Algonquin Citygates, Dec-Feb contracts |
| TETCO M3 basis +$5 (winter) | TETCO M3, Dec-Feb contracts |
| Chicago basis +$2 (winter) | Chicago Citygate, Dec-Feb contracts |
| SoCal basis +$3 (winter) | SoCal Citygate, Dec-Feb contracts |
| Waha basis -$2.00 (first year) | Waha, tenors 0-11 |
| Algonquin basis -$5 (winter) | Algonquin Citygates, Dec-Feb contracts |
| TETCO M3 basis -$2 (winter) | TETCO M3, Dec-Feb contracts |
| Chicago basis -$0.50 (winter) | Chicago Citygate, Dec-Feb contracts |
| SoCal basis -$1 (winter) | SoCal Citygate, Dec-Feb contracts |
| Waha basis +$1.00 (first year) | Waha, tenors 0-11 |
| Winter-summer spread +$1.00 | All hubs: Dec-Feb +$0.75, Apr-Oct -$0.25 (Nov and Mar unchanged) |

Each basis move comes in both directions, so a book that is long a hub's basis and
one that is short it both see a loss. The blowouts are the larger move; the
collapses take away about the whole synthetic winter premium of the hub (for Waha,
the discount to Henry Hub narrows).

The sizes are judgement calls on a synthetic market, chosen to be large relative to
the 5-day moves in the generated history; they are not calibrated to any real event.

## 3. Assumptions

- The book is static from today to the stress month: no hedging, no new trades.
- Seasonality is the main thing that makes one month's history relevant to another:
  windows are taken from the same calendar month, and distance to delivery is kept.
- A 5-day horizon is long enough to cover an exit from the positions.
- P&L is linear in price (no gamma, no discounting, no volume optionality), as in the
  VaR model.
- Hypothetical shifts are additive in $/MMBtu, the same on every contract a row hits.

## 4. Limitations

- **Few years.** The synthetic history has three or four years per calendar month,
  about 65-90 windows. The worst window is one observation and P5 rests on about four;
  a month in which history held no event shows little stress. The one cold snap in
  the history (January 2024) only reaches January stress months.
- **Overlapping windows.** Consecutive windows share four of five days, so the
  windows are far from independent and P5 and median are descriptive, not
  estimates with a known error.
- **Netting.** Long and short legs are split after netting each (hub, delivery month)
  within the book; a book's gross positions within one contract are not shown.
- **Main hubs** lists the three hubs that lost most in the worst window; a hub that
  gained offsets them in the net and is not listed.
- **Missing prices.** A shifted position with no history at all raises an error.
  A window where a contract has no price at its start or end is dropped, which can
  shorten the sample silently for back contracts near the end of the curve.
- **Hypothetical shocks** move one hub's basis at a time; a blowout at one hub with a
  collapse at another (a spread between regions) is not in the default table. The
  table is meant to be edited and extended for the book at hand. No correlation or
  probability is implied between scenarios.

## 5. Testing

Tests in `tests/test_stress.py`, all on hand-built data with answers worked out by
hand:

| Test | Result |
|---|---|
| Two-curve market with a +$1.00 jump on 17 Jan 2024; book long 100,000 X Mar-25, short 200,000 Henry Feb-25, stressed in Jan 2025: 23 windows, 18 at +5,000, 5 at -95,000 (long +105,000, short -200,000) | Exact |
| Summary of the same: worst -95,000 with legs from the same window, main hubs "Henry Hub -200k", P5 -95,000, median +5,000 | Exact |
| Forward month (Feb 2025): the delivered Feb-25 position rolls off; 21 windows at +5,000 | Exact |
| Only prior years and history up to the as-of date are used; desk and total books | Exact |
| A shifted position with no history raises | Passes |
| Default shocks on a hand-built four-line book: parallel, prompt, Algonquin winter basis blowout and collapse, Waha first-year basis both ways, winter-summer spread | Exact |
| A basis swap is flat to a Henry Hub parallel move and gains 310,000 x 5 on a +$5 TETCO M3 basis move; by-hub split | Exact |
| A custom shock table; a bad tenor range or missing column raises | Passes |

## 6. Change log

| Version | Date | Change |
|---|---|---|
| 0.2.0 | 2026-11-22 | First version |

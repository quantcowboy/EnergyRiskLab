"""Historical stress by calendar month.

VaR answers "how bad is a normal bad day"; this module answers "what did the worst
week of a January do to a book like ours". For a *stress month* M (the as-of month
and each of the next months) it:

1. Takes the book as it will stand in M: today's trades, static, with every delivery
   month up to and including M rolled off.
2. For each prior year with history, shifts every position back by whole years
   (12, 24, ... months), so a position three months from delivery in M is placed on
   the contract that was three months from delivery in the same month of that year.
3. Runs the shifted book through every `horizon`-day window that ends in that month:
   P&L = sum over (hub, contract) of exposure x price change on the same contract
   from the window's start to its end.

Each window's P&L is also split into its **long leg** (positions with positive net
exposure at a hub and delivery month) and **short leg** (negative net exposure),
both from the same window, so a small net loss that hides two large offsetting legs
is visible.

Only history up to the as-of date is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.synthetic import month_index

TOP_HUBS = 3  # hubs listed as the main contributors to the worst window


def _first_of_month(m) -> pd.DatetimeIndex:
    m = np.atleast_1d(np.asarray(m))
    return pd.DatetimeIndex(pd.to_datetime({"year": m // 12, "month": m % 12 + 1, "day": 1}))


def price_panel(market: pd.DataFrame, asof=None) -> pd.DataFrame:
    """Prices as a wide table: one row per price date, one column per (hub, contract).
    Rows after `asof` are dropped."""
    m = market if asof is None else market[market["obs_date"] <= pd.Timestamp(asof)]
    return m.pivot(index="obs_date", columns=["hub", "contract"], values="price").sort_index()


def book_in_month(exposures: pd.DataFrame, month) -> pd.DataFrame:
    """The book as it stands in `month`: delivery months up to and including `month`
    are gone. Columns desk, hub, contract, mmbtu."""
    keep = month_index(exposures["contract"]) > month_index(pd.Timestamp(month))
    return exposures[keep]


def net_by_contract(exposures: pd.DataFrame) -> pd.Series:
    """Net exposure per (hub, contract), zeros dropped."""
    s = exposures.groupby(["hub", "contract"])["mmbtu"].sum()
    return s[s != 0]


def month_windows(
    exposure: pd.Series,
    panel: pd.DataFrame,
    month,
    horizon: int = 5,
) -> pd.DataFrame:
    """Every `horizon`-day window ending in the calendar month of `month`, in each
    year before it, for a book `exposure` indexed by (hub, contract) as it stands in
    `month`.

    One row per window: year, shift (months back), start, end, pnl, long, short,
    and one column per hub with that hub's P&L. Windows where a position has no
    price at the start or end (a contract not yet listed) are dropped. Raises if a
    shifted position has no price history at all.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    month = pd.Timestamp(month)
    target = month_index(month)
    dates = panel.index
    hubs = sorted(exposure.index.get_level_values("hub").unique())
    rows = []
    for years_back in range(1, month.year - dates[0].year + 1):
        shift = 12 * years_back
        in_month = month_index(dates) == target - shift
        ends = np.flatnonzero(in_month)
        ends = ends[ends >= horizon]  # need a start date `horizon` days earlier
        if ends.size == 0:
            continue
        if exposure.empty:
            hub_pnl = np.zeros((ends.size, 0))
            long_ = short = np.zeros(ends.size)
        else:
            cols = pd.MultiIndex.from_arrays(
                [
                    exposure.index.get_level_values("hub"),
                    _first_of_month(
                        month_index(exposure.index.get_level_values("contract")) - shift
                    ),
                ]
            )
            missing = [c for c in cols if c not in panel.columns]
            if missing:
                raise ValueError(
                    f"no price history for {len(missing)} shifted position(s), e.g. {missing[:3]}"
                )
            px = panel[cols].to_numpy()
            chg = px[ends] - px[ends - horizon]  # same contract, start to end
            w = exposure.to_numpy()
            ok = np.isfinite(chg).all(axis=1)
            ends, chg = ends[ok], chg[ok]
            leg_pnl = chg * w
            long_ = leg_pnl[:, w > 0].sum(axis=1)
            short = leg_pnl[:, w < 0].sum(axis=1)
            hub_of = exposure.index.get_level_values("hub")
            hub_pnl = np.column_stack([leg_pnl[:, hub_of == h].sum(axis=1) for h in hubs])
        part = pd.DataFrame(
            {
                "year": month.year - years_back,
                "shift": shift,
                "start": dates[ends - horizon],
                "end": dates[ends],
                "pnl": long_ + short,
                "long": long_,
                "short": short,
            }
        )
        rows.append(part.join(pd.DataFrame(hub_pnl, columns=hubs, index=part.index)))
    if not rows:
        return pd.DataFrame(columns=["year", "shift", "start", "end", "pnl", "long", "short"])
    return pd.concat(rows, ignore_index=True)


def main_hubs(window_row: pd.Series, hubs, top: int = TOP_HUBS) -> str:
    """The hubs that lost most in one window, e.g. "Waha -310k, Henry Hub -95k"."""
    losses = window_row[list(hubs)].astype(float)
    losses = losses[losses < 0].sort_values().head(top)
    return ", ".join(f"{h} {v / 1e3:,.0f}k" for h, v in losses.items())


def summarise(windows: pd.DataFrame) -> dict:
    """Worst net window (with its legs, dates and main hubs), P5 and median."""
    if windows.empty:
        return {"windows": 0, "years": 0}
    hubs = [c for c in windows.columns if c not in _WINDOW_FIELDS]
    worst = windows.loc[windows["pnl"].idxmin()]
    pnl = windows["pnl"].to_numpy(dtype=float)
    return {
        "windows": len(windows),
        "years": windows["year"].nunique(),
        "worst": float(worst["pnl"]),
        "worst_long": float(worst["long"]),
        "worst_short": float(worst["short"]),
        "worst_start": worst["start"],
        "worst_end": worst["end"],
        "main_hubs": main_hubs(worst, hubs),
        "p5": float(np.quantile(pnl, 0.05)),
        "median": float(np.median(pnl)),
    }


_WINDOW_FIELDS = ("year", "shift", "start", "end", "pnl", "long", "short")


def stress_by_month(
    exposures: pd.DataFrame,
    market: pd.DataFrame,
    asof,
    months_ahead: int = 12,
    horizon: int = 5,
    by_desk: bool = True,
) -> pd.DataFrame:
    """Historical stress for the as-of month and each of the next `months_ahead`
    months, per desk (if `by_desk`) and for the total book.

    `exposures` has columns desk, hub, contract, mmbtu (see `erl.positions.exposures`).
    Returns one row per (book, stress month) with the summary of `summarise`.
    """
    asof = pd.Timestamp(asof)
    panel = price_panel(market, asof)
    books = {"Total": exposures}
    if by_desk:
        books = {d: g for d, g in exposures.groupby("desk")} | books
    rows = []
    for k in range(months_ahead + 1):
        month = _first_of_month(month_index(asof) + k)[0]
        for name, e in books.items():
            expo = net_by_contract(book_in_month(e, month))
            w = month_windows(expo, panel, month, horizon)
            rows.append({"book": name, "month": month, **summarise(w)})
    return pd.DataFrame(rows)

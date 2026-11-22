"""A fictional trade blotter.

Trade types:

* ``fixed_price``: a fixed-price swap at one hub. Positive volume = the desk buys
  gas at the fixed price (long the hub's outright price).
* ``basis``: a basis swap. Positive volume = long the hub, short Henry Hub, in
  equal volume.

Volumes are MMBtu per day over a strip of whole delivery months. Fixed prices
are the day-weighted market price of the strip on the trade date, plus a little
noise, so trades start close to at the money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic import HENRY, month_index

DEFAULT_DESKS: dict[str, list[str]] = {
    "Desk A": ["Henry Hub", "Houston Ship Channel", "Waha"],
    "Desk B": ["Chicago Citygate", "Dominion South", "TETCO M3"],
    "Desk C": ["Algonquin Citygates", "SoCal Citygate", "TETCO M3"],
}
STRIP_LENGTHS = (1, 1, 3, 5, 12)  # months; 5 = a winter strip (Nov-Mar)


def generate_trades(
    market: pd.DataFrame,
    asof: str | pd.Timestamp,
    n_trades: int = 150,
    desks: dict[str, list[str]] | None = None,
    lookback_days: int = 250,
    seed: int = 11,
) -> pd.DataFrame:
    """Trades done over the `lookback_days` business days up to `asof`, priced off
    `market`. Only trades still delivering after the as-of month are returned."""
    desks = desks or DEFAULT_DESKS
    rng = np.random.default_rng(seed)
    asof = pd.Timestamp(asof)
    days = np.sort(market["obs_date"].unique())
    days = days[days <= np.datetime64(asof)][-lookback_days:]
    curve = market.set_index(["obs_date", "hub", "contract"])["price"]
    max_tenor = int(market["tenor"].max())

    rows = []
    desk_names = list(desks)
    for i in range(n_trades):
        desk = desk_names[i % len(desk_names)]
        hub = desks[desk][rng.integers(len(desks[desk]))]
        kind = "fixed_price" if hub == HENRY or rng.random() < 0.5 else "basis"
        tdate = pd.Timestamp(days[rng.integers(len(days))])
        length = int(rng.choice(STRIP_LENGTHS))
        first = month_index(tdate) + 1 + int(rng.integers(0, 15))
        if length == 5:  # winter strip starts in November
            first = first + (10 - first % 12) % 12
        limit = month_index(tdate) + 1 + max_tenor  # last contract priced that day
        first = min(first, limit)
        last = min(first + length - 1, limit)
        months = pd.to_datetime(
            {
                "year": np.arange(first, last + 1) // 12,
                "month": np.arange(first, last + 1) % 12 + 1,
                "day": 1,
            }
        )
        px = [curve.get((tdate, hub, m), np.nan) for m in months]
        if kind == "basis":
            px = [p - curve.get((tdate, HENRY, m), np.nan) for p, m in zip(px, months, strict=True)]
        days_in = months.dt.days_in_month.to_numpy()
        fixed = float(np.nansum(np.array(px) * days_in) / days_in.sum()) + rng.normal(0, 0.02)
        volume = float(rng.choice([-1, 1]) * rng.integers(1, 9) * 2_500)
        rows.append(
            {
                "trade_id": f"T{i + 1:05d}",
                "desk": desk,
                "trade_type": kind,
                "hub": hub,
                "trade_date": tdate,
                "start": months.iloc[0],
                "end": months.iloc[-1],
                "volume_per_day": volume,
                "fixed_price": round(fixed, 4),
            }
        )
    t = pd.DataFrame(rows)
    live = month_index(t["end"]) > month_index(asof)
    return t[live].reset_index(drop=True)

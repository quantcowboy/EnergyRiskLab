"""Trades to price exposure, by hub and delivery month.

Exposure is the volume, in MMBtu, whose value moves one-for-one with the hub's
price for that delivery month: a $0.10 rise on 100,000 MMBtu of long exposure is
a $10,000 gain.

* A fixed-price swap gives volume x days in the month at its hub.
* A basis swap gives the same exposure at its hub, and the opposite at Henry Hub.
* Only delivery months after the as-of month count. The current month is
  treated as already delivering, and is left out in this version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data.synthetic import HENRY, month_index


def explode_months(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per trade and delivery month, with that month's volume in MMBtu."""
    if trades.empty:
        return trades.assign(contract=pd.NaT, mmbtu=0.0)
    first, last = month_index(trades["start"]), month_index(trades["end"])
    n = last - first + 1
    idx = np.repeat(np.arange(len(trades)), n)
    m = np.concatenate([np.arange(a, b + 1) for a, b in zip(first, last, strict=True)])
    out = trades.iloc[idx].reset_index(drop=True)
    out["contract"] = pd.to_datetime({"year": m // 12, "month": m % 12 + 1, "day": 1})
    out["mmbtu"] = out["volume_per_day"] * out["contract"].dt.days_in_month
    return out


def exposures(trades: pd.DataFrame, asof) -> pd.DataFrame:
    """Exposure per desk, hub and contract month: columns desk, hub, contract, mmbtu."""
    legs = explode_months(trades)
    legs = legs[month_index(legs["contract"]) > month_index(pd.Timestamp(asof))]
    hub_leg = legs[["desk", "hub", "contract", "mmbtu"]]
    basis = legs[legs["trade_type"] == "basis"]
    henry_leg = basis.assign(hub=HENRY, mmbtu=-basis["mmbtu"])[["desk", "hub", "contract", "mmbtu"]]
    out = pd.concat([hub_leg, henry_leg], ignore_index=True)
    out = out.groupby(["desk", "hub", "contract"], as_index=False)["mmbtu"].sum()
    return out[out["mmbtu"] != 0].reset_index(drop=True)


def mark_to_market(trades: pd.DataFrame, curves: pd.DataFrame, asof) -> pd.DataFrame:
    """Undiscounted value of each trade against one day's curves (hub, contract, price).
    Positive = the trade is worth money to the desk."""
    legs = explode_months(trades)
    legs = legs[month_index(legs["contract"]) > month_index(pd.Timestamp(asof))]
    px = curves.set_index(["hub", "contract"])["price"]
    hub_px = px.reindex(pd.MultiIndex.from_frame(legs[["hub", "contract"]])).to_numpy()
    henry_px = px.reindex(
        pd.MultiIndex.from_arrays([[HENRY] * len(legs), legs["contract"]])
    ).to_numpy()
    floating = np.where(legs["trade_type"] == "basis", hub_px - henry_px, hub_px)
    legs = legs.assign(value=(floating - legs["fixed_price"]) * legs["mmbtu"])
    return legs.groupby(["trade_id", "desk"], as_index=False)["value"].sum()

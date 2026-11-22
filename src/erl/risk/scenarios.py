"""Historical scenarios, matched by tenor.

A position in "the contract three months out" should be shocked with how
three-months-out contracts moved in the past, not with how one calendar contract
moved. So each historical price change is measured on the SAME contract
(price today minus price `horizon` business days ago) and labelled with that
contract's tenor on the day the change ends (0 = prompt month).

Today's exposure is then mapped to tenors the same way, and each historical day
gives one scenario P&L: the sum over hubs and tenors of exposure x change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.synthetic import month_index


def tenor_changes(market: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Price changes over `horizon` business days, per (hub, tenor).

    Returns a frame indexed by the end date of each change, with columns
    (hub, tenor). With horizon > 1 the windows overlap (one per day).
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    wide = market.pivot(index="obs_date", columns=["hub", "contract"], values="price").sort_index()
    chg = wide - wide.shift(horizon)
    long = chg.melt(ignore_index=False, value_name="chg").dropna(subset=["chg"]).reset_index()
    prompt = month_index(long["obs_date"]) + 1
    long["tenor"] = month_index(long["contract"]) - prompt
    out = long.pivot(index="obs_date", columns=["hub", "tenor"], values="chg").sort_index()
    return out.sort_index(axis=1)


def exposure_by_tenor(exposures: pd.DataFrame, asof) -> pd.Series:
    """Exposure (desk, hub, contract, mmbtu) summed into (hub, tenor) as of `asof`."""
    tenor = month_index(exposures["contract"]) - (month_index(pd.Timestamp(asof)) + 1)
    s = exposures.assign(tenor=tenor).groupby(["hub", "tenor"])["mmbtu"].sum()
    return s[s != 0]


def scenario_pnl(exposure: pd.Series, changes: pd.DataFrame) -> pd.Series:
    """One P&L per historical day: exposure (hub, tenor) x that day's changes.

    Raises if a position has no history at its tenor, rather than silently
    giving it zero risk. Days where a used tenor has no change (the back
    contract on a roll day) are dropped.
    """
    missing = [k for k in exposure.index if k not in changes.columns]
    if missing:
        raise ValueError(f"no price history for {len(missing)} position(s), e.g. {missing[:3]}")
    used = changes[exposure.index].dropna()
    pnl = used.to_numpy() @ exposure.to_numpy()
    return pd.Series(pnl, index=used.index, name="pnl")


def window(changes: pd.DataFrame, end, days: int | None = None) -> pd.DataFrame:
    """The scenarios up to and including `end`, the last `days` of them if given."""
    c = changes.loc[: pd.Timestamp(end)]
    return c.iloc[-days:] if days else c


def coverage(exposure: pd.Series, changes: pd.DataFrame) -> float:
    """Share of gross exposure that has price history at its tenor."""
    have = np.array([k in changes.columns for k in exposure.index])
    gross = np.abs(exposure.to_numpy())
    return float(gross[have].sum() / gross.sum()) if gross.sum() else 1.0

"""Hypothetical shocks: price moves chosen by hand, not taken from history.

A shock is a small table. Each row moves the price of some contracts by a fixed
amount in $/MMBtu; a scenario can have several rows (the winter-spread shock moves
winter up and summer down). Columns:

* ``scenario``: name; rows with the same name are applied together.
* ``hub``: ``ALL_HUBS`` for a Henry Hub move that every hub follows (regional prices
  are Henry Hub plus a basis, and the basis is left unchanged), or one hub's name
  for a move of that hub only (a basis move: Henry Hub is unchanged).
* ``tenor_from``, ``tenor_to``: tenors on the as-of date (0 = prompt month), both
  inclusive; ``tenor_to`` = None means no upper limit.
* ``months``: calendar months of delivery the row applies to (e.g. (12, 1, 2)), or
  None for all.
* ``shift``: price change, $/MMBtu.

P&L is linear: exposure (MMBtu) x shift, summed over the contracts each row hits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.synthetic import HENRY, month_index

ALL_HUBS = "all hubs"
WINTER = (12, 1, 2)
SUMMER = (4, 5, 6, 7, 8, 9, 10)
_COLUMNS = ["scenario", "hub", "tenor_from", "tenor_to", "months", "shift"]

# fmt: off
DEFAULT_SHOCKS = pd.DataFrame(
    [
        # scenario                          hub                    from  to    months  $/MMBtu
        # Henry Hub moves, followed by every hub
        ("HH parallel +$1.00",              ALL_HUBS,              0,    None, None,   1.00),
        ("HH parallel -$1.00",              ALL_HUBS,              0,    None, None,  -1.00),
        ("HH prompt +$2.00",                ALL_HUBS,              0,    0,    None,   2.00),
        ("HH prompt -$1.50",                ALL_HUBS,              0,    0,    None,  -1.50),
        # basis blowout: one hub's price, Henry Hub unchanged
        ("Algonquin basis +$10 (winter)",   "Algonquin Citygates", 0,    None, WINTER, 10.00),
        ("TETCO M3 basis +$5 (winter)",     "TETCO M3",            0,    None, WINTER,  5.00),
        ("Chicago basis +$2 (winter)",      "Chicago Citygate",    0,    None, WINTER,  2.00),
        ("SoCal basis +$3 (winter)",        "SoCal Citygate",      0,    None, WINTER,  3.00),
        ("Waha basis -$2.00 (first year)",  "Waha",                0,    11,   None,  -2.00),
        # basis collapse: the winter premium disappears (Waha: the discount narrows)
        ("Algonquin basis -$5 (winter)",    "Algonquin Citygates", 0,    None, WINTER, -5.00),
        ("TETCO M3 basis -$2 (winter)",     "TETCO M3",            0,    None, WINTER, -2.00),
        ("Chicago basis -$0.50 (winter)",   "Chicago Citygate",    0,    None, WINTER, -0.50),
        ("SoCal basis -$1 (winter)",        "SoCal Citygate",      0,    None, WINTER, -1.00),
        ("Waha basis +$1.00 (first year)",  "Waha",                0,    11,   None,   1.00),
        # winter-summer spread widens
        ("Winter-summer spread +$1.00",     ALL_HUBS,              0,    None, WINTER,  0.75),
        ("Winter-summer spread +$1.00",     ALL_HUBS,              0,    None, SUMMER, -0.25),
    ],
    columns=_COLUMNS,
)
# fmt: on


def _open(value) -> bool:
    """True for "no limit": None, or the NaN pandas puts in a numeric column."""
    return value is None or (np.isscalar(value) and pd.isna(value))


def _validate(shocks: pd.DataFrame) -> None:
    missing = set(_COLUMNS) - set(shocks.columns)
    if missing:
        raise ValueError(f"shock table is missing column(s) {sorted(missing)}")
    for r in shocks.itertuples():
        if r.tenor_from < 0 or (not _open(r.tenor_to) and r.tenor_to < r.tenor_from):
            raise ValueError(f"bad tenor range in shock {r.scenario!r}")
        if not _open(r.months) and not set(r.months) <= set(range(1, 13)):
            raise ValueError(f"bad delivery months in shock {r.scenario!r}")


def shift_matrix(exposures: pd.DataFrame, asof, shocks: pd.DataFrame) -> pd.DataFrame:
    """Price shift ($/MMBtu) of each exposure row (columns) under each scenario (rows)."""
    _validate(shocks)
    tenor = month_index(exposures["contract"]) - (month_index(pd.Timestamp(asof)) + 1)
    moy = np.asarray(exposures["contract"].dt.month)
    hub = exposures["hub"].to_numpy()
    out = {}
    for r in shocks.itertuples():
        hit = tenor >= r.tenor_from
        if not _open(r.tenor_to):
            hit &= tenor <= r.tenor_to
        if not _open(r.months):
            hit &= np.isin(moy, r.months)
        if r.hub != ALL_HUBS:
            hit &= hub == r.hub
        prev = out.get(r.scenario, np.zeros(len(exposures)))
        out[r.scenario] = prev + np.where(hit, r.shift, 0.0)
    return pd.DataFrame(out, index=exposures.index).T


def shock_pnl(
    exposures: pd.DataFrame, asof, shocks: pd.DataFrame | None = None, by_desk: bool = True
) -> pd.DataFrame:
    """P&L of each scenario: one row per scenario, one column per desk (if `by_desk`)
    and a Total column. `exposures` has columns desk, hub, contract, mmbtu."""
    shocks = DEFAULT_SHOCKS if shocks is None else shocks
    shifts = shift_matrix(exposures, asof, shocks)
    legs = shifts * exposures["mmbtu"].to_numpy()  # scenario x exposure row
    out = pd.DataFrame(index=shifts.index)
    if by_desk:
        for desk, idx in exposures.groupby("desk").groups.items():
            out[desk] = legs[idx].sum(axis=1)
    out["Total"] = legs.sum(axis=1)
    out.index.name = "scenario"
    return out


def shock_by_hub(exposures: pd.DataFrame, asof, shocks: pd.DataFrame | None = None):
    """Total-book P&L of each scenario split by hub (scenario x hub)."""
    shocks = DEFAULT_SHOCKS if shocks is None else shocks
    legs = shift_matrix(exposures, asof, shocks) * exposures["mmbtu"].to_numpy()
    by_hub = legs.T.groupby(exposures["hub"].to_numpy()).sum().T
    cols = [HENRY] + sorted(c for c in by_hub.columns if c != HENRY)
    return by_hub.reindex(columns=[c for c in cols if c in by_hub.columns])

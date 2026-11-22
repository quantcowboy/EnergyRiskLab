"""Synthetic natural gas forward curves.

Every number here is made up. The model is simple, but it has the features that
matter for market risk:

* **Henry Hub** forward prices follow a two-factor model: a long-term level that
  reverts slowly to a long-run price, plus a short-term deviation that
  mean-reverts quickly. The
  short-term factor decays with time to delivery, so prompt contracts move more
  than back contracts (the Samuelson effect).
* **Seasonality:** winter contracts are priced above summer ones, and winter
  short-term volatility is higher.
* **Regional hubs** are Henry Hub plus a basis that has its own level, its own
  winter premium and its own mean-reverting shocks, correlated across hubs
  through one common factor.
* **A cold snap:** a few days of large prompt and winter-basis moves, so stress
  tests and backtests have something to find.

The output is a long table: one row per (price date, hub, contract month).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HENRY = "Henry Hub"


@dataclass(frozen=True)
class HubSpec:
    """Basis of a hub to Henry Hub, $/MMBtu."""

    level: float  # average basis
    winter_premium: float  # extra basis in Dec-Feb (half of it in Nov and Mar)
    vol: float  # annualised volatility of the basis shock, $/MMBtu
    spike: float = 0.0  # basis jump during the cold snap


DEFAULT_HUBS: dict[str, HubSpec] = {
    "Houston Ship Channel": HubSpec(-0.10, 0.05, 0.6),
    "Waha": HubSpec(-1.20, 0.30, 2.5),
    "Chicago Citygate": HubSpec(-0.25, 0.40, 1.0, spike=2.0),
    "Dominion South": HubSpec(-0.70, 0.30, 1.2),
    "TETCO M3": HubSpec(0.40, 2.00, 3.0, spike=6.0),
    "Algonquin Citygates": HubSpec(1.50, 5.00, 5.0, spike=12.0),
    "SoCal Citygate": HubSpec(0.60, 0.80, 2.0, spike=3.0),
}


@dataclass(frozen=True)
class MarketSpec:
    start: str = "2022-01-03"
    end: str = "2025-12-31"
    n_tenors: int = 36  # contract months priced each day, from the prompt month
    henry_start: float = 3.00  # $/MMBtu
    level_vol: float = 0.25  # long-term factor, annualised log vol
    level_kappa: float = 0.5  # slow reversion of the long-term factor, per year
    short_vol: float = 0.50  # short-term factor, annualised log vol (outside winter)
    short_kappa: float = 3.0  # mean reversion of the short-term factor, per year
    winter_vol_mult: float = 1.6  # short-term vol multiplier in Dec-Feb
    seasonal_amp: float = 0.12  # Jan contract ~12% above the average, Jul ~12% below
    basis_kappa: float = 4.0  # mean reversion of basis shocks, per year
    basis_corr: float = 0.5  # correlation of basis shocks across hubs
    snap_date: str | None = "2024-01-15"  # first day of the cold snap (None: no event)
    snap_days: int = 4
    snap_henry: float = 0.35  # log jump of the short-term factor, spread over the event
    hubs: dict[str, HubSpec] = field(default_factory=lambda: dict(DEFAULT_HUBS))


def _winter_weight(month: np.ndarray) -> np.ndarray:
    """1 in Dec-Feb, 0.5 in Nov and Mar, 0 otherwise."""
    w = np.zeros(month.shape)
    w[np.isin(month, (12, 1, 2))] = 1.0
    w[np.isin(month, (11, 3))] = 0.5
    return w


def month_index(d: pd.DatetimeIndex | pd.Series | pd.Timestamp) -> np.ndarray | int:
    """Months since year 0, so contract arithmetic is integer arithmetic."""
    if isinstance(d, pd.Timestamp):
        return d.year * 12 + d.month - 1
    d = pd.DatetimeIndex(d)
    return np.asarray(d.year * 12 + d.month - 1)


def _from_month_index(m: np.ndarray) -> pd.DatetimeIndex:
    m = np.asarray(m)
    return pd.to_datetime({"year": m // 12, "month": m % 12 + 1, "day": 1})


def generate_market(spec: MarketSpec | None = None, seed: int = 7) -> pd.DataFrame:
    """Daily forward curves for Henry Hub and the regional hubs.

    Returns columns: obs_date, hub, contract (first day of the delivery month),
    tenor (0 = prompt month, the month after the price date), price ($/MMBtu).
    The same seed always gives the same market.
    """
    spec = spec or MarketSpec()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(spec.start, spec.end)
    n, k = len(dates), spec.n_tenors
    dt = 1.0 / 252.0

    # --- time-to-delivery grid (n dates x k tenors)
    prompt = month_index(dates) + 1
    cmonth = prompt[:, None] + np.arange(k)[None, :]
    contract = _from_month_index(cmonth.ravel()).values.reshape(n, k)
    tau = (contract - dates.values[:, None]).astype("timedelta64[D]").astype(float) / 365.25
    moy = cmonth % 12 + 1  # month of year of each contract
    winter_c = _winter_weight(moy)

    # --- cold snap: a few days of extra shocks
    snap = np.zeros(n)
    if spec.snap_date is not None:
        first = int(np.searchsorted(dates.values, np.datetime64(spec.snap_date)))
        snap[first : first + spec.snap_days] = 1.0 / spec.snap_days

    # --- Henry Hub factors
    winter_t = _winter_weight(np.asarray(dates.month))
    short_sig = spec.short_vol * (1.0 + (spec.winter_vol_mult - 1.0) * winter_t)
    mu = np.log(spec.henry_start)
    lvl = np.full(n, mu)
    ldecay = np.exp(-spec.level_kappa * dt)
    leps = rng.standard_normal(n)
    for t in range(1, n):
        lvl[t] = mu + (lvl[t - 1] - mu) * ldecay + spec.level_vol * np.sqrt(dt) * leps[t]
    x = np.zeros(n)
    decay = np.exp(-spec.short_kappa * dt)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = x[t - 1] * decay + short_sig[t] * np.sqrt(dt) * eps[t] + spec.snap_henry * snap[t]
    seasonal = spec.seasonal_amp * np.cos(2 * np.pi * (moy - 1) / 12)
    henry = np.exp(lvl[:, None] + x[:, None] * np.exp(-spec.short_kappa * tau) + seasonal)

    frames = [_long(dates, contract, henry, HENRY)]

    # --- regional basis: level + winter premium + correlated mean-reverting shocks
    names = list(spec.hubs)
    common = rng.standard_normal(n)
    bdecay = np.exp(-spec.basis_kappa * dt)
    rho = spec.basis_corr
    for name in names:
        h = spec.hubs[name]
        z = rho * common + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] * bdecay + h.vol * np.sqrt(dt) * z[t] + h.spike * snap[t]
        # a shock is felt most at the prompt and in winter contracts
        reach = np.exp(-spec.basis_kappa * tau) * (0.3 + 0.7 * winter_c)
        basis = h.level + h.winter_premium * winter_c + y[:, None] * reach
        frames.append(_long(dates, contract, henry + basis, name))

    out = pd.concat(frames, ignore_index=True)
    out["tenor"] = np.tile(np.arange(k), n * (len(names) + 1))
    return out[["obs_date", "hub", "contract", "tenor", "price"]]


def _long(dates, contract, price, hub) -> pd.DataFrame:
    n, k = price.shape
    return pd.DataFrame(
        {
            "obs_date": np.repeat(dates.values, k),
            "hub": hub,
            "contract": contract.ravel(),
            "price": np.round(price.ravel(), 4),
        }
    )


def price_on(market: pd.DataFrame, date) -> pd.DataFrame:
    """The forward curves of one price date: hub, contract, price."""
    d = pd.Timestamp(date)
    return market.loc[market["obs_date"] == d, ["hub", "contract", "price"]]

"""Value at Risk and Expected Shortfall.

Sign convention: P&L is positive for a gain. VaR and ES are reported as
positive numbers, the size of the loss.

* Historical VaR at confidence `alpha` is the loss at the (1 - alpha) quantile
  of the scenario P&L.
* Historical ES is the average loss on the days at or beyond that quantile.
* Delta-normal VaR and ES assume normally distributed price changes with zero
  mean and the sample covariance of the scenarios.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var(pnl, alpha: float = 0.95) -> float:
    pnl = _clean(pnl)
    return float(-np.quantile(pnl, 1 - alpha))


def historical_es(pnl, alpha: float = 0.95) -> float:
    pnl = _clean(pnl)
    q = np.quantile(pnl, 1 - alpha)
    return float(-pnl[pnl <= q].mean())


def delta_normal(exposure: pd.Series, changes: pd.DataFrame, alpha: float = 0.95):
    """(VaR, ES, sigma) of a linear book under a zero-mean normal model."""
    cov = changes[exposure.index].dropna().cov().to_numpy()
    w = exposure.to_numpy()
    sigma = float(np.sqrt(w @ cov @ w))
    z = norm.ppf(alpha)
    return z * sigma, sigma * norm.pdf(z) / (1 - alpha), sigma


@dataclass
class RiskFigures:
    alpha: float
    horizon: int
    scenarios: int
    hist_var: float
    hist_es: float
    normal_var: float
    normal_es: float

    def as_dict(self) -> dict:
        return asdict(self)


def risk_figures(
    exposure: pd.Series, changes: pd.DataFrame, alpha: float = 0.95, horizon: int = 1
) -> RiskFigures:
    from .scenarios import scenario_pnl

    pnl = scenario_pnl(exposure, changes)
    nv, ne, _ = delta_normal(exposure, changes, alpha)
    return RiskFigures(
        alpha, horizon, len(pnl), historical_var(pnl, alpha), historical_es(pnl, alpha), nv, ne
    )


def _clean(pnl) -> np.ndarray:
    a = np.asarray(pnl, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("no P&L scenarios")
    return a

"""VaR backtesting.

A static book is run through history: each day's VaR is computed from the
`window` days before it, and compared with that day's actual (hypothetical)
P&L. A day where the loss is larger than VaR is an exception.

* Kupiec's proportion-of-failures test: are there about as many exceptions as
  the confidence level implies?
* Christoffersen's independence test: do exceptions cluster (an exception today
  making one tomorrow more likely)?

Both give a likelihood-ratio statistic, chi-squared with one degree of freedom
under the null hypothesis that the model is right; a small p-value rejects it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def rolling_var_backtest(pnl: pd.Series, window: int = 250, alpha: float = 0.99) -> pd.DataFrame:
    """Historical VaR from the previous `window` days vs each day's P&L."""
    var = -pnl.shift(1).rolling(window).quantile(1 - alpha)
    out = pd.DataFrame({"pnl": pnl, "var": var}).dropna()
    out["exception"] = out["pnl"] < -out["var"]
    return out


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(n: int, exceptions: int, alpha: float = 0.99) -> tuple[float, float]:
    """(LR statistic, p-value) for `exceptions` out of `n` days at confidence `alpha`."""
    if not 0 <= exceptions <= n or n == 0:
        raise ValueError("need 0 <= exceptions <= n and n > 0")
    p, phat = 1 - alpha, exceptions / n
    ll_null = _xlogy(n - exceptions, 1 - p) + _xlogy(exceptions, p)
    ll_alt = _xlogy(n - exceptions, 1 - phat) + _xlogy(exceptions, phat)
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_independence(exceptions) -> tuple[float, float]:
    """(LR statistic, p-value) of the test that exceptions are independent."""
    e = np.asarray(exceptions, dtype=bool)
    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum(~prev & ~cur))
    n01 = int(np.sum(~prev & cur))
    n10 = int(np.sum(prev & ~cur))
    n11 = int(np.sum(prev & cur))
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr = max(-2 * (ll_null - ll_alt), 0.0)
    return float(lr), float(chi2.sf(lr, 1))

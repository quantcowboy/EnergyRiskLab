import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from erl.risk.backtest import christoffersen_independence, kupiec_pof, rolling_var_backtest
from erl.risk.scenarios import exposure_by_tenor, scenario_pnl, tenor_changes
from erl.risk.var import delta_normal, historical_es, historical_var

RNG = np.random.default_rng(0)
NORMAL = RNG.standard_normal(400_000)


@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_historical_var_es_match_normal_theory(alpha):
    z = norm.ppf(alpha)
    assert historical_var(NORMAL, alpha) == pytest.approx(z, rel=0.01)
    assert historical_es(NORMAL, alpha) == pytest.approx(norm.pdf(z) / (1 - alpha), rel=0.01)


def test_es_at_least_var():
    pnl = RNG.standard_t(3, 5_000)
    assert historical_es(pnl, 0.95) >= historical_var(pnl, 0.95)


def test_delta_normal_two_assets():
    cov = np.array([[1.0, 0.5], [0.5, 4.0]])
    x = RNG.multivariate_normal([0, 0], cov, 200_000)
    changes = pd.DataFrame(x, columns=pd.MultiIndex.from_tuples([("A", 0), ("B", 0)]))
    w = pd.Series([2.0, -1.0], index=changes.columns)
    var, es, sigma = delta_normal(w, changes, 0.95)
    true_sigma = np.sqrt(w.to_numpy() @ cov @ w.to_numpy())
    assert sigma == pytest.approx(true_sigma, rel=0.01)
    assert var == pytest.approx(norm.ppf(0.95) * true_sigma, rel=0.01)
    assert es > var


def _toy_market():
    """One hub, three price dates, contracts Feb-Apr 2025; prices chosen by hand."""
    rows = []
    for d, px in [
        ("2025-01-02", [3.0, 2.8, 2.6]),
        ("2025-01-03", [3.1, 2.9, 2.6]),
        ("2025-01-06", [3.3, 2.8, 2.7]),
    ]:
        for c, p in zip(["2025-02-01", "2025-03-01", "2025-04-01"], px, strict=True):
            rows.append(
                {"obs_date": pd.Timestamp(d), "hub": "X", "contract": pd.Timestamp(c), "price": p}
            )
    return pd.DataFrame(rows)


def test_tenor_changes_use_the_same_contract():
    chg = tenor_changes(_toy_market(), horizon=1)
    assert list(chg.index) == list(pd.to_datetime(["2025-01-03", "2025-01-06"]))
    assert chg.loc["2025-01-03", ("X", 0)] == pytest.approx(0.1)  # Feb: 3.0 -> 3.1
    assert chg.loc["2025-01-06", ("X", 1)] == pytest.approx(-0.1)  # Mar: 2.9 -> 2.8
    two = tenor_changes(_toy_market(), horizon=2)
    assert two.loc["2025-01-06", ("X", 0)] == pytest.approx(0.3)  # Feb: 3.0 -> 3.3


def test_scenario_pnl_and_missing_history():
    chg = tenor_changes(_toy_market(), 1)
    expo = pd.DataFrame(
        {"desk": "D", "hub": "X", "contract": pd.to_datetime(["2025-02-01"]), "mmbtu": [100_000.0]}
    )
    pnl = scenario_pnl(exposure_by_tenor(expo, "2025-01-06"), chg)
    assert pnl.tolist() == pytest.approx([10_000.0, 20_000.0])
    far = expo.assign(contract=pd.to_datetime(["2026-12-01"]))
    with pytest.raises(ValueError, match="no price history"):
        scenario_pnl(exposure_by_tenor(far, "2025-01-06"), chg)


def test_kupiec():
    assert kupiec_pof(1000, 10, 0.99)[1] == pytest.approx(1.0)
    assert kupiec_pof(1000, 30, 0.99)[1] < 0.001
    assert kupiec_pof(1000, 0, 0.99)[1] < 0.001


def test_christoffersen_flags_clustering():
    spread = np.zeros(1000, bool)
    spread[::100] = True
    clustered = np.zeros(1000, bool)
    clustered[500:510] = True
    assert christoffersen_independence(spread)[1] > 0.5
    assert christoffersen_independence(clustered)[1] < 0.001


def test_rolling_backtest_on_normal_pnl():
    pnl = pd.Series(RNG.standard_normal(3_000), index=pd.bdate_range("2015-01-01", periods=3_000))
    bt = rolling_var_backtest(pnl, window=500, alpha=0.99)
    assert len(bt) == 2_500
    assert bt["exception"].mean() == pytest.approx(0.01, abs=0.005)

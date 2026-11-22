import numpy as np
import pandas as pd
import pytest

from erl.data.synthetic import HENRY, MarketSpec, generate_market, price_on
from erl.data.trades import generate_trades
from erl.positions import exposures, mark_to_market

SMALL = MarketSpec(start="2024-01-02", end="2024-12-31", n_tenors=18)


@pytest.fixture(scope="module")
def market():
    return generate_market(SMALL, seed=1)


def test_same_seed_same_market(market):
    pd.testing.assert_frame_equal(market, generate_market(SMALL, seed=1))
    assert not market.equals(generate_market(SMALL, seed=2))


def test_shape_and_values(market):
    n_days = len(pd.bdate_range(SMALL.start, SMALL.end))
    assert len(market) == n_days * SMALL.n_tenors * (1 + len(SMALL.hubs))
    assert np.isfinite(market["price"]).all()
    assert (market.loc[market["hub"] == HENRY, "price"] > 0).all()


def test_winter_above_summer(market):
    h = market[market["hub"] == HENRY]
    m = h["contract"].dt.month
    assert h.loc[m == 1, "price"].mean() > h.loc[m == 7, "price"].mean()


def test_prompt_moves_more_than_back(market):
    h = market[market["hub"] == HENRY].pivot(index="obs_date", columns="tenor", values="price")
    vol = np.log(h).diff().std()
    assert vol[0] > vol[12]


def _trade(kind, hub, volume, fixed, start="2025-03-01", end="2025-04-01"):
    return pd.DataFrame(
        [
            {
                "trade_id": "T1",
                "desk": "Desk A",
                "trade_type": kind,
                "hub": hub,
                "trade_date": pd.Timestamp("2025-01-10"),
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(end),
                "volume_per_day": volume,
                "fixed_price": fixed,
            }
        ]
    )


def test_fixed_price_exposure_is_volume_times_days():
    e = exposures(_trade("fixed_price", "Waha", 10_000, 1.0), "2025-01-15")
    assert e["mmbtu"].tolist() == [310_000, 300_000]  # March 31 days, April 30
    assert set(e["hub"]) == {"Waha"}


def test_basis_swap_is_long_hub_short_henry():
    e = exposures(_trade("basis", "Waha", 10_000, -1.0), "2025-01-15")
    by_hub = e.groupby("hub")["mmbtu"].sum()
    assert by_hub["Waha"] == 610_000
    assert by_hub[HENRY] == -610_000


def test_delivered_months_drop_out():
    assert (
        exposures(_trade("fixed_price", "Waha", 10_000, 1.0), "2025-03-20")["mmbtu"].sum()
        == 300_000
    )


def test_mark_to_market_sign():
    curves = pd.DataFrame(
        {
            "hub": ["Waha", "Waha", HENRY, HENRY],
            "contract": pd.to_datetime(["2025-03-01", "2025-04-01"] * 2),
            "price": [2.0, 2.0, 3.0, 3.0],
        }
    )
    long_ = mark_to_market(_trade("fixed_price", "Waha", 10_000, 1.5), curves, "2025-01-15")
    assert long_["value"].iloc[0] == pytest.approx(0.5 * 610_000)
    basis = mark_to_market(_trade("basis", "Waha", -10_000, -0.5), curves, "2025-01-15")
    assert basis["value"].iloc[0] == pytest.approx(-0.5 * -610_000)  # basis -1.0 vs fixed -0.5


def test_trades_at_the_money_on_trade_date(market):
    asof = market["obs_date"].max()
    t = generate_trades(market, asof, n_trades=60, seed=5)
    assert len(t) > 0 and set(t["trade_type"]) <= {"fixed_price", "basis"}
    for _, tr in t.head(10).iterrows():  # revalue on its own trade date: close to zero
        one = t[t["trade_id"] == tr["trade_id"]]
        v = mark_to_market(one, price_on(market, tr["trade_date"]), tr["trade_date"])
        gross = abs(tr["volume_per_day"]) * 31 * ((tr["end"] - tr["start"]).days / 30 + 1)
        assert abs(v["value"].sum()) < 0.05 * gross

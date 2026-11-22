import numpy as np
import pandas as pd
import pytest

from erl.data.synthetic import HENRY
from erl.risk.shocks import ALL_HUBS, DEFAULT_SHOCKS, shock_by_hub, shock_pnl
from erl.risk.stress import (
    book_in_month,
    month_windows,
    net_by_contract,
    price_panel,
    stress_by_month,
    summarise,
)

# --- historical stress ----------------------------------------------------------------
#
# Hand-built market, 1 Dec 2023 to 29 Feb 2024, two curves:
#   X  Mar-2024: 3.00 + 0.01 a business day, plus a +1.00 jump from 17 Jan 2024
#   HH Feb-2024: 2.00 flat, plus a +1.00 jump from 17 Jan 2024
# Book on 10 Jan 2025: long 100,000 MMBtu X Mar-2025, short 200,000 Henry Feb-2025.
# Stress month Jan 2025, shifted back 12 months: X Mar-2024 and Henry Feb-2024 over the
# 5-day windows ending in Jan 2024 (23 business days). A normal window makes
# 100,000 x 0.05 = +5,000. The 5 windows ending 17-23 Jan hold the jump:
# long 100,000 x 1.05 = +105,000, short -200,000 x 1.00 = -200,000, net -95,000.

JUMP = pd.Timestamp("2024-01-17")


def _toy_market():
    dates = pd.bdate_range("2023-12-01", "2024-02-29")
    i = np.arange(len(dates))
    jump = (dates >= JUMP).astype(float)
    x = pd.DataFrame(
        {"obs_date": dates, "hub": "X", "contract": pd.Timestamp("2024-03-01")}
    ).assign(price=3.0 + 0.01 * i + jump)
    henry = pd.DataFrame(
        {"obs_date": dates, "hub": HENRY, "contract": pd.Timestamp("2024-02-01")}
    ).assign(price=2.0 + jump)
    henry = henry[henry["obs_date"] < "2024-02-01"]  # Feb-2024 stops trading at month end
    return pd.concat([x, henry], ignore_index=True)


def _toy_book():
    return pd.DataFrame(
        {
            "desk": ["D1", "D2"],
            "hub": ["X", HENRY],
            "contract": pd.to_datetime(["2025-03-01", "2025-02-01"]),
            "mmbtu": [100_000.0, -200_000.0],
        }
    )


def test_month_windows_hand_built():
    panel = price_panel(_toy_market())
    w = month_windows(net_by_contract(_toy_book()), panel, "2025-01-01", horizon=5)
    assert len(w) == 23 and set(w["year"]) == {2024} and set(w["shift"]) == {12}
    assert (w["end"].dt.month == 1).all()
    jumped = w["end"].between(JUMP, "2024-01-23")
    assert jumped.sum() == 5
    assert w.loc[~jumped, "pnl"].to_numpy() == pytest.approx(5_000.0)
    assert w.loc[jumped, "pnl"].to_numpy() == pytest.approx(-95_000.0)
    assert w.loc[jumped, "long"].to_numpy() == pytest.approx(105_000.0)
    assert w.loc[jumped, "short"].to_numpy() == pytest.approx(-200_000.0)
    assert (w["pnl"] == w["long"] + w["short"]).all()
    assert w.loc[jumped, HENRY].to_numpy() == pytest.approx(-200_000.0)


def test_summary_worst_window_legs_p5_median():
    panel = price_panel(_toy_market())
    s = summarise(month_windows(net_by_contract(_toy_book()), panel, "2025-01-01"))
    assert s["windows"] == 23 and s["years"] == 1
    assert s["worst"] == pytest.approx(-95_000.0)
    # the legs come from the worst net window, not each leg's own worst
    assert s["worst_long"] == pytest.approx(105_000.0)
    assert s["worst_short"] == pytest.approx(-200_000.0)
    # five windows tie at -95,000; the worst is one of them, 5 business days long
    assert JUMP <= s["worst_end"] <= pd.Timestamp("2024-01-23")
    assert s["worst_start"] == s["worst_end"] - pd.offsets.BDay(5)
    assert s["main_hubs"] == "Henry Hub -200k"
    # 23 sorted values: 5 x -95,000 then 18 x +5,000; the 5% point is at 1.1 -> -95,000
    assert s["p5"] == pytest.approx(-95_000.0)
    assert s["median"] == pytest.approx(5_000.0)


def test_forward_month_rolls_off_delivered_positions():
    book = book_in_month(_toy_book(), "2025-02-01")  # Feb-2025 is delivering: gone
    assert list(book["hub"]) == ["X"]
    panel = price_panel(_toy_market())
    w = month_windows(net_by_contract(book), panel, "2025-02-01")
    assert len(w) == 21  # business days in Feb 2024
    assert w["pnl"].to_numpy() == pytest.approx(5_000.0)  # X Mar-2024 drifts +0.05


def test_stress_by_month_uses_prior_years_only_and_history_to_asof():
    market = _toy_market()
    # as of 10 Jan 2024 the Jan 2024 windows are not "prior years": no windows at all
    year_earlier = _toy_book().assign(contract=lambda d: d["contract"] - pd.DateOffset(years=1))
    s = stress_by_month(year_earlier, market, "2024-01-10", months_ahead=0, by_desk=False)
    assert s["windows"].tolist() == [0]
    # as of 10 Jan 2025, by desk and in total, Jan and Feb 2025
    s = stress_by_month(_toy_book(), market, "2025-01-10", months_ahead=1).set_index(
        ["book", "month"]
    )
    jan, feb = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")
    assert s.loc[("Total", jan), "worst"] == pytest.approx(-95_000.0)
    assert s.loc[("D2", jan), "worst"] == pytest.approx(-200_000.0)
    assert s.loc[("D1", jan), "median"] == pytest.approx(5_000.0)
    assert s.loc[("D2", feb), "worst"] == 0.0  # nothing left to deliver after Feb
    assert s.loc[("Total", feb), "worst"] == pytest.approx(5_000.0)


def test_missing_shifted_history_raises():
    panel = price_panel(_toy_market())
    far = net_by_contract(_toy_book().assign(contract=pd.to_datetime(["2025-06-01"] * 2)))
    with pytest.raises(ValueError, match="no price history"):
        month_windows(far, panel, "2025-01-01")


# --- hypothetical shocks ----------------------------------------------------------------
#
# Book on 14 Nov 2025 (prompt = Dec 2025, tenor 0):
#   D1  Henry Dec-25  +310,000      D1  Waha Dec-25      +310,000
#   D1  Henry Dec-25  -310,000 (the Henry leg of a Waha basis swap, so Henry nets to 0)
#   D2  Henry Jul-26  -100,000      D2  Algonquin Jan-26 +50,000


def _shock_book():
    return pd.DataFrame(
        {
            "desk": ["D1", "D1", "D2", "D2"],
            "hub": [HENRY, "Waha", HENRY, "Algonquin Citygates"],
            "contract": pd.to_datetime(["2025-12-01", "2025-12-01", "2026-07-01", "2026-01-01"]),
            "mmbtu": [0.0, 310_000.0, -100_000.0, 50_000.0],
        }
    )


def test_default_shocks_hand_built():
    p = shock_pnl(_shock_book(), "2025-11-14")
    # +$1 on every hub: net 310,000 - 100,000 + 50,000 = 260,000
    assert p.loc["HH parallel +$1.00"].tolist() == pytest.approx([310_000, -50_000, 260_000])
    # prompt only (Dec-25): Waha 310,000 x 2
    assert p.loc["HH prompt +$2.00", "Total"] == pytest.approx(620_000)
    # Algonquin basis, winter only: Jan-26 50,000 x 10
    assert p.loc["Algonquin basis +$10 (winter)", "Total"] == pytest.approx(500_000)
    # Waha basis within 12 months: 310,000 x -2
    assert p.loc["Waha basis -$2.00 (first year)", "D1"] == pytest.approx(-620_000)
    # winter +0.75 on Dec-25 and Jan-26, summer -0.25 on Jul-26
    assert p.loc["Winter-summer spread +$1.00", "Total"] == pytest.approx(
        0.75 * (310_000 + 50_000) - 0.25 * -100_000
    )
    assert len(p) == DEFAULT_SHOCKS["scenario"].nunique()


def test_basis_swap_immune_to_henry_parallel_but_not_basis():
    swap = pd.DataFrame(
        {
            "desk": "D",
            "hub": ["TETCO M3", HENRY],
            "contract": pd.Timestamp("2026-01-01"),
            "mmbtu": [310_000.0, -310_000.0],
        }
    )
    p = shock_pnl(swap, "2025-11-14")
    assert p.loc["HH parallel -$1.00", "Total"] == pytest.approx(0.0)
    assert p.loc["TETCO M3 basis +$5 (winter)", "Total"] == pytest.approx(1_550_000)
    h = shock_by_hub(swap, "2025-11-14")
    assert h.loc["HH parallel -$1.00", HENRY] == pytest.approx(310_000)
    assert h.loc["HH parallel -$1.00", "TETCO M3"] == pytest.approx(-310_000)


def test_custom_shock_table_and_validation():
    table = pd.DataFrame(
        [("Back +$0.50", ALL_HUBS, 1, None, None, 0.5)],
        columns=["scenario", "hub", "tenor_from", "tenor_to", "months", "shift"],
    )
    p = shock_pnl(_shock_book(), "2025-11-14", table, by_desk=False)
    assert p.loc["Back +$0.50", "Total"] == pytest.approx(0.5 * (-100_000 + 50_000))
    bad = table.assign(tenor_from=3, tenor_to=1)
    with pytest.raises(ValueError, match="tenor range"):
        shock_pnl(_shock_book(), "2025-11-14", bad)
    with pytest.raises(ValueError, match="missing column"):
        shock_pnl(_shock_book(), "2025-11-14", table.drop(columns="months"))

"""End-to-end demo: synthetic market -> trades -> exposure -> VaR / ES -> backtest -> stress.

python scripts/run_demo.py            (prints the results, writes outputs/)
python scripts/run_demo.py --seed 3   (a different synthetic market)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from erl.data.synthetic import generate_market
from erl.data.trades import generate_trades
from erl.positions import exposures, mark_to_market
from erl.risk.backtest import christoffersen_independence, kupiec_pof, rolling_var_backtest
from erl.risk.scenarios import exposure_by_tenor, scenario_pnl, tenor_changes, window
from erl.risk.shocks import shock_by_hub, shock_pnl
from erl.risk.stress import stress_by_month
from erl.risk.var import risk_figures

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--asof", default="2025-11-14")
    ap.add_argument("--window", type=int, default=500, help="scenario days for VaR")
    args = ap.parse_args()

    market = generate_market(seed=args.seed)
    asof = pd.Timestamp(args.asof)
    asof = market.loc[market["obs_date"] <= asof, "obs_date"].max()  # last business day
    trades = generate_trades(market, asof, seed=args.seed + 4)
    expo = exposures(trades, asof)
    mtm = mark_to_market(trades, market[market["obs_date"] == asof], asof)

    print(
        f"As of {asof:%Y-%m-%d}: {len(trades)} live trades, "
        f"{expo['hub'].nunique()} hubs, {expo['contract'].nunique()} delivery months"
    )

    rows = []
    for horizon in (1, 5):
        chg = window(tenor_changes(market, horizon), asof, args.window)
        books = {d: g for d, g in expo.groupby("desk")} | {"Total": expo}
        for desk, e in books.items():
            f = risk_figures(exposure_by_tenor(e, asof), chg, alpha=0.95, horizon=horizon)
            rows.append(
                {
                    "book": desk,
                    **f.as_dict(),
                    "mtm": mtm.loc[mtm["desk"] == desk, "value"].sum()
                    if desk != "Total"
                    else mtm["value"].sum(),
                }
            )
    table = pd.DataFrame(rows)
    show = table.assign(
        **{
            c: (table[c] / 1e3).round(0)
            for c in ("hist_var", "hist_es", "normal_var", "normal_es", "mtm")
        }
    )
    print(
        "\n95% VaR and Expected Shortfall, $ thousands "
        f"({args.window} historical scenarios, tenor-matched)"
    )
    print(
        show[
            ["book", "horizon", "hist_var", "hist_es", "normal_var", "normal_es", "mtm"]
        ].to_string(index=False)
    )

    # backtest: today's book run through all of history, 99% one-day VaR
    pnl = scenario_pnl(exposure_by_tenor(expo, asof), tenor_changes(market, 1))
    bt = rolling_var_backtest(pnl, window=250, alpha=0.99)
    n, x = len(bt), int(bt["exception"].sum())
    lr_k, p_k = kupiec_pof(n, x, 0.99)
    lr_c, p_c = christoffersen_independence(bt["exception"])
    print(
        f"\nBacktest, 99% one-day VaR, 250-day window: {x} exceptions in {n} days "
        f"(expected {0.01 * n:.1f})"
    )
    print(f"  Kupiec p-value {p_k:.3f}   Christoffersen independence p-value {p_c:.3f}")

    # historical stress: worst 5-day window of each calendar month in prior years
    stress = stress_by_month(expo, market, asof, months_ahead=12, horizon=5)
    total = stress[stress["book"] == "Total"]
    k = {
        c: (total[c] / 1e3).round(0) for c in ("worst", "worst_long", "worst_short", "p5", "median")
    }
    show = total.assign(
        **k,
        month=total["month"].dt.strftime("%b %Y"),
        window=total["worst_start"].dt.strftime("%d %b %Y")
        + " - "
        + total["worst_end"].dt.strftime("%d %b"),
    )
    print(
        "\nHistorical stress, total book, $ thousands: every 5-day window of the same "
        "calendar month in prior years"
    )
    print(
        show[
            ["month", "windows", "worst", "worst_long", "worst_short", "window", "p5", "median"]
        ].to_string(index=False)
    )
    print("Main hubs in the worst window:")
    for _, r in show.iterrows():
        print(f"  {r['month']}: {r['main_hubs']}")

    # hypothetical shocks
    shocks = shock_pnl(expo, asof)
    print("\nHypothetical shocks, $ thousands")
    print((shocks / 1e3).round(0).to_string())

    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / f"risk_{asof:%Y%m%d}.csv", index=False)
    stress.to_csv(OUT / f"stress_by_month_{asof:%Y%m%d}.csv", index=False)
    shocks.to_csv(OUT / f"shocks_{asof:%Y%m%d}.csv")
    shock_by_hub(expo, asof).to_csv(OUT / f"shocks_by_hub_{asof:%Y%m%d}.csv")
    expo.to_csv(OUT / f"exposure_{asof:%Y%m%d}.csv", index=False)
    bt.to_csv(OUT / f"backtest_{asof:%Y%m%d}.csv")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()

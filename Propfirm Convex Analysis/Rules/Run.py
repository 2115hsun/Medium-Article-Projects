"""
run.py — orchestrator. Pick a firm + strategy, run all three analyses.

Usage:
    python run.py                      # defaults: apex_intraday + BOS example
    python run.py topstep              # a specific firm
    python run.py mffu_rapid           # etc.

Firm keys: apex_eod, apex_intraday, lucid, tradeify_select,
           tradeify_growth, mffu_rapid, topstep
"""

import sys, os, importlib.util
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from firms import get_firm, FIRMS                     # noqa: E402
from strategy import SyntheticStrategy                # noqa: E402
from analysis import (plot_equity_paths,              # noqa: E402
                      plot_strategy_comparison,
                      plot_risk_rr_heatmap)

OUT = os.environ.get("OUT_DIR", "/mnt/user-data/outputs")
os.makedirs(OUT, exist_ok=True)


def load_strategy(path: str):
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_strategy()


def main(firm_key="apex_intraday", strategy_path=None, risk_pct=1.0, trades_per_day=3):
    firm = get_firm(firm_key)
    print(f"\n=== {firm.name} | ${firm.start_balance:,.0f} account ===")
    print(f"target ${firm.target_equity:,.0f} | floor ${firm.initial_floor:,.0f} "
          f"| dd={firm.dd_type.value} | consistency={firm.consistency_pct} "
          f"| min_days={firm.min_days}")

    risk_dollars = firm.start_balance * risk_pct / 100.0

    # ---- 1) equity paths across risk geometries at fixed EV=0 ----
    ev0_geoms = [(rr, 1.0 / (rr + 1.0)) for rr in (4, 3, 2, 1, 0.5, 0.33)]
    plot_equity_paths(
        firm, ev0_geoms, risk_per_trade=risk_dollars, trades_per_day=trades_per_day,
        savepath=f"{OUT}/1_equity_paths_{firm.key}.png")
    print("  -> equity paths saved")

    # ---- 2) strategy comparison (same EV=0 geometries here; swap in real strats) ----
    strat_set = [SyntheticStrategy.from_edge(rr, edge_R=0.0,
                 name=f"RR {rr:g} (EV=0)") for rr in (0.33, 0.5, 1, 2, 3, 4)]
    plot_strategy_comparison(
        firm, strat_set, risk_per_trade=risk_dollars, trades_per_day=trades_per_day,
        savepath=f"{OUT}/2_comparison_{firm.key}.png")
    print("  -> comparison saved")

    # ---- 3) risk% x R:R heatmap at a fixed positive edge ----
    risk_pcts = np.round(np.arange(0.25, 3.75 + 0.01, 0.25), 2)
    rr_values = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    plot_risk_rr_heatmap(
        firm, risk_pcts, rr_values, edge_R=0.10, trades_per_day=trades_per_day,
        savepath=f"{OUT}/3_heatmap_{firm.key}.png")
    print("  -> heatmap saved")


if __name__ == "__main__":
    fk = sys.argv[1] if len(sys.argv) > 1 else "apex_intraday"
    main(fk)
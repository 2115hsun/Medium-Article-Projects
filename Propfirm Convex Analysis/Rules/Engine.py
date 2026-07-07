"""
engine.py — vectorized Monte Carlo of prop-firm challenge paths.

Simulates N accounts in parallel, one trade "step" at a time (vectorized over
accounts with numpy, so it's fast). Trades are grouped into DAYS via
`trades_per_day` because the day boundary is what activates EOD-trailing
drawdown updates, daily loss limits, consistency, and min-days rules.

Outcome codes per path: +1 PASS, -1 FAIL, 0 ONGOING (hit max_trades unresolved).

IMPORTANT MODELLING NOTES
-------------------------
* We only have each trade's NET R-multiple, so intraday breaches are checked on
  equity *after each trade closes*, not on intra-trade excursion (MAE). That
  slightly UNDER-states intraday-trailing breaches. If your strategy has big
  adverse excursions, add an MAE haircut to `returns_R`.
* Consistency here is modelled as: reaching target with a violated consistency
  ratio does NOT pass -> the account keeps trading to dilute the big day. That
  matches MFFU/Tradeify/Topstep intent (trade more days), not "instant fail".
* trades_per_day is a real assumption. It changes EOD/DLL/consistency behaviour.
  Set it to your strategy's realistic trades/day (BOS intraday ~ a few).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from Firms import FirmConfig, DrawdownType, DLLBehavior
from Strategy import Strategy


@dataclass
class SimResult:
    outcome: np.ndarray        # [N] in {+1, -1, 0}
    trades_to_pass: np.ndarray # [N] int, valid where outcome==+1 (else -1)
    days_to_pass: np.ndarray   # [N] int, valid where outcome==+1 (else -1)
    equity_paths: np.ndarray | None  # [N, T] if record_paths else None

    @property
    def pass_rate(self) -> float:
        return float(np.mean(self.outcome == 1))

    @property
    def fail_rate(self) -> float:
        return float(np.mean(self.outcome == -1))

    @property
    def ongoing_rate(self) -> float:
        return float(np.mean(self.outcome == 0))

    @property
    def mean_trades_to_pass(self) -> float:
        w = self.outcome == 1
        return float(self.trades_to_pass[w].mean()) if w.any() else float("nan")

    @property
    def median_trades_to_pass(self) -> float:
        w = self.outcome == 1
        return float(np.median(self.trades_to_pass[w])) if w.any() else float("nan")


def simulate(
    Strategy: Strategy,
    firm: FirmConfig,
    risk_per_trade: float,          # $ risked per trade (a full -1R loss = -risk_per_trade)
    n_paths: int = 5_000,
    max_trades: int = 400,
    trades_per_day: int = 3,
    seed: int | None = 0,
    record_paths: bool = False,
) -> SimResult:
    rng = np.random.default_rng(seed)
    N, T = n_paths, max_trades
    R = Strategy.sample(N, T, rng)          # [N, T] R-multiples
    pnl_all = R * risk_per_trade            # [N, T] dollar pnl per trade

    start = firm.start_balance
    target_eq = firm.target_equity
    dd = firm.max_drawdown
    lock = firm.trail_lock                  # None or a ceiling on the dd level

    # State vectors
    equity = np.full(N, start)
    peak = np.full(N, start)                # intraday peak (for INTRADAY_TRAIL)
    eod_peak = np.full(N, start)            # highest end-of-day balance (for EOD_TRAIL)
    dd_level = np.full(N, start - dd)       # current breach floor
    day_pnl = np.zeros(N)
    best_day = np.full(N, -np.inf)          # best COMPLETED-day pnl so far
    trades_in_day = np.zeros(N, dtype=int)
    day_index = np.zeros(N, dtype=int)
    status = np.zeros(N, dtype=int)         # 0 ongoing, +1 pass, -1 fail
    ttp = np.full(N, -1, dtype=int)         # trades to pass
    dtp = np.full(N, -1, dtype=int)         # days to pass

    paths = np.empty((N, T), dtype=float) if record_paths else None

    is_intraday = firm.dd_type == DrawdownType.INTRADAY_TRAIL
    is_eod_trail = firm.dd_type == DrawdownType.EOD_TRAIL
    has_dll = firm.daily_loss_limit is not None and firm.dll_behavior != DLLBehavior.NONE

    for t in range(T):
        live = status == 0
        # ---- apply the trade ----
        equity[live] += pnl_all[live, t]
        day_pnl[live] += pnl_all[live, t]
        trades_in_day[live] += 1
        if record_paths:
            paths[:, t] = equity

        # ---- intraday trailing DD update + intraday/static breach check ----
        if is_intraday:
            np.maximum(peak, equity, out=peak, where=live)
            new_level = peak - dd
            if lock is not None:
                new_level = np.minimum(new_level, lock)
            np.maximum(dd_level, new_level, out=dd_level, where=live)

        if firm.dd_type in (DrawdownType.STATIC, DrawdownType.INTRADAY_TRAIL):
            breach = live & (equity <= dd_level + 1e-9)
            status[breach] = -1

        # ---- daily loss limit ----
        forced_day_end = np.zeros(N, dtype=bool)
        if has_dll:
            hit = (status == 0) & (day_pnl <= -firm.daily_loss_limit + 1e-9)
            if firm.dll_behavior == DLLBehavior.HARD_FAIL:
                status[hit] = -1
            else:  # SOFT_LOCK -> end the day, no fail
                forced_day_end |= hit

        # ---- day boundary (natural or forced) ----
        day_end = (status == 0) & ((trades_in_day >= trades_per_day) | forced_day_end)
        if day_end.any():
            # record the just-finished day into best_day
            np.maximum(best_day, day_pnl, out=best_day, where=day_end)

            if is_eod_trail:
                np.maximum(eod_peak, equity, out=eod_peak, where=day_end)
                new_level = eod_peak - dd
                if lock is not None:
                    new_level = np.minimum(new_level, lock)
                np.maximum(dd_level, new_level, out=dd_level, where=day_end)

            # EOD breach (for EOD-trailing and static, checked at close)
            if firm.dd_type in (DrawdownType.EOD_TRAIL, DrawdownType.STATIC):
                eod_breach = day_end & (equity <= dd_level + 1e-9)
                status[eod_breach] = -1
                day_end = day_end & (status == 0)

            day_index[day_end] += 1

            # ---- pass check at end of day ----
            total_profit = equity - start
            cons_ok = np.ones(N, dtype=bool)
            if firm.consistency_pct is not None:
                # best day (incl. the one just closed) <= pct * total profit
                cons_ok = best_day <= firm.consistency_pct * total_profit + 1e-9
            passed = (
                day_end
                & (equity >= target_eq - 1e-9)
                & (day_index >= firm.min_days)
                & cons_ok
            )
            status[passed] = 1
            ttp[passed] = t + 1
            dtp[passed] = day_index[passed]

            # reset the day for accounts whose day ended and are still live
            reset = day_end & (status == 0) | (day_end & forced_day_end & (status == 1))
            reset = day_end  # everything that ended a day resets its day counters
            trades_in_day[reset] = 0
            day_pnl[reset] = 0.0

        if (status != 0).all():
            if record_paths:
                paths[:, t + 1:] = equity[:, None]
            break

    return SimResult(outcome=status, trades_to_pass=ttp, days_to_pass=dtp,
                     equity_paths=paths)
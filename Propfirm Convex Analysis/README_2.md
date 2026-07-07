# Prop Firm Challenge Simulator

Monte Carlo framework that models a $50K prop-firm evaluation as a **first-passage
problem**: reach $53,000 (profit target) before the drawdown barrier ends you.
The whole point is not P&L — it's maximizing **P(pass)**, then choosing the risk
sizing and R:R geometry that games each firm's specific barrier structure.

## Files

| file | what it does |
|---|---|
| `firms.py` | `$50K` rule configs for all 5 firms (7 eval paths). Edit here. |
| `strategy.py` | `SyntheticStrategy` (WR/RR) and `EmpiricalStrategy` (bootstrap real backtest R-multiples). |
| `engine.py` | Vectorized MC engine. Trailing/static DD, DLL, consistency, min-days. |
| `analysis.py` | The 3 deliverables: equity paths, comparison summary, risk×RR heatmap. |
| `strategies/example_bos.py` | Template for plugging in a real strategy. |
| `run.py` | Orchestrator — runs all 3 analyses for a firm. |

## Quick start

```bash
python run.py apex_intraday      # firm keys below
```

Firm keys: `apex_eod`, `apex_intraday`, `lucid`, `tradeify_select`,
`tradeify_growth`, `mffu_rapid`, `topstep`.

## Plugging in YOUR strategy (BOS / SuperTrend / EMA crossover …)

The engine only needs per-trade outcomes in **R-multiples** (a +2:1 winner = +2.0R,
a full stop-out = -1.0R). From any backtest:

```python
R = trade_pnl / risk_dollars_on_that_trade   # array of R-multiples
strat = EmpiricalStrategy(R, name="BOS 5m")   # bootstrap-resamples it
```

Bootstrapping preserves your real win rate, R:R spread, partial exits, and fat
tails. Use `block=N` for a block bootstrap if your trades are autocorrelated.
See `strategies/example_bos.py`.

## The 3 analyses

1. **Equity paths** (`plot_equity_paths`) — green=pass, red=fail, grey=ongoing,
   with the firm's actual target/floor lines.
2. **Strategy comparison** (`plot_strategy_comparison`) — pass rate, outcome
   distribution, and speed-to-target across strategies or R:R geometries.
3. **Risk × R:R heatmap** (`plot_risk_rr_heatmap`) — holds a fixed edge
   (`edge_R`, e.g. +0.10R) and sweeps risk-per-trade × R:R. Each cell = pass rate
   (colour) + avg trades to pass. This is your optimizer: it shows the sizing
   that maximizes pass probability under each firm's barrier.

---

## ⚠️ Modelling assumptions that materially change results — VERIFY THESE

These are the levers that dominate pass rates. They're marked `# VERIFY` in
`firms.py`. My defaults are best-reads of your rule docs, not gospel.

1. **Trailing vs static drawdown.** Most of these firms use a *trailing* barrier
   that ratchets up as you profit. A trailing DD is much harder to survive than a
   static $48k floor — your earlier static-floor model overstates pass rates.

2. **`trail_lock` (does the trailing DD stop at breakeven?).** This is arguably
   the single biggest driver. Example from the cross-firm run: Apex-Intraday
   (locks at $50k → 54% pass) vs MFFU-Rapid (never locks → 35% pass) on the
   *same* strategy. Confirm each firm's lock point.

3. **Daily loss limit behaviour.** Apex-EOD's hard $1,000 DLL crushes its pass
   rate (~30% vs Apex-Intraday's ~54% on the same strategy) because two $500
   losers in a day = instant fail. Confirm hard-fail vs soft-lock per firm.

4. **`trades_per_day`.** Day grouping drives EOD trailing, DLL, and consistency.
   Set it to your strategy's realistic trades/day.

5. **Consistency & min-days** are modelled as "keep trading to dilute the big
   day / meet the day count", not instant fail. This slows passing (see Tradeify
   Select) but rarely fails outright.

6. **Intraday breaches use trade-close equity**, not intra-trade excursion (MAE).
   If your strategy has large adverse excursions, apply an MAE haircut to your R
   values, or the model slightly under-states intraday-trailing failures.

7. This models the **challenge/eval stage only.** The funded-stage payout games
   (consistency for withdrawals, safety-net buffers, scaling) are a separate
   layer — easy to add as a second simulation phase once the eval model is solid.

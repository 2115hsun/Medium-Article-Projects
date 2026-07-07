"""
analysis.py — the three deliverables, all firm-aware.

  1. plot_equity_paths(firm, geometries)     -> image-1 style grid
  2. plot_strategy_comparison(firm, strats)  -> image-2 style summary bars
  3. plot_risk_rr_heatmap(firm, ...)         -> image-3 style heatmap

Everything takes a FirmConfig so barriers, consistency, and min-days come from
the firm, not hard-coded numbers.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from Firms import FirmConfig
from Strategy import Strategy, SyntheticStrategy
from Engine import simulate

# red -> yellow -> green colormap for the heatmap
_HEAT = LinearSegmentedColormap.from_list(
    "rag", ["#8a1f1f", "#c0392b", "#e67e22", "#c9b02a", "#4a9a3a", "#1e7a2e"]
)


# ---------------------------------------------------------------------------
# 1) EQUITY PATHS
# ---------------------------------------------------------------------------
def plot_equity_paths(
    firm: FirmConfig,
    geometries: list[tuple[float, float]],   # list of (rr, win_rate)
    risk_per_trade: float,
    n_show: int = 120,
    max_trades: int = 300,
    trades_per_day: int = 3,
    seed: int = 0,
    savepath: str | None = None,
):
    ncols = 3
    nrows = int(np.ceil(len(geometries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)

    for i, (rr, wr) in enumerate(geometries):
        ax = axes[i // ncols][i % ncols]
        strat = SyntheticStrategy(win_rate=wr, rr=rr)
        res = simulate(strat, firm, risk_per_trade, n_paths=n_show,
                       max_trades=max_trades, trades_per_day=trades_per_day,
                       seed=seed, record_paths=True)
        for p in range(n_show):
            color = {1: "#2e8b57", -1: "#c0392b", 0: "#999999"}[res.outcome[p]]
            ax.plot(res.equity_paths[p], color=color, alpha=0.35, lw=0.8)
        ax.axhline(firm.target_equity, color="#1e7a2e", lw=1.5)
        ax.axhline(firm.initial_floor, color="#8a1f1f", lw=1.5)
        ax.axhline(firm.start_balance, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(f"{rr:g}:1 R:R, {wr:.0%} WR")
        ax.set_xlabel("Trade #"); ax.set_ylabel("Equity ($)")
        txt = (f"Pass: {res.pass_rate:.1%}\nFail: {res.fail_rate:.1%}\n"
               f"Ongoing: {res.ongoing_rate:.1%}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    for j in range(len(geometries), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Equity Paths — {firm.name}  (risk ${risk_per_trade:.0f}/trade)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2) STRATEGY COMPARISON SUMMARY
# ---------------------------------------------------------------------------
def plot_strategy_comparison(
    firm: FirmConfig,
    strategies: list[Strategy],
    risk_per_trade: float,
    n_paths: int = 5_000,
    max_trades: int = 400,
    trades_per_day: int = 3,
    seed: int = 0,
    savepath: str | None = None,
):
    labels, pr, fr, orr, speed = [], [], [], [], []
    for s in strategies:
        res = simulate(s, firm, risk_per_trade, n_paths=n_paths,
                       max_trades=max_trades, trades_per_day=trades_per_day, seed=seed)
        labels.append(s.name)
        pr.append(res.pass_rate); fr.append(res.fail_rate); orr.append(res.ongoing_rate)
        speed.append(res.mean_trades_to_pass)

    y = np.arange(len(labels))[::-1]  # top-to-bottom
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.7 * len(labels) + 2))

    ax = axes[0]
    ax.barh(y, np.array(pr) * 100, color="#2ecc71")
    for yi, v in zip(y, pr):
        ax.text(v * 100 + 1, yi, f"{v:.1%}", va="center", fontsize=9, fontweight="bold")
    ax.axvline(50, color="grey", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, 100)
    ax.set_xlabel("Pass Rate (%)"); ax.set_title("Challenge Pass Rate")

    ax = axes[1]
    pr_, fr_, or_ = np.array(pr) * 100, np.array(fr) * 100, np.array(orr) * 100
    ax.barh(y, pr_, color="#2ecc71", label="Pass")
    ax.barh(y, fr_, left=pr_, color="#e74c3c", label="Fail")
    ax.barh(y, or_, left=pr_ + fr_, color="#95a5a6", label="Ongoing")
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage (%)"); ax.set_title("Outcome Distribution")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    sp = np.array([0 if np.isnan(v) else v for v in speed])
    ax.barh(y, sp, color="#16a085")
    for yi, v in zip(y, speed):
        if not np.isnan(v):
            ax.text(v + max(sp) * 0.01, yi, f"{v:.0f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("Avg Trades to Pass"); ax.set_title("Speed to Target (winners only)")

    fig.suptitle(f"Strategy Comparison — {firm.name}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3) RISK / RR HEATMAP  (holds a fixed edge, sweeps risk% x R:R)
# ---------------------------------------------------------------------------
def compute_risk_rr_grid(
    firm: FirmConfig,
    risk_pcts: np.ndarray,          # e.g. np.arange(0.25, 4.0, 0.25) as percents
    rr_values: list[float],
    edge_R: float = 0.10,           # fixed per-trade expectancy in R
    n_paths: int = 3_000,
    max_trades: int = 400,
    trades_per_day: int = 3,
    seed: int = 0,
):
    pass_grid = np.zeros((len(risk_pcts), len(rr_values)))
    speed_grid = np.full_like(pass_grid, np.nan)
    wrs = []
    for j, rr in enumerate(rr_values):
        strat = SyntheticStrategy.from_edge(rr, edge_R=edge_R)
        wrs.append(strat.win_rate)
        for i, rp in enumerate(risk_pcts):
            risk_dollars = firm.start_balance * (rp / 100.0)
            res = simulate(strat, firm, risk_dollars, n_paths=n_paths,
                           max_trades=max_trades, trades_per_day=trades_per_day, seed=seed)
            pass_grid[i, j] = res.pass_rate * 100
            speed_grid[i, j] = res.mean_trades_to_pass
    return pass_grid, speed_grid, wrs


def plot_risk_rr_heatmap(
    firm: FirmConfig,
    risk_pcts: np.ndarray,
    rr_values: list[float],
    edge_R: float = 0.10,
    n_paths: int = 3_000,
    max_trades: int = 400,
    trades_per_day: int = 3,
    seed: int = 0,
    savepath: str | None = None,
):
    pass_grid, speed_grid, wrs = compute_risk_rr_grid(
        firm, risk_pcts, rr_values, edge_R, n_paths, max_trades, trades_per_day, seed)

    fig, ax = plt.subplots(figsize=(1.5 * len(rr_values) + 2, 0.5 * len(risk_pcts) + 2))
    im = ax.imshow(pass_grid, cmap=_HEAT, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(rr_values)))
    ax.set_xticklabels([f"{rr:g} RR\n{wr:.1%} WR" for rr, wr in zip(rr_values, wrs)])
    ax.set_yticks(range(len(risk_pcts)))
    ax.set_yticklabels([f"{p:.2f}%\n(${firm.start_balance*p/100:.0f})" for p in risk_pcts],
                       fontsize=8)
    ax.set_xlabel("Risk : Reward geometry")
    ax.set_ylabel("Risk per trade")

    for i in range(len(risk_pcts)):
        for j in range(len(rr_values)):
            v = pass_grid[i, j]
            sp = speed_grid[i, j]
            sp_txt = "" if np.isnan(sp) else f"\n{sp:.0f} trd"
            ax.text(j, i, f"{v:.0f}%{sp_txt}", ha="center", va="center",
                    fontsize=7, color="white" if v < 55 else "black")
    fig.colorbar(im, ax=ax, label="Pass rate (%)")
    ax.set_title(f"Pass Rate by Risk/Trade × R:R — {firm.name}\n"
                 f"(fixed edge = +{edge_R:g}R/trade, {trades_per_day} trades/day)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig, pass_grid, speed_grid
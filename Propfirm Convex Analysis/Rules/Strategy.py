"""
strategy.py — how a strategy feeds the Monte Carlo engine.

The engine only ever needs ONE thing from a strategy: a way to draw a matrix of
per-trade outcomes expressed in **R-multiples** (a win at 2:1 R:R is +2.0R, a
full loss is -1.0R). R-multiples are unit-free; the engine converts R -> $ using
risk_per_trade, so risk sizing (the prop-firm lever) stays separate from edge.

Two ways to define a strategy:

  1. SyntheticStrategy(win_rate, rr)          -- Bernoulli wins, fixed R:R.
     Use for the "risk geometry" experiments (hold EV constant, vary R:R).

  2. EmpiricalStrategy(returns_R)             -- bootstrap-resample the actual
     R-multiples your BOS / SuperTrend / EMA backtest produced. This preserves
     the real win rate, R:R spread, partial exits, fat tails, everything.

To plug in a real .py strategy file, have it produce a list/array of per-trade
R-multiples and wrap it: EmpiricalStrategy(returns_R=my_backtest_R). See
strategies/example_bos.py for the template.
"""

from __future__ import annotations
import numpy as np


class Strategy:
    """Base: sample(n_paths, n_trades, rng) -> ndarray[n_paths, n_trades] of R."""
    name: str = "strategy"

    def sample(self, n_paths: int, n_trades: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    @property
    def expectancy_R(self) -> float:
        """Mean R per trade (the edge). Handy for labelling."""
        raise NotImplementedError


class SyntheticStrategy(Strategy):
    def __init__(self, win_rate: float, rr: float, name: str | None = None):
        assert 0 < win_rate < 1
        self.win_rate = win_rate
        self.rr = rr
        self.name = name or f"WR={win_rate:.0%}, RR={rr:g}"

    def sample(self, n_paths, n_trades, rng):
        wins = rng.random((n_paths, n_trades)) < self.win_rate
        return np.where(wins, self.rr, -1.0)

    @property
    def expectancy_R(self) -> float:
        return self.win_rate * self.rr - (1 - self.win_rate) * 1.0

    @staticmethod
    def from_edge(rr: float, edge_R: float = 0.0, name: str | None = None) -> "SyntheticStrategy":
        """Build a strategy at a given R:R that has a fixed expectancy `edge_R`.

        EV = WR*RR - (1-WR)  ->  WR = (edge_R + 1) / (rr + 1)
        edge_R=0.0 -> break-even geometry (the EV=0 experiment in image 1).
        edge_R=0.10 -> the +0.10R edge used in the image-3 heatmap.
        """
        wr = (edge_R + 1.0) / (rr + 1.0)
        wr = min(max(wr, 1e-4), 1 - 1e-4)
        return SyntheticStrategy(win_rate=wr, rr=rr,
                                 name=name or f"RR={rr:g} (WR={wr:.1%})")


class EmpiricalStrategy(Strategy):
    """Bootstrap resampling of a real backtest's per-trade R-multiples."""

    def __init__(self, returns_R, name: str = "empirical", block: int = 1):
        self.returns_R = np.asarray(returns_R, dtype=float)
        assert self.returns_R.ndim == 1 and self.returns_R.size > 0
        self.name = name
        self.block = max(int(block), 1)  # block>1 = block bootstrap (keeps some autocorrelation)

    def sample(self, n_paths, n_trades, rng):
        r = self.returns_R
        if self.block == 1:
            idx = rng.integers(0, r.size, size=(n_paths, n_trades))
            return r[idx]
        # simple circular block bootstrap
        n_blocks = int(np.ceil(n_trades / self.block))
        starts = rng.integers(0, r.size, size=(n_paths, n_blocks))
        offs = np.arange(self.block)
        idx = (starts[:, :, None] + offs[None, None, :]) % r.size  # [paths, blocks, block]
        out = r[idx].reshape(n_paths, n_blocks * self.block)[:, :n_trades]
        return out

    @property
    def expectancy_R(self) -> float:
        return float(self.returns_R.mean())
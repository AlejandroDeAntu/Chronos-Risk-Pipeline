"""Backtest de VaR: Kupiec (cobertura) y Christoffersen (independencia)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.var_backtest import var_backtest


def _hits(positions: list[int], n: int) -> np.ndarray:
    returns = np.zeros(n)
    returns[positions] = -10.0
    return returns


def test_kupiec_matches_hand_calculation() -> None:
    # n=100, 5 excepciones, q=1%:
    # LR = -2[95 ln .99 + 5 ln .01 - (95 ln .95 + 5 ln .05)] = 8.2583
    returns = _hits([10, 30, 50, 70, 90], 100)
    result = var_backtest(returns, np.full(100, -1.0), level=0.01)

    assert result.n_hits == 5
    assert result.hit_rate == pytest.approx(0.05)
    assert result.lr_unconditional == pytest.approx(8.2583, abs=1e-4)
    assert result.p_unconditional < 0.01


def test_exact_coverage_gives_zero_statistic() -> None:
    returns = _hits([0, 20, 40, 60, 80], 100)
    result = var_backtest(returns, np.full(100, -1.0), level=0.05)
    assert result.lr_unconditional == pytest.approx(0.0, abs=1e-12)
    assert result.p_unconditional == pytest.approx(1.0)


def test_clustered_hits_fail_independence_spread_hits_pass() -> None:
    n = 400
    spread = var_backtest(
        _hits(list(range(0, n, 20)), n), np.full(n, -1.0), level=0.05
    )
    clustered = var_backtest(
        _hits(list(range(100, 120)), n), np.full(n, -1.0), level=0.05
    )

    assert spread.n_hits == clustered.n_hits == 20
    assert spread.p_independence > 0.05  # no se rechaza al 5%
    assert clustered.p_independence < 0.001
    assert clustered.p_conditional < 0.001


def test_no_hits_is_handled_without_nan() -> None:
    result = var_backtest(np.zeros(250), np.full(250, -1.0), level=0.01)
    assert result.n_hits == 0
    assert np.isfinite(result.lr_unconditional)
    assert np.isfinite(result.p_independence)


def test_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="longitud"):
        var_backtest(np.zeros(10), np.zeros(9), level=0.05)

"""Pruebas de retornos y bandas causales (sin fuga, sin retornos de salto)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exceptions import InsufficientHistoryError
from src.features.returns import (
    compute_causal_zscore_bands,
    compute_intraday_log_returns,
    compute_log_returns,
)


def test_compute_log_returns_matches_manual_calculation() -> None:
    prices = pd.Series(
        [100.0, 110.0, 99.0], index=pd.date_range("2026-01-01", periods=3)
    )
    expected = np.log(np.array([110.0 / 100.0, 99.0 / 110.0]))
    np.testing.assert_allclose(compute_log_returns(prices), expected)


def test_compute_log_returns_raises_on_insufficient_data() -> None:
    with pytest.raises(InsufficientHistoryError):
        compute_log_returns(
            pd.Series([100.0], index=pd.date_range("2026-01-01", periods=1))
        )


def test_intraday_returns_drop_returns_across_market_gaps() -> None:
    session_1 = pd.date_range("2026-09-17 19:50", periods=3, freq="5min")
    session_2 = pd.date_range("2026-09-18 13:30", periods=3, freq="5min")
    index = session_1.append(session_2).tz_localize("UTC")
    close = pd.Series([100, 101, 102, 110, 111, 112], index=index, dtype=float)

    returns = compute_intraday_log_returns(close, bar_minutes=5)

    # 5 retornos posibles; el que cruza la noche (102 -> 110) se descarta.
    assert len(returns) == 4
    assert index[3] not in returns.index


def test_causal_bands_never_use_current_bar_in_its_own_window(
    rng: np.random.Generator,
) -> None:
    index = pd.date_range("2026-01-01", periods=80, freq="D")
    base = pd.Series(rng.normal(0.0, 0.01, 80), index=index)
    perturbed = base.copy()
    perturbed.iloc[50] = 0.5  # Valor extremo en la barra 50.

    bands_base = compute_causal_zscore_bands(base, 10, 10)
    bands_pert = compute_causal_zscore_bands(perturbed, 10, 10)

    # La banda de la barra 50 no puede depender de su propio valor...
    for column in ("rolling_mean", "rolling_std"):
        assert bands_base.loc[index[50], column] == pytest.approx(
            bands_pert.loc[index[50], column]
        )
    # ...pero la de la barra 51 sí, porque ya es pasado.
    assert bands_base.loc[index[51], "rolling_std"] != pytest.approx(
        bands_pert.loc[index[51], "rolling_std"]
    )


def test_causal_bands_raise_when_history_too_short() -> None:
    index = pd.date_range("2026-01-01", periods=15, freq="D")
    returns = pd.Series(np.linspace(-0.01, 0.01, 15), index=index)
    with pytest.raises(InsufficientHistoryError):
        compute_causal_zscore_bands(
            returns, rolling_window=10, min_bars_required=50
        )

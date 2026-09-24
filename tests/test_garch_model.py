"""Pruebas del GARCH: ajuste, filtro causal y pronóstico no congelado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exceptions import InsufficientHistoryError
from src.features.returns import compute_log_returns
from src.models.garch_model import GarchVolatilityModel


@pytest.fixture()
def fitted(synthetic_daily_prices: pd.Series) -> GarchVolatilityModel:
    returns = compute_log_returns(synthetic_daily_prices)
    return GarchVolatilityModel.fit(returns.iloc[:1000], min_history_days=300)


def test_fit_recovers_stationary_parameters(
    fitted: GarchVolatilityModel,
) -> None:
    assert 0.0 < fitted.alpha < 1.0
    assert 0.0 < fitted.beta < 1.0
    assert fitted.alpha + fitted.beta < 1.0
    assert fitted.nu > 2.0


def test_fit_rejects_short_history(synthetic_daily_prices: pd.Series) -> None:
    returns = compute_log_returns(synthetic_daily_prices).iloc[:50]
    with pytest.raises(InsufficientHistoryError):
        GarchVolatilityModel.fit(returns, min_history_days=300)


def test_variance_filter_is_causal(fitted: GarchVolatilityModel) -> None:
    returns = np.random.default_rng(1).normal(0.0, 1.0, 200)
    shocked = returns.copy()
    shocked[120] = 15.0

    base = fitted.conditional_variance_path(returns)
    moved = fitted.conditional_variance_path(shocked)

    # h[120] solo usa retornos hasta 119: no puede ver el shock de 120.
    assert base[120] == pytest.approx(moved[120])
    assert moved[121] > base[121]


def test_forecast_updates_with_new_data(
    fitted: GarchVolatilityModel, synthetic_daily_prices: pd.Series
) -> None:
    """Regresión: el pronóstico no debe congelarse entre reentrenamientos."""
    returns_pct = compute_log_returns(synthetic_daily_prices).to_numpy() * 100
    first = fitted.forecast_next_volatility_pct(returns_pct[:1000])
    later = fitted.forecast_next_volatility_pct(returns_pct[:1100])
    assert first != pytest.approx(later)


def test_simulation_is_reproducible(fitted: GarchVolatilityModel) -> None:
    returns = np.random.default_rng(2).normal(0.0, 1.0, 300)
    first = fitted.simulate_paths(returns, 5, 200, rng_seed=7)
    second = fitted.simulate_paths(returns, 5, 200, rng_seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (200, 5)


def test_metadata_roundtrip(fitted: GarchVolatilityModel) -> None:
    assert GarchVolatilityModel.from_metadata(fitted.to_metadata()) == fitted

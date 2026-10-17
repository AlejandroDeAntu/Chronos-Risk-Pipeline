"""Escenarios de precio, cola de pérdidas y contrato del payload v2."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import PipelineSettings
from src.evaluation.volatility_selection import walk_forward_volatility
from src.models.scenarios import price_quantiles, tail_risk
from src.models.volatility import volatility_frame
from src.schemas import PriceQuantiles, VolatilityForecastPayload
from tests.helpers import synthetic_ohlc


def _normal_paths(n_paths: int = 200_000, horizon: int = 5) -> np.ndarray:
    return np.random.default_rng(0).normal(0.0, 1.0, (n_paths, horizon))


def test_price_quantiles_follow_cumulative_log_returns() -> None:
    rows = price_quantiles(100.0, _normal_paths())

    assert [r.day for r in rows] == [1, 2, 3, 4, 5]
    # Mediana de exp(N(0, k)) -> 100; p95 del día k: 100 e^{1.645 sqrt(k) %}.
    for row in rows:
        assert row.p50 == pytest.approx(100.0, rel=2e-3)
        expected = 100.0 * np.exp(1.6449 * np.sqrt(row.day) / 100.0)
        assert row.p95 == pytest.approx(expected, rel=2e-3)
    # La incertidumbre crece con el horizonte.
    widths = [r.p95 - r.p05 for r in rows]
    assert widths == sorted(widths)


def test_tail_risk_matches_normal_quantiles() -> None:
    var_95, cvar_95 = tail_risk(_normal_paths(), level=0.05)
    # VaR 5% de N(0, 1%) = -1.645%; CVaR = -phi(1.645)/0.05 % = -2.063%.
    assert var_95 == pytest.approx(-0.016449, rel=1e-2)
    assert cvar_95 == pytest.approx(-0.020627, rel=1e-2)


def test_price_quantiles_reject_non_monotonic_rows() -> None:
    with pytest.raises(ValidationError, match="ordenados"):
        PriceQuantiles(
            day=1, p05=101.0, p25=100.0, p50=100.5, p75=102, p95=103
        )


def test_payload_requires_one_row_per_horizon_day() -> None:
    rows = price_quantiles(100.0, _normal_paths(1000, 3))
    with pytest.raises(ValidationError, match="horizonte"):
        VolatilityForecastPayload(
            asset_symbol="^GSPC",
            model_version="v",
            model_family="har",
            forecast_origin=datetime.now(UTC),
            horizon_days=5,
            predicted_volatility_pct=1.0,
            var_95=-0.016,
            cvar_95=-0.02,
            var_99=-0.023,
            cvar_99=-0.027,
            scenarios_monte_carlo=rows,
            scenarios_fhs=rows,
        )


def test_payload_rejects_cvar_less_severe_than_var() -> None:
    rows = price_quantiles(100.0, _normal_paths(1000, 5))
    with pytest.raises(ValidationError, match="CVaR"):
        VolatilityForecastPayload(
            asset_symbol="^GSPC",
            model_version="v",
            model_family="garch",
            forecast_origin=datetime.now(UTC),
            horizon_days=5,
            predicted_volatility_pct=1.0,
            var_95=-0.016,
            cvar_95=-0.010,
            var_99=-0.023,
            cvar_99=-0.027,
            scenarios_monte_carlo=rows,
            scenarios_fhs=rows,
        )


def test_walk_forward_var_has_nominal_coverage_on_garch_data(
    synthetic_daily_prices: pd.Series, test_settings: PipelineSettings
) -> None:
    frame = volatility_frame(synthetic_ohlc(synthetic_daily_prices))
    wf = walk_forward_volatility(frame, test_settings, ("garch",))

    var_05 = wf.value_at_risk("garch", 0.05)
    hit_rate = float(np.mean(wf.realized_pct < var_05))
    # Datos simulados con el propio modelo: tasa cercana al 5% nominal.
    assert hit_rate == pytest.approx(0.05, abs=0.02)
    assert np.all(wf.value_at_risk("garch", 0.01) < var_05)

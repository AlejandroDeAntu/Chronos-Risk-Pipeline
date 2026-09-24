"""Pruebas de contratos Pydantic: rechazan payloads corruptos."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.schemas import AnomalyPredictionPayload, GarchForecastPayload


def test_anomaly_payload_accepts_valid_data() -> None:
    payload = AnomalyPredictionPayload(
        asset_symbol="^GSPC",
        model_version="20260101T000000Z",
        observed_at=datetime.now(UTC),
        log_return=0.01,
        rolling_mean=0.0001,
        rolling_std=0.002,
        zscore=2.5,
        is_anomaly=True,
        reversion_horizon_bars=10,
    )
    assert payload.is_anomaly is True


def test_anomaly_payload_rejects_negative_rolling_std() -> None:
    with pytest.raises(ValidationError):
        AnomalyPredictionPayload(
            asset_symbol="^GSPC",
            model_version="v1",
            observed_at=datetime.now(UTC),
            log_return=0.01,
            rolling_mean=0.0,
            rolling_std=-0.001,
            zscore=1.0,
            is_anomaly=False,
            reversion_horizon_bars=10,
        )


def test_anomaly_payload_rejects_nan_rolling_std() -> None:
    with pytest.raises(ValidationError):
        AnomalyPredictionPayload(
            asset_symbol="^GSPC",
            model_version="v1",
            observed_at=datetime.now(UTC),
            log_return=0.01,
            rolling_mean=0.0,
            rolling_std=float("nan"),
            zscore=1.0,
            is_anomaly=False,
            reversion_horizon_bars=10,
        )


def test_anomaly_payload_rejects_empty_symbol() -> None:
    with pytest.raises(ValidationError):
        AnomalyPredictionPayload(
            asset_symbol="",
            model_version="v1",
            observed_at=datetime.now(UTC),
            log_return=0.0,
            rolling_mean=0.0,
            rolling_std=0.001,
            zscore=0.0,
            is_anomaly=False,
            reversion_horizon_bars=10,
        )


def test_garch_payload_rejects_p95_below_p05() -> None:
    with pytest.raises(ValidationError):
        GarchForecastPayload(
            asset_symbol="SPY",
            model_version="v1",
            forecast_origin=datetime.now(UTC),
            horizon_days=5,
            predicted_volatility_pct=1.2,
            predicted_price_median=100.0,
            predicted_price_p05=110.0,
            predicted_price_p95=90.0,
            var_95=-0.02,
            cvar_95=-0.03,
        )


def test_garch_payload_rejects_non_positive_price() -> None:
    with pytest.raises(ValidationError):
        GarchForecastPayload(
            asset_symbol="SPY",
            model_version="v1",
            forecast_origin=datetime.now(UTC),
            horizon_days=5,
            predicted_volatility_pct=1.2,
            predicted_price_median=0.0,
            predicted_price_p05=90.0,
            predicted_price_p95=110.0,
            var_95=-0.02,
            cvar_95=-0.03,
        )

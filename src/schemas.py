"""Contratos Pydantic para los payloads que cruzan fronteras del sistema.

Pandera (``src/data/validation.py``) valida los DataFrames masivos de OHLCV;
Pydantic valida estos objetos pequeños y estructurados justo antes de
persistirlos o de usarlos para decidir un reentrenamiento.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class SignalType(StrEnum):
    """Tipo de señal generada por el pipeline."""

    ANOMALY = "anomaly"
    GARCH_VOLATILITY_FORECAST = "garch_volatility_forecast"


class AnomalyPredictionPayload(BaseModel):
    """Contrato de una predicción del detector de anomalías."""

    asset_symbol: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    observed_at: datetime
    log_return: float
    rolling_mean: float
    rolling_std: float = Field(gt=0.0)
    zscore: float
    is_anomaly: bool
    reversion_horizon_bars: int = Field(ge=1)

    @field_validator("log_return", "rolling_mean", "zscore", "rolling_std")
    @classmethod
    def _must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("El valor debe ser finito (no NaN ni infinito).")
        return value


class GarchForecastPayload(BaseModel):
    """Contrato de un forecast de volatilidad GARCH."""

    asset_symbol: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    forecast_origin: datetime
    horizon_days: int = Field(ge=1)
    predicted_volatility_pct: float = Field(gt=0.0)
    predicted_price_median: float = Field(gt=0.0)
    predicted_price_p05: float = Field(gt=0.0)
    predicted_price_p95: float = Field(gt=0.0)
    var_95: float
    cvar_95: float

    @field_validator("predicted_price_p95")
    @classmethod
    def _p95_gte_p05(cls, value: float, info: ValidationInfo) -> float:
        p05 = info.data.get("predicted_price_p05")
        if p05 is not None and value < p05:
            raise ValueError("El percentil 95 no puede ser menor que el 5.")
        return value


RetrainTrigger = Literal[
    "fixed_cadence", "degradation", "drift", "cooldown", "none"
]


class RetrainDecision(BaseModel):
    """Decisión de reentrenamiento emitida por ``feedback_loop``."""

    asset_symbol: str
    signal_type: SignalType
    should_retrain: bool
    reason: str = Field(min_length=1)
    triggered_by: RetrainTrigger
    evaluated_at: datetime

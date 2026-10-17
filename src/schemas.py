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

from pydantic import BaseModel, Field, field_validator, model_validator


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
    seasonal_factor: float = Field(default=1.0, gt=0.0)

    @field_validator("log_return", "rolling_mean", "zscore", "rolling_std")
    @classmethod
    def _must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("El valor debe ser finito (no NaN ni infinito).")
        return value


class PriceQuantiles(BaseModel):
    """Percentiles del precio al cierre de un día del horizonte."""

    day: int = Field(ge=1)
    p05: float = Field(gt=0.0)
    p25: float = Field(gt=0.0)
    p50: float = Field(gt=0.0)
    p75: float = Field(gt=0.0)
    p95: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> PriceQuantiles:
        levels = [self.p05, self.p25, self.p50, self.p75, self.p95]
        if levels != sorted(levels):
            raise ValueError(f"Los percentiles deben ir ordenados: {levels}.")
        return self


class VolatilityForecastPayload(BaseModel):
    """Contrato de un pronóstico de volatilidad con escenarios (v2).

    ``var_*`` y ``cvar_*`` son cuantiles del log-retorno del día 1 (en
    proporción, negativos para pérdidas) bajo el Monte Carlo del modelo.
    Los escenarios de precio se guardan por día del horizonte con dos
    métodos: Monte Carlo paramétrico y simulación histórica filtrada.
    """

    schema_version: Literal[2] = 2
    asset_symbol: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    forecast_origin: datetime
    horizon_days: int = Field(ge=1)
    predicted_volatility_pct: float = Field(gt=0.0)
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    scenarios_monte_carlo: list[PriceQuantiles]
    scenarios_fhs: list[PriceQuantiles]

    @model_validator(mode="after")
    def _consistent(self) -> VolatilityForecastPayload:
        for name in ("scenarios_monte_carlo", "scenarios_fhs"):
            if len(getattr(self, name)) != self.horizon_days:
                raise ValueError(
                    f"{name} debe tener una fila por día del horizonte "
                    f"({self.horizon_days})."
                )
        if self.cvar_95 > self.var_95 or self.cvar_99 > self.var_99:
            raise ValueError("El CVaR no puede ser menos severo que el VaR.")
        if self.var_99 > self.var_95:
            raise ValueError(
                "El VaR 99% no puede ser menos severo que el 95%."
            )
        return self


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

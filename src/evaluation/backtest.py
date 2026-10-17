"""Backtest walk-forward sobre datos históricos, sin escribir en la base.

Reutiliza exactamente la misma validación que decide las promociones en
producción (``walk_forward_anomaly`` y ``walk_forward_volatility``). El
resultado alimenta el reporte (``src/reporting``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.config import AssetConfig, PipelineSettings
from src.evaluation.volatility_selection import (
    ChampionDecision,
    VolatilityWalkForward,
    select_champion,
    walk_forward_volatility,
)
from src.exceptions import TradingPipelineError
from src.models.volatility import volatility_frame
from src.pipeline.predict import (
    load_daily_ohlc_with_repairs,
    load_intraday_returns,
)
from src.pipeline.retrain import AnomalyWalkForwardResult, walk_forward_anomaly
from src.schemas import SignalType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssetBacktest:
    """Resultado del backtest de un activo (``None`` si no se evaluó).

    Attributes
    ----------
    ohlc:
        Velas diarias validadas usadas en la validación.
    n_repaired_bars:
        Velas OHLC que hubo que reparar (calidad de la fuente).
    zero_return_share:
        Proporción de retornos intradía exactamente cero (cotizaciones
        rancias), o ``None`` si no se evaluó el detector.
    """

    asset: AssetConfig
    ohlc: pd.DataFrame | None
    n_repaired_bars: int
    zero_return_share: float | None
    anomaly: AnomalyWalkForwardResult | None
    volatility: VolatilityWalkForward | None
    champion: ChampionDecision | None
    errors: tuple[str, ...]


def _backtest_asset(
    asset: AssetConfig,
    settings: PipelineSettings,
    signals: tuple[SignalType, ...],
    now: datetime,
) -> AssetBacktest:
    """Valida un activo; los fallos quedan en ``errors``."""
    errors: list[str] = []
    anomaly = volatility = champion = ohlc = stale = None
    n_repaired = 0
    if SignalType.ANOMALY in signals:
        try:
            log_return = load_intraday_returns(
                asset, settings, settings.anomaly.history_period, now
            )
            stale = float((log_return == 0.0).mean())
            anomaly = walk_forward_anomaly(log_return, settings)
        except TradingPipelineError as exc:
            errors.append(f"anomaly: {type(exc).__name__}: {exc}")
    if SignalType.GARCH_VOLATILITY_FORECAST in signals:
        try:
            ohlc, n_repaired = load_daily_ohlc_with_repairs(
                asset, settings, now
            )
            volatility = walk_forward_volatility(
                volatility_frame(ohlc),
                settings,
                settings.garch.candidate_families,
            )
            champion = select_champion(
                volatility,
                settings.garch.default_family,
                settings.retrain.promotion_alpha,
            )
        except TradingPipelineError as exc:
            errors.append(f"volatility: {type(exc).__name__}: {exc}")
    for error in errors:
        logger.error("[backtest] %s: %s", asset.symbol, error)
    return AssetBacktest(
        asset=asset,
        ohlc=ohlc,
        n_repaired_bars=n_repaired,
        zero_return_share=stale,
        anomaly=anomaly,
        volatility=volatility,
        champion=champion,
        errors=tuple(errors),
    )


def run_backtest(
    settings: PipelineSettings,
    signals: tuple[SignalType, ...],
    now: datetime,
) -> list[AssetBacktest]:
    """Ejecuta la validación walk-forward de cada señal sobre cada activo.

    Un fallo en un activo o señal se registra en ``errors`` y no detiene el
    resto del backtest.
    """
    return [
        _backtest_asset(asset, settings, signals, now)
        for asset in settings.assets
    ]

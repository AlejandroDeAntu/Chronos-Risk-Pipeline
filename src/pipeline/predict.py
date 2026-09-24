"""Etapa de predicción: reemplaza el menú interactivo del proyecto original.

Genera y persiste, sin intervención humana, la señal más reciente de cada
activo configurado: anomalía intradía (velas de 5 minutos ya cerradas) o
volatilidad GARCH diaria. Si un activo no tiene modelo activo, se entrena el
primero (arranque en frío) y se registra como tal.

Un fallo en un activo se notifica y se registra en el resultado, pero no
detiene a los demás: ``run_pipeline.py`` decide el código de salida.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.config import AssetConfig, PipelineSettings
from src.data.fetch import fetch_daily_closes, fetch_intraday_bars
from src.data.validation import (
    validate_daily_closes,
    validate_intraday_closes,
    validate_log_returns,
)
from src.exceptions import DataContractError, TradingPipelineError
from src.features.returns import (
    compute_intraday_log_returns,
    compute_log_returns,
)
from src.models.anomaly_model import AnomalyDetector
from src.models.garch_model import GarchVolatilityModel
from src.notifications.notifier import notify_failure
from src.persistence import database, registry
from src.pipeline.results import AssetStageResult, Stage
from src.schemas import (
    AnomalyPredictionPayload,
    GarchForecastPayload,
    SignalType,
)

logger = logging.getLogger(__name__)

_COLD_START_METRICS = {"note": "arranque en frío, sin validación previa"}


def load_intraday_returns(
    asset: AssetConfig, settings: PipelineSettings, period: str, now: datetime
) -> pd.Series:
    """Descarga, valida y transforma velas intradía en retornos sin saltos.

    Raises
    ------
    DataUnavailableError, DataContractError, InsufficientHistoryError
        Si los datos no existen o no cumplen el contrato.
    """
    cfg = settings.anomaly
    frame = fetch_intraday_bars(
        symbol=asset.symbol,
        bar_minutes=cfg.bar_minutes,
        period=period,
        min_rows=cfg.min_bars_required + cfg.rolling_window,
        now=now,
    )
    frame = validate_intraday_closes(frame)
    log_return = compute_intraday_log_returns(frame["close"], cfg.bar_minutes)
    validate_log_returns(log_return.to_frame())
    return log_return


def load_daily_returns(
    asset: AssetConfig, settings: PipelineSettings
) -> tuple[pd.Series, pd.Series]:
    """Descarga y valida cierres diarios; devuelve ``(close, log_return)``.

    Raises
    ------
    DataUnavailableError, DataContractError, InsufficientHistoryError
        Si los datos no existen o no cumplen el contrato.
    """
    close = fetch_daily_closes(
        asset.symbol,
        settings.garch.history_start_date,
        settings.garch.min_history_days,
    )
    validate_daily_closes(close.to_frame())
    log_return = compute_log_returns(close)
    validate_log_returns(log_return.to_frame())
    return close, log_return


def predict_anomaly_for_asset(
    asset: AssetConfig, settings: PipelineSettings, run_ts: datetime
) -> AssetStageResult:
    """Calcula y persiste la señal de anomalía de la última vela cerrada.

    Parameters
    ----------
    asset:
        Activo a procesar.
    settings:
        Configuración del pipeline.
    run_ts:
        Instante UTC de la corrida (define qué velas ya cerraron).

    Returns
    -------
    AssetStageResult
        Id de la predicción insertada y si hubo arranque en frío.

    Raises
    ------
    TradingPipelineError
        Cualquier fallo de datos, contrato, modelo o persistencia.
    """
    cfg = settings.anomaly
    db_path = settings.paths.database_path
    detector = registry.load_active_anomaly_model(db_path, asset.symbol)
    cold_start = detector is None
    if detector is None:
        logger.warning(
            "'%s' sin detector activo: calibrando el primero con %s de "
            "historia (arranque en frío).",
            asset.symbol,
            cfg.history_period,
        )
        history = load_intraday_returns(
            asset, settings, cfg.history_period, run_ts
        )
        detector = AnomalyDetector.fit(
            log_return=history,
            rolling_window=cfg.rolling_window,
            target_anomaly_rate=cfg.target_anomaly_rate,
            reversion_horizon_bars=cfg.reversion_horizon_bars,
            min_bars_required=cfg.min_bars_required,
        )
        registry.register_candidate(
            db_path,
            asset.symbol,
            SignalType.ANOMALY,
            detector,
            _COLD_START_METRICS,
        )
        registry.promote_candidate(
            db_path, asset.symbol, SignalType.ANOMALY, detector.version
        )

    log_return = load_intraday_returns(
        asset, settings, cfg.predict_period, run_ts
    )
    latest = detector.predict(log_return, cfg.min_bars_required).iloc[-1]

    try:
        payload = AnomalyPredictionPayload(
            asset_symbol=asset.symbol,
            model_version=detector.version,
            observed_at=latest.name.to_pydatetime(),
            log_return=float(latest["log_return"]),
            rolling_mean=float(latest["rolling_mean"]),
            rolling_std=float(latest["rolling_std"]),
            zscore=float(latest["zscore"]),
            is_anomaly=bool(latest["is_anomaly"]),
            reversion_horizon_bars=detector.reversion_horizon_bars,
        )
    except ValidationError as exc:
        raise DataContractError(
            f"Predicción de anomalía inválida para '{asset.symbol}': {exc}"
        ) from exc

    prediction_id = database.insert_prediction(
        db_path=db_path,
        asset_symbol=asset.symbol,
        signal_type=SignalType.ANOMALY.value,
        model_version=detector.version,
        reference_timestamp=payload.observed_at,
        horizon_bars=detector.reversion_horizon_bars,
        payload=payload.model_dump(mode="json"),
        run_ts=run_ts,
    )
    detail = (
        f"vela {payload.observed_at:%Y-%m-%d %H:%M} UTC, "
        f"z={payload.zscore:.2f}, anomalía={payload.is_anomaly}"
    )
    if prediction_id is None:
        detail += " (ya registrada, no se duplica)"
    logger.info("[predict:anomaly] %s: %s", asset.symbol, detail)
    return AssetStageResult(
        asset_symbol=asset.symbol,
        signal_type=SignalType.ANOMALY,
        stage=Stage.PREDICT,
        succeeded=True,
        detail=detail,
        prediction_id=prediction_id,
        cold_start=cold_start,
    )


def predict_garch_for_asset(
    asset: AssetConfig, settings: PipelineSettings, run_ts: datetime
) -> AssetStageResult:
    """Calcula y persiste el forecast de volatilidad del siguiente día.

    Parameters
    ----------
    asset:
        Activo a procesar.
    settings:
        Configuración del pipeline.
    run_ts:
        Instante UTC de la corrida.

    Returns
    -------
    AssetStageResult
        Id de la predicción insertada y si hubo arranque en frío.

    Raises
    ------
    TradingPipelineError
        Cualquier fallo de datos, contrato, modelo o persistencia.
    """
    cfg = settings.garch
    db_path = settings.paths.database_path
    close, log_return = load_daily_returns(asset, settings)

    model = registry.load_active_garch_model(db_path, asset.symbol)
    cold_start = model is None
    if model is None:
        logger.warning(
            "'%s' sin GARCH activo: ajustando el primero (arranque en frío).",
            asset.symbol,
        )
        model = GarchVolatilityModel.fit(log_return, cfg.min_history_days)
        registry.register_candidate(
            db_path,
            asset.symbol,
            SignalType.GARCH_VOLATILITY_FORECAST,
            model,
            _COLD_START_METRICS,
        )
        registry.promote_candidate(
            db_path,
            asset.symbol,
            SignalType.GARCH_VOLATILITY_FORECAST,
            model.version,
        )

    returns_pct = (log_return * 100.0).to_numpy()
    forecast_origin = close.index[-1].to_pydatetime()
    # zlib.crc32 es determinista entre procesos (hash() de str no lo es).
    seed = zlib.crc32(
        f"{asset.symbol}|{model.version}|{forecast_origin}".encode()
    )
    sim_returns_pct = model.simulate_paths(
        returns_pct, cfg.forecast_horizon_days, cfg.n_monte_carlo_paths, seed
    )
    sim_log_returns = sim_returns_pct / 100.0
    final_prices = float(close.iloc[-1]) * np.exp(sim_log_returns.sum(axis=1))
    day1 = sim_log_returns[:, 0]
    var_95 = float(np.percentile(day1, 5.0))

    try:
        payload = GarchForecastPayload(
            asset_symbol=asset.symbol,
            model_version=model.version,
            forecast_origin=forecast_origin,
            horizon_days=cfg.forecast_horizon_days,
            predicted_volatility_pct=model.forecast_next_volatility_pct(
                returns_pct
            ),
            predicted_price_median=float(np.percentile(final_prices, 50.0)),
            predicted_price_p05=float(np.percentile(final_prices, 5.0)),
            predicted_price_p95=float(np.percentile(final_prices, 95.0)),
            var_95=var_95,
            cvar_95=float(day1[day1 <= var_95].mean()),
        )
    except ValidationError as exc:
        raise DataContractError(
            f"Forecast GARCH inválido para '{asset.symbol}': {exc}"
        ) from exc

    prediction_id = database.insert_prediction(
        db_path=db_path,
        asset_symbol=asset.symbol,
        signal_type=SignalType.GARCH_VOLATILITY_FORECAST.value,
        model_version=model.version,
        reference_timestamp=forecast_origin,
        horizon_bars=1,
        payload=payload.model_dump(mode="json"),
        run_ts=run_ts,
    )
    detail = (
        f"origen {forecast_origin:%Y-%m-%d}, "
        f"vol 1d={payload.predicted_volatility_pct:.3f}%, "
        f"VaR95={payload.var_95:.4f}"
    )
    if prediction_id is None:
        detail += " (ya registrado, no se duplica)"
    logger.info("[predict:garch] %s: %s", asset.symbol, detail)
    return AssetStageResult(
        asset_symbol=asset.symbol,
        signal_type=SignalType.GARCH_VOLATILITY_FORECAST,
        stage=Stage.PREDICT,
        succeeded=True,
        detail=detail,
        prediction_id=prediction_id,
        cold_start=cold_start,
    )


_PREDICTORS: dict[
    SignalType,
    Callable[[AssetConfig, PipelineSettings, datetime], AssetStageResult],
] = {
    SignalType.ANOMALY: predict_anomaly_for_asset,
    SignalType.GARCH_VOLATILITY_FORECAST: predict_garch_for_asset,
}


def run_predictions(
    settings: PipelineSettings,
    signal_type: SignalType,
    run_ts: datetime | None = None,
) -> list[AssetStageResult]:
    """Ejecuta la etapa de predicción sobre todos los activos configurados.

    Parameters
    ----------
    settings:
        Configuración del pipeline.
    signal_type:
        Señal a generar.
    run_ts:
        Instante UTC de la corrida; por defecto, ahora.

    Returns
    -------
    list[AssetStageResult]
        Un resultado por activo; los fallos quedan con ``succeeded=False``
        y ya fueron notificados.
    """
    timestamp = run_ts or datetime.now(UTC)
    predictor = _PREDICTORS[signal_type]
    results: list[AssetStageResult] = []
    for asset in settings.assets:
        try:
            results.append(predictor(asset, settings, timestamp))
        except TradingPipelineError as exc:
            notify_failure(f"predict:{signal_type.value}:{asset.symbol}", exc)
            results.append(
                AssetStageResult(
                    asset_symbol=asset.symbol,
                    signal_type=signal_type,
                    stage=Stage.PREDICT,
                    succeeded=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    return results

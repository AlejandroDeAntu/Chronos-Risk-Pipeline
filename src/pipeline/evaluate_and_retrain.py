"""Etapa de evaluación: cierra predicciones y decide si reentrenar.

Por cada activo:

1. Busca predicciones pendientes cuyo resultado real ya es observable y
   guarda, para el mismo punto, la pérdida del modelo y la de su baseline.
   En ``outcomes.error_metric`` se guarda siempre una **pérdida** (menor es
   mejor): 0/1 de error para anomalías, QLIKE para GARCH.
2. Si se cerró al menos un resultado nuevo, calcula la métrica agregada del
   modelo activo sobre su ventana reciente (F1 o QLIKE, contra baseline) y
   la guarda en ``evaluation_runs``. Si no hubo resultados nuevos no se
   registra corrida, para no contar dos veces la misma ventana.
3. Pide la decisión a ``feedback_loop.decide_retrain`` y, si corresponde,
   ejecuta ``retrain.py`` (walk-forward + promoción condicionada).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import AssetClass, AssetConfig, PipelineSettings
from src.evaluation.feedback_loop import decide_retrain
from src.evaluation.metrics import (
    evaluate_anomaly_classifier,
    evaluate_volatility_forecast,
    qlike_loss,
)
from src.exceptions import InsufficientHistoryError, TradingPipelineError
from src.models.anomaly_model import label_reversion_outcomes
from src.models.baseline import (
    random_anomaly_baseline,
    trailing_volatility_baseline,
)
from src.models.volatility import volatility_frame
from src.notifications.notifier import notify_failure
from src.persistence import database, registry
from src.pipeline.predict import load_daily_ohlc, load_intraday_returns
from src.pipeline.results import AssetStageResult, Stage
from src.pipeline.retrain import (
    retrain_anomaly_model,
    retrain_volatility_model,
)
from src.schemas import RetrainDecision, SignalType

logger = logging.getLogger(__name__)


def _close_anomaly_outcomes(
    settings: PipelineSettings,
    pending: list[dict[str, Any]],
    log_return: pd.Series,
    now: datetime,
) -> int:
    """Evalúa predicciones de anomalía cuyo horizonte ya transcurrió."""
    if not pending:
        return 0
    try:
        labeled = label_reversion_outcomes(
            log_return.to_frame(), settings.anomaly.reversion_horizon_bars
        )
    except InsufficientHistoryError:
        return 0

    horizon = settings.anomaly.reversion_horizon_bars
    closed = superseded = 0
    for row in pending:
        reference = pd.Timestamp(row["reference_timestamp"])
        if reference not in labeled.index:
            # Empate o barra fuera de la ventana: nunca tendrá etiqueta.
            # Si su horizonte ya pasó se retira de la cola (sin métrica);
            # si no, sigue pendiente hasta que se observe.
            if _never_labelable(reference, log_return.index, horizon):
                database.supersede_prediction(
                    settings.paths.database_path, row["id"]
                )
                superseded += 1
            continue
        reverted = bool(labeled.loc[reference, "reverted"])
        is_anomaly = bool(json.loads(row["payload_json"])["is_anomaly"])
        baseline_flag = bool(
            random_anomaly_baseline(
                1, settings.anomaly.target_anomaly_rate, rng_seed=row["id"]
            )[0]
        )
        database.insert_outcome(
            db_path=settings.paths.database_path,
            prediction_id=row["id"],
            realized_value=float(reverted),
            baseline_predicted_value=float(baseline_flag),
            error_metric=float(is_anomaly != reverted),
            baseline_error_metric=float(baseline_flag != reverted),
            evaluated_at=now,
        )
        closed += 1
    if superseded:
        logger.info(
            "%d predicciones de anomalía sin etiqueta posible (empates o "
            "fuera de la ventana) retiradas de la cola.",
            superseded,
        )
    return closed


def _never_labelable(
    reference: pd.Timestamp, index: pd.DatetimeIndex, horizon: int
) -> bool:
    """La barra ya no podrá etiquetarse.

    Salió de la ventana descargada, o sus ``horizon`` barras futuras ya se
    observaron sin producir etiqueta (empate o hueco en los datos).
    """
    if index.empty:
        return False
    if reference < index[0]:
        return True
    return index.searchsorted(reference) + horizon < len(index)


def _record_anomaly_run(
    asset: AssetConfig,
    settings: PipelineSettings,
    model_version: str,
    now: datetime,
) -> list[dict[str, Any]]:
    """Guarda F1 modelo vs baseline de la ventana reciente y la devuelve."""
    recent = database.fetch_recent_outcomes(
        settings.paths.database_path,
        asset.symbol,
        SignalType.ANOMALY.value,
        model_version,
        settings.retrain.degradation_window_anomaly,
    )
    if not recent:
        return recent
    report = evaluate_anomaly_classifier(
        np.array([r["realized_value"] == 1.0 for r in recent]),
        np.array(
            [json.loads(r["payload_json"])["is_anomaly"] for r in recent]
        ),
        np.array([r["baseline_predicted_value"] == 1.0 for r in recent]),
    )
    database.insert_evaluation_run(
        db_path=settings.paths.database_path,
        asset_symbol=asset.symbol,
        signal_type=SignalType.ANOMALY.value,
        model_version=model_version,
        evaluated_at=now,
        n_outcomes=report.n_samples,
        metric_name="f1",
        model_metric=report.f1,
        baseline_metric=report.baseline_f1,
        sufficient_data=(
            report.n_predicted_positive
            >= settings.retrain.min_predicted_positives
        ),
        report=asdict(report),
    )
    return recent


def _supersede_misaligned_fx(
    asset: AssetConfig,
    settings: PipelineSettings,
    pending: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retira predicciones FX v1 de la cola; devuelve las evaluables.

    Las predicciones FX con payload v1 se generaron con el cierre diario de
    Yahoo fechado al inicio del día; con la re-alineación su origen apunta
    a otro día y evaluarlas mediría otra cosa.
    """
    if asset.asset_class is not AssetClass.FX:
        return pending
    keep = []
    for row in pending:
        if "schema_version" in json.loads(row["payload_json"]):
            keep.append(row)
        else:
            database.supersede_prediction(
                settings.paths.database_path, row["id"]
            )
    if len(keep) < len(pending):
        logger.warning(
            "'%s': %d predicciones FX v1 marcadas como superadas.",
            asset.symbol,
            len(pending) - len(keep),
        )
    return keep


def _close_garch_outcomes(
    settings: PipelineSettings,
    pending: list[dict[str, Any]],
    log_return: pd.Series,
    now: datetime,
) -> int:
    """Evalúa forecasts GARCH cuyo día siguiente ya se observó."""
    if not pending:
        return 0
    returns_pct = log_return * 100.0
    baseline = trailing_volatility_baseline(
        returns_pct, settings.garch.baseline_trailing_window
    )
    closed = 0
    for row in pending:
        origin = pd.Timestamp(row["reference_timestamp"])
        if origin not in returns_pct.index:
            continue
        target = returns_pct.index.get_loc(origin) + 1
        if target >= len(returns_pct):
            continue  # El día pronosticado aún no cierra.
        baseline_vol = float(baseline.iloc[target])
        if not np.isfinite(baseline_vol) or baseline_vol <= 0.0:
            continue
        realized = np.array([returns_pct.iloc[target]])
        predicted = float(
            json.loads(row["payload_json"])["predicted_volatility_pct"]
        )
        database.insert_outcome(
            db_path=settings.paths.database_path,
            prediction_id=row["id"],
            realized_value=float(realized[0]),
            baseline_predicted_value=baseline_vol,
            error_metric=float(qlike_loss(np.array([predicted]), realized)[0]),
            baseline_error_metric=float(
                qlike_loss(np.array([baseline_vol]), realized)[0]
            ),
            evaluated_at=now,
        )
        closed += 1
    return closed


def _record_garch_run(
    asset: AssetConfig,
    settings: PipelineSettings,
    model_version: str,
    now: datetime,
) -> list[dict[str, Any]]:
    """Guarda QLIKE modelo vs baseline de la ventana reciente."""
    recent = database.fetch_recent_outcomes(
        settings.paths.database_path,
        asset.symbol,
        SignalType.GARCH_VOLATILITY_FORECAST.value,
        model_version,
        settings.retrain.degradation_window_garch,
    )
    if not recent:
        return recent
    report = evaluate_volatility_forecast(
        np.array(
            [
                json.loads(r["payload_json"])["predicted_volatility_pct"]
                for r in recent
            ]
        ),
        np.array([r["realized_value"] for r in recent]),
        np.array([r["baseline_predicted_value"] for r in recent]),
    )
    database.insert_evaluation_run(
        db_path=settings.paths.database_path,
        asset_symbol=asset.symbol,
        signal_type=SignalType.GARCH_VOLATILITY_FORECAST.value,
        model_version=model_version,
        evaluated_at=now,
        n_outcomes=report.n_samples,
        metric_name="qlike",
        model_metric=report.qlike,
        baseline_metric=report.baseline_qlike,
        sufficient_data=report.n_samples
        >= settings.retrain.min_outcomes_garch,
        report=asdict(report),
    )
    return recent


def _decide(
    asset: AssetConfig,
    settings: PipelineSettings,
    signal_type: SignalType,
    model_version: str,
    trained_at: datetime,
    recent: list[dict[str, Any]],
    higher_is_better: bool,
    now: datetime,
) -> RetrainDecision:
    """Arma las entradas del disparador desde la base y pide la decisión."""
    db_path = settings.paths.database_path
    runs = database.fetch_recent_evaluation_runs(
        db_path,
        asset.symbol,
        signal_type.value,
        model_version,
        settings.retrain.degradation_consecutive_windows,
    )
    return decide_retrain(
        asset_symbol=asset.symbol,
        signal_type=signal_type,
        last_trained_at=trained_at,
        last_registration_at=database.get_last_registration_time(
            db_path, asset.symbol, signal_type.value
        ),
        run_model_metrics=np.array([r["model_metric"] for r in runs]),
        run_baseline_metrics=np.array([r["baseline_metric"] for r in runs]),
        higher_is_better=higher_is_better,
        loss_stream=np.array([r["error_metric"] for r in recent]),
        config=settings.retrain,
        now=now,
    )


def evaluate_anomaly_for_asset(
    asset: AssetConfig, settings: PipelineSettings, now: datetime
) -> AssetStageResult:
    """Cierra resultados de anomalía, registra métricas y decide reentrenar.

    Raises
    ------
    TradingPipelineError
        Cualquier fallo de datos, contrato, modelo o persistencia.
    """
    db_path = settings.paths.database_path
    detector = registry.load_active_anomaly_model(db_path, asset.symbol)
    if detector is None:
        return AssetStageResult(
            asset_symbol=asset.symbol,
            signal_type=SignalType.ANOMALY,
            stage=Stage.EVALUATE,
            succeeded=True,
            detail="sin modelo activo todavía (se crea al predecir)",
        )

    log_return = load_intraday_returns(
        asset, settings, settings.anomaly.history_period, now
    )
    pending = database.fetch_pending_predictions(
        db_path, asset.symbol, SignalType.ANOMALY.value
    )
    closed = _close_anomaly_outcomes(settings, pending, log_return, now)
    recent = (
        _record_anomaly_run(asset, settings, detector.version, now)
        if closed
        else database.fetch_recent_outcomes(
            db_path,
            asset.symbol,
            SignalType.ANOMALY.value,
            detector.version,
            settings.retrain.degradation_window_anomaly,
        )
    )
    decision = _decide(
        asset,
        settings,
        SignalType.ANOMALY,
        detector.version,
        detector.trained_at,
        recent,
        higher_is_better=True,
        now=now,
    )
    promoted = None
    if decision.should_retrain:
        promoted = retrain_anomaly_model(asset, settings, log_return).promoted

    detail = (
        f"{closed} resultados cerrados, {len(pending) - closed} pendientes; "
        f"reentrenar={decision.should_retrain} ({decision.triggered_by})"
    )
    logger.info("[evaluate:anomaly] %s: %s", asset.symbol, detail)
    return AssetStageResult(
        asset_symbol=asset.symbol,
        signal_type=SignalType.ANOMALY,
        stage=Stage.EVALUATE,
        succeeded=True,
        detail=detail,
        outcomes_closed=closed,
        retrain_decision=decision,
        retrain_promoted=promoted,
    )


def evaluate_garch_for_asset(
    asset: AssetConfig, settings: PipelineSettings, now: datetime
) -> AssetStageResult:
    """Cierra resultados GARCH, registra métricas y decide reentrenar.

    Raises
    ------
    TradingPipelineError
        Cualquier fallo de datos, contrato, modelo o persistencia.
    """
    db_path = settings.paths.database_path
    signal = SignalType.GARCH_VOLATILITY_FORECAST
    model = registry.load_active_volatility_model(db_path, asset.symbol)
    if model is None:
        return AssetStageResult(
            asset_symbol=asset.symbol,
            signal_type=signal,
            stage=Stage.EVALUATE,
            succeeded=True,
            detail="sin modelo activo todavía (se crea al predecir)",
        )

    frame = volatility_frame(load_daily_ohlc(asset, settings, now))
    log_return = frame["return_pct"] / 100.0
    pending = _supersede_misaligned_fx(
        asset,
        settings,
        database.fetch_pending_predictions(
            db_path, asset.symbol, signal.value
        ),
    )
    closed = _close_garch_outcomes(settings, pending, log_return, now)
    recent = (
        _record_garch_run(asset, settings, model.version, now)
        if closed
        else database.fetch_recent_outcomes(
            db_path,
            asset.symbol,
            signal.value,
            model.version,
            settings.retrain.degradation_window_garch,
        )
    )
    migrate = asset.asset_class is AssetClass.FX and (
        registry.active_volatility_is_legacy(db_path, asset.symbol)
    )
    if migrate:
        # El modelo FX heredado se ajustó sobre velas desalineadas: se
        # reemplaza ya, sin esperar cadencia ni degradación.
        decision = RetrainDecision(
            asset_symbol=asset.symbol,
            signal_type=signal,
            should_retrain=True,
            reason="Modelo FX ajustado antes de re-alinear las velas.",
            triggered_by="migration",
            evaluated_at=now,
        )
    else:
        decision = _decide(
            asset,
            settings,
            signal,
            model.version,
            model.trained_at,
            recent,
            higher_is_better=False,
            now=now,
        )
    promoted = None
    if decision.should_retrain:
        promoted = retrain_volatility_model(
            asset, settings, frame, model.family, force_promote=migrate
        ).promoted

    detail = (
        f"{closed} resultados cerrados, {len(pending) - closed} pendientes; "
        f"reentrenar={decision.should_retrain} ({decision.triggered_by})"
    )
    logger.info("[evaluate:garch] %s: %s", asset.symbol, detail)
    return AssetStageResult(
        asset_symbol=asset.symbol,
        signal_type=signal,
        stage=Stage.EVALUATE,
        succeeded=True,
        detail=detail,
        outcomes_closed=closed,
        retrain_decision=decision,
        retrain_promoted=promoted,
    )


_EVALUATORS: dict[
    SignalType,
    Callable[[AssetConfig, PipelineSettings, datetime], AssetStageResult],
] = {
    SignalType.ANOMALY: evaluate_anomaly_for_asset,
    SignalType.GARCH_VOLATILITY_FORECAST: evaluate_garch_for_asset,
}


def run_evaluations(
    settings: PipelineSettings,
    signal_type: SignalType,
    now: datetime | None = None,
) -> list[AssetStageResult]:
    """Ejecuta la etapa de evaluación sobre todos los activos configurados.

    Parameters
    ----------
    settings:
        Configuración del pipeline.
    signal_type:
        Señal a evaluar.
    now:
        Instante UTC de la corrida; por defecto, ahora.

    Returns
    -------
    list[AssetStageResult]
        Un resultado por activo; los fallos quedan con ``succeeded=False``
        y ya fueron notificados.
    """
    timestamp = now or datetime.now(UTC)
    evaluator = _EVALUATORS[signal_type]
    results: list[AssetStageResult] = []
    for asset in settings.assets:
        try:
            results.append(evaluator(asset, settings, timestamp))
        except TradingPipelineError as exc:
            notify_failure(f"evaluate:{signal_type.value}:{asset.symbol}", exc)
            results.append(
                AssetStageResult(
                    asset_symbol=asset.symbol,
                    signal_type=signal_type,
                    stage=Stage.EVALUATE,
                    succeeded=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    return results

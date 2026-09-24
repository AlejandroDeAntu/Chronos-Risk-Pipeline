"""Validación walk-forward y reentrenamiento con promoción condicionada.

Un modelo reentrenado nunca reemplaza al activo solo por haber terminado de
entrenar. Antes:

1. Se valida el *procedimiento* de entrenamiento con
   ``sklearn.model_selection.TimeSeriesSplit`` (ventana expansiva, orden
   cronológico, nunca un split aleatorio). En cada bloque se ajusta solo con
   datos anteriores al bloque de prueba.
2. Para anomalías se usa ``gap = reversion_horizon_bars`` entre
   entrenamiento y prueba (purga): la etiqueta de una barra mira ``H``
   barras hacia adelante, así que ninguna barra de entrenamiento comparte
   ventana de resultado con la prueba. El GARCH no tiene horizonte de
   etiqueta (se ajusta por máxima verosimilitud sobre los propios
   retornos), por lo que no necesita hueco.
3. Las predicciones de todos los bloques se agrupan y se comparan contra el
   baseline sobre exactamente las mismas observaciones.

Solo si el candidato supera al baseline se promueve a ``active``; si no,
queda registrado como ``candidate`` para auditoría y el modelo anterior
sigue sirviendo. Las funciones ``walk_forward_*`` no escriben en la base de
datos, por eso también las usa el modo ``backtest`` de ``run_pipeline.py``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import AssetConfig, PipelineSettings
from src.evaluation.metrics import (
    ClassificationReport,
    RegressionReport,
    evaluate_anomaly_classifier,
    evaluate_volatility_forecast,
)
from src.exceptions import InsufficientHistoryError, ModelFitError
from src.models.anomaly_model import AnomalyDetector, label_reversion_outcomes
from src.models.baseline import (
    random_anomaly_baseline,
    trailing_volatility_baseline,
)
from src.models.garch_model import GarchVolatilityModel
from src.persistence import registry
from src.schemas import SignalType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyWalkForwardResult:
    """Resultado agregado de la validación walk-forward del detector."""

    n_folds: int
    fold_f1: tuple[float, ...]
    fold_baseline_f1: tuple[float, ...]
    pooled: ClassificationReport


@dataclass(frozen=True)
class GarchWalkForwardResult:
    """Resultado agregado de la validación walk-forward del GARCH."""

    n_folds: int
    fold_qlike: tuple[float, ...]
    fold_baseline_qlike: tuple[float, ...]
    pooled: RegressionReport
    standardized_residuals: tuple[float, ...]


@dataclass(frozen=True)
class RetrainOutcome:
    """Resultado de un intento de reentrenamiento."""

    candidate_version: str
    promoted: bool
    reason: str
    validation_metrics: dict[str, Any]


def walk_forward_anomaly(
    log_return: pd.Series, settings: PipelineSettings
) -> AnomalyWalkForwardResult:
    """Valida el detector de anomalías con walk-forward purgado.

    Parameters
    ----------
    log_return:
        Retornos intradía (sin retornos de salto), en orden cronológico.
    settings:
        Configuración del pipeline.

    Returns
    -------
    AnomalyWalkForwardResult
        Métricas por bloque y reporte agrupado contra el baseline.

    Raises
    ------
    InsufficientHistoryError
        Si ningún bloque tuvo datos suficientes para evaluarse.
    """
    cfg = settings.anomaly
    splitter = TimeSeriesSplit(
        n_splits=settings.retrain.walk_forward_splits,
        gap=cfg.reversion_horizon_bars,
    )
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    y_base_parts: list[np.ndarray] = []
    fold_f1: list[float] = []
    fold_baseline_f1: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(log_return)):
        train = log_return.iloc[train_idx]
        # Las bandas de la prueba usan la historia previa (incluido el
        # hueco) solo como pasado; se conservan únicamente filas de prueba.
        history = log_return.iloc[: test_idx[-1] + 1]
        test_index = log_return.index[test_idx]
        try:
            detector = AnomalyDetector.fit(
                log_return=train,
                rolling_window=cfg.rolling_window,
                target_anomaly_rate=cfg.target_anomaly_rate,
                reversion_horizon_bars=cfg.reversion_horizon_bars,
                min_bars_required=cfg.min_bars_required,
                version=f"walk_forward_fold_{fold}",
            )
            bands = detector.predict(history, cfg.min_bars_required)
            bands = bands[bands.index.isin(test_index)]
            labeled = label_reversion_outcomes(
                bands, cfg.reversion_horizon_bars
            )
        except InsufficientHistoryError as exc:
            logger.info("Bloque %d omitido: %s", fold, exc)
            continue

        y_true = labeled["reverted"].to_numpy(dtype=bool)
        y_pred = labeled["is_anomaly"].to_numpy(dtype=bool)
        y_base = random_anomaly_baseline(
            len(y_true), float(y_pred.mean()), rng_seed=fold
        )
        report = evaluate_anomaly_classifier(y_true, y_pred, y_base)
        fold_f1.append(report.f1)
        fold_baseline_f1.append(report.baseline_f1)
        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)
        y_base_parts.append(y_base)

    if not fold_f1:
        raise InsufficientHistoryError(
            "Ningún bloque walk-forward del detector tuvo datos suficientes."
        )
    pooled = evaluate_anomaly_classifier(
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        np.concatenate(y_base_parts),
    )
    return AnomalyWalkForwardResult(
        n_folds=len(fold_f1),
        fold_f1=tuple(fold_f1),
        fold_baseline_f1=tuple(fold_baseline_f1),
        pooled=pooled,
    )


def walk_forward_garch(
    log_return: pd.Series, settings: PipelineSettings
) -> GarchWalkForwardResult:
    """Valida el GARCH con pronósticos a un paso fuera de muestra.

    En cada bloque los parámetros se estiman solo con el entrenamiento y la
    varianza se filtra con parámetros fijos: el pronóstico del día ``t`` usa
    retornos hasta ``t-1``. El baseline es la desviación estándar móvil de
    los ``baseline_trailing_window`` días previos (``shift(1)``).

    Parameters
    ----------
    log_return:
        Retornos diarios, en orden cronológico.
    settings:
        Configuración del pipeline.

    Returns
    -------
    GarchWalkForwardResult
        QLIKE por bloque, reporte agrupado y residuos estandarizados.

    Raises
    ------
    InsufficientHistoryError
        Si ningún bloque pudo evaluarse.
    """
    cfg = settings.garch
    returns_pct = log_return * 100.0
    baseline_all = trailing_volatility_baseline(
        returns_pct, cfg.baseline_trailing_window
    ).to_numpy()
    values = returns_pct.to_numpy()
    splitter = TimeSeriesSplit(n_splits=settings.retrain.walk_forward_splits)

    predicted_parts: list[np.ndarray] = []
    realized_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    fold_qlike: list[float] = []
    fold_baseline_qlike: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(values)):
        try:
            model = GarchVolatilityModel.fit(
                log_return.iloc[train_idx],
                cfg.min_history_days,
                version=f"walk_forward_fold_{fold}",
            )
        except (InsufficientHistoryError, ModelFitError) as exc:
            logger.info("Bloque %d omitido: %s", fold, exc)
            continue

        variance = model.conditional_variance_path(values[: test_idx[-1] + 1])
        predicted = np.sqrt(variance[test_idx])
        realized = values[test_idx]
        baseline = baseline_all[test_idx]
        valid = np.isfinite(baseline) & (baseline > 0.0)

        report = evaluate_volatility_forecast(
            predicted[valid], realized[valid], baseline[valid]
        )
        fold_qlike.append(report.qlike)
        fold_baseline_qlike.append(report.baseline_qlike)
        predicted_parts.append(predicted[valid])
        realized_parts.append(realized[valid])
        baseline_parts.append(baseline[valid])

    if not fold_qlike:
        raise InsufficientHistoryError(
            "Ningún bloque walk-forward del GARCH tuvo datos suficientes."
        )
    predicted_all = np.concatenate(predicted_parts)
    realized_all = np.concatenate(realized_parts)
    pooled = evaluate_volatility_forecast(
        predicted_all, realized_all, np.concatenate(baseline_parts)
    )
    return GarchWalkForwardResult(
        n_folds=len(fold_qlike),
        fold_qlike=tuple(fold_qlike),
        fold_baseline_qlike=tuple(fold_baseline_qlike),
        pooled=pooled,
        standardized_residuals=tuple(realized_all / predicted_all),
    )


def retrain_anomaly_model(
    asset: AssetConfig, settings: PipelineSettings, log_return: pd.Series
) -> RetrainOutcome:
    """Valida, reentrena y promueve (si corresponde) el detector de un activo.

    Parameters
    ----------
    asset:
        Activo a reentrenar.
    settings:
        Configuración del pipeline.
    log_return:
        Retornos intradía disponibles (sin retornos de salto).

    Returns
    -------
    RetrainOutcome
        Versión candidata, si se promovió y por qué.
    """
    result = walk_forward_anomaly(log_return, settings)
    pooled = result.pooled
    enough_positives = (
        pooled.n_predicted_positive >= settings.retrain.min_predicted_positives
    )
    promote = pooled.beats_baseline_f1 and enough_positives

    candidate = AnomalyDetector.fit(
        log_return=log_return,
        rolling_window=settings.anomaly.rolling_window,
        target_anomaly_rate=settings.anomaly.target_anomaly_rate,
        reversion_horizon_bars=settings.anomaly.reversion_horizon_bars,
        min_bars_required=settings.anomaly.min_bars_required,
    )
    metrics = {
        "walk_forward_folds": result.n_folds,
        "pooled_report": asdict(pooled),
        "promoted": promote,
    }
    registry.register_candidate(
        settings.paths.database_path,
        asset.symbol,
        SignalType.ANOMALY,
        candidate,
        metrics,
    )
    reason = (
        f"F1 walk-forward {pooled.f1:.3f} vs baseline "
        f"{pooled.baseline_f1:.3f}; anomalías predichas "
        f"{pooled.n_predicted_positive}"
    )
    if promote:
        registry.promote_candidate(
            settings.paths.database_path,
            asset.symbol,
            SignalType.ANOMALY,
            candidate.version,
        )
        logger.info("Detector de '%s' promovido: %s.", asset.symbol, reason)
    else:
        logger.warning(
            "Detector candidato de '%s' NO promovido (%s); se mantiene el "
            "modelo activo.",
            asset.symbol,
            reason,
        )
    return RetrainOutcome(candidate.version, promote, reason, metrics)


def retrain_garch_model(
    asset: AssetConfig, settings: PipelineSettings, log_return: pd.Series
) -> RetrainOutcome:
    """Valida, reentrena y promueve (si corresponde) el GARCH de un activo.

    Parameters
    ----------
    asset:
        Activo a reentrenar.
    settings:
        Configuración del pipeline.
    log_return:
        Retornos diarios disponibles.

    Returns
    -------
    RetrainOutcome
        Versión candidata, si se promovió y por qué.
    """
    result = walk_forward_garch(log_return, settings)
    pooled = result.pooled
    promote = pooled.beats_baseline_qlike

    candidate = GarchVolatilityModel.fit(
        log_return, settings.garch.min_history_days
    )
    metrics = {
        "walk_forward_folds": result.n_folds,
        "pooled_report": asdict(pooled),
        "promoted": promote,
    }
    registry.register_candidate(
        settings.paths.database_path,
        asset.symbol,
        SignalType.GARCH_VOLATILITY_FORECAST,
        candidate,
        metrics,
    )
    reason = (
        f"QLIKE walk-forward {pooled.qlike:.4f} vs baseline "
        f"{pooled.baseline_qlike:.4f}"
    )
    if promote:
        registry.promote_candidate(
            settings.paths.database_path,
            asset.symbol,
            SignalType.GARCH_VOLATILITY_FORECAST,
            candidate.version,
        )
        logger.info("GARCH de '%s' promovido: %s.", asset.symbol, reason)
    else:
        logger.warning(
            "GARCH candidato de '%s' NO promovido (%s); se mantiene el "
            "modelo activo.",
            asset.symbol,
            reason,
        )
    return RetrainOutcome(candidate.version, promote, reason, metrics)

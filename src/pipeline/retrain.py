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
   ventana de resultado con la prueba. Los modelos de volatilidad no
   tienen horizonte de etiqueta (se ajustan sobre los propios retornos),
   por lo que no necesitan hueco; su walk-forward vive en
   ``src/evaluation/volatility_selection.py``.
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
    diebold_mariano,
    evaluate_anomaly_classifier,
    precision_lift_test,
)
from src.evaluation.volatility_selection import (
    select_champion,
    walk_forward_volatility,
)
from src.exceptions import InsufficientHistoryError
from src.models.anomaly_model import AnomalyDetector, label_reversion_outcomes
from src.models.baseline import (
    random_anomaly_baseline,
)
from src.models.volatility import fit_volatility_model
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
    # F1 contra UNA tirada aleatoria es casi una moneda al aire cuando el
    # detector está cerca del azar: se exige significancia sobre el lift.
    lift = precision_lift_test(pooled) if enough_positives else None
    promote = (
        lift is not None
        and lift.lift > 1.0
        and lift.p_value < settings.retrain.promotion_alpha
    )

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
        "precision_lift": asdict(lift) if lift is not None else None,
        "promoted": promote,
    }
    registry.register_candidate(
        settings.paths.database_path,
        asset.symbol,
        SignalType.ANOMALY,
        candidate,
        metrics,
    )
    lift_text = (
        f"lift precisión {lift.lift:.3f} (p={lift.p_value:.4f})"
        if lift is not None
        else "positivos insuficientes"
    )
    reason = f"{lift_text}; anomalías {pooled.n_predicted_positive}"
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


def retrain_volatility_model(
    asset: AssetConfig,
    settings: PipelineSettings,
    frame: pd.DataFrame,
    current_family: str,
) -> RetrainOutcome:
    """Selecciona la familia campeona, reentrena y promueve si corresponde.

    1. Walk-forward de todas las familias candidatas en los mismos días.
    2. Champion-challenger: el campeón vigente solo cambia si un retador le
       gana con Diebold-Mariano significativo tras el ajuste de Holm.
    3. Piso de calidad: el campeón debe superar al baseline SD-20d con
       Diebold-Mariano (``promotion_alpha``); si no, se registra como
       candidato y el modelo activo sigue sirviendo.

    Parameters
    ----------
    asset:
        Activo a reentrenar.
    settings:
        Configuración del pipeline.
    frame:
        Salida de ``volatility_frame`` con todo el historial disponible.
    current_family:
        Familia del modelo activo.

    Returns
    -------
    RetrainOutcome
        Versión candidata, si se promovió y por qué.
    """
    alpha = settings.retrain.promotion_alpha
    walk_forward = walk_forward_volatility(
        frame, settings, settings.garch.candidate_families
    )
    decision = select_champion(walk_forward, current_family, alpha)
    champion_loss = walk_forward.qlike(decision.champion)
    baseline_loss = walk_forward.qlike("sd20")
    common = walk_forward.common_mask
    dm = diebold_mariano(champion_loss[common], baseline_loss[common])
    promote = dm.p_value_one_sided < alpha

    candidate = fit_volatility_model(
        frame, decision.champion, settings.garch.min_history_days
    )
    metrics = {
        "champion_decision": asdict(decision),
        "mean_qlike": {
            name: walk_forward.mean_qlike(name)
            for name in (*walk_forward.forecasts, "sd20", "ewma")
        },
        "n_folds": walk_forward.n_folds,
        "failed_families": walk_forward.failed,
        "n_test_days": len(walk_forward.dates),
        "n_common_days": int(walk_forward.common_mask.sum()),
        "diebold_mariano_vs_sd20": asdict(dm),
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
        f"{decision.reason} Campeón '{decision.champion}' vs SD-20d: DM "
        f"t={dm.t_stat:.2f} (p={dm.p_value_one_sided:.4f})"
    )
    if promote:
        registry.promote_candidate(
            settings.paths.database_path,
            asset.symbol,
            SignalType.GARCH_VOLATILITY_FORECAST,
            candidate.version,
        )
        logger.info("Volatilidad de '%s' promovida: %s.", asset.symbol, reason)
    else:
        logger.warning(
            "Modelo de volatilidad candidato de '%s' NO promovido (%s); se "
            "mantiene el activo.",
            asset.symbol,
            reason,
        )
    return RetrainOutcome(candidate.version, promote, reason, metrics)

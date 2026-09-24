"""Disparador de reentrenamiento: enfriamiento, cadencia, degradación y drift.

Orden de evaluación y justificación:

1. **Enfriamiento** (``retrain_cooldown_hours``): si hubo un intento de
   entrenamiento reciente para ese activo/señal, no se reintenta. Sin esto,
   un candidato que no supera al baseline deja el modelo viejo activo y el
   mismo disparador se volvería a cumplir en la siguiente corrida.
2. **Cadencia fija**: red de seguridad aunque las métricas se vean bien
   (30 días anomalías, 7 días GARCH).
3. **Degradación sostenida**: en cada corrida de evaluación se guarda en
   ``evaluation_runs`` la métrica agregada del modelo activo y la de su
   baseline sobre la misma ventana (F1 para anomalías, QLIKE para GARCH).
   Se dispara si las últimas ``degradation_consecutive_windows`` corridas
   con datos suficientes están todas peor que el baseline. Una sola mala
   ventana es ruido esperable en series financieras; varias seguidas, no.
4. **Drift** (ADWIN) sobre la pérdida punto a punto del modelo activo.

Solo se consideran resultados de la versión de modelo activa: mezclar
versiones haría que un cambio viejo siguiera disparando reentrenamientos
después de haberse corregido.

Prevención de fuga de información: el disparador solo decide *cuándo*
reentrenar. El *cómo* (``src/pipeline/retrain.py``) valida al candidato con
``TimeSeriesSplit`` (orden cronológico, sin mezcla aleatoria) y con un
``gap`` igual al horizonte de la etiqueta entre entrenamiento y prueba, para
que ninguna observación de entrenamiento comparta ventana de resultado con
el bloque de prueba (purga, López de Prado, *Advances in Financial Machine
Learning*, cap. 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from river.drift import ADWIN

from src.config import RetrainTriggerConfig
from src.schemas import RetrainDecision, SignalType


@dataclass(frozen=True)
class DegradationCheckResult:
    """Resultado de comparar corridas recientes contra el baseline."""

    is_degraded: bool
    detail: str


def check_cooldown(
    last_registration_at: datetime | None,
    now: datetime,
    config: RetrainTriggerConfig,
) -> bool:
    """Indica si hubo un intento de entrenamiento en el enfriamiento."""
    if last_registration_at is None or config.retrain_cooldown_hours == 0:
        return False
    cooldown = timedelta(hours=config.retrain_cooldown_hours)
    return now - last_registration_at < cooldown


def check_fixed_cadence(
    last_trained_at: datetime,
    now: datetime,
    signal_type: SignalType,
    config: RetrainTriggerConfig,
) -> bool:
    """Indica si ya se cumplió la cadencia fija de reentrenamiento."""
    cadence_days = (
        config.fixed_cadence_days_anomaly
        if signal_type is SignalType.ANOMALY
        else config.fixed_cadence_days_garch
    )
    return now - last_trained_at >= timedelta(days=cadence_days)


def check_metric_degradation(
    run_model_metrics: np.ndarray,
    run_baseline_metrics: np.ndarray,
    higher_is_better: bool,
    consecutive_windows_required: int,
) -> DegradationCheckResult:
    """Evalúa si las últimas corridas están todas peor que el baseline.

    Parameters
    ----------
    run_model_metrics:
        Métrica agregada del modelo por corrida, en orden cronológico.
    run_baseline_metrics:
        Métrica del baseline en las mismas corridas.
    higher_is_better:
        ``True`` para F1, ``False`` para QLIKE.
    consecutive_windows_required:
        Corridas consecutivas más recientes que deben estar peor.

    Returns
    -------
    DegradationCheckResult
        Si hay degradación sostenida y el detalle.

    Raises
    ------
    ValueError
        Si los arreglos difieren en longitud.
    """
    if len(run_model_metrics) != len(run_baseline_metrics):
        raise ValueError("Las métricas de modelo y baseline deben coincidir.")
    if len(run_model_metrics) < consecutive_windows_required:
        return DegradationCheckResult(
            is_degraded=False,
            detail=(
                f"Solo hay {len(run_model_metrics)} corridas con datos "
                f"suficientes; se requieren {consecutive_windows_required}."
            ),
        )

    tail_model = run_model_metrics[-consecutive_windows_required:]
    tail_baseline = run_baseline_metrics[-consecutive_windows_required:]
    is_worse = (
        tail_model < tail_baseline
        if higher_is_better
        else tail_model > tail_baseline
    )
    return DegradationCheckResult(
        is_degraded=bool(np.all(is_worse)),
        detail=(
            f"Modelo peor que baseline en {int(is_worse.sum())}/"
            f"{consecutive_windows_required} corridas más recientes."
        ),
    )


def check_drift(loss_stream: np.ndarray) -> bool:
    """Aplica ADWIN sobre la pérdida punto a punto en orden cronológico.

    ADWIN compara sub-ventanas adaptativas y señala drift cuando sus medias
    difieren de forma significativa; no requiere fijar un tamaño de
    ventana. El recorrido es secuencial por diseño (detector en línea).
    """
    detector = ADWIN()
    for value in loss_stream:
        detector.update(float(value))
        if detector.drift_detected:
            return True
    return False


def decide_retrain(
    asset_symbol: str,
    signal_type: SignalType,
    last_trained_at: datetime,
    last_registration_at: datetime | None,
    run_model_metrics: np.ndarray,
    run_baseline_metrics: np.ndarray,
    higher_is_better: bool,
    loss_stream: np.ndarray,
    config: RetrainTriggerConfig,
    now: datetime | None = None,
) -> RetrainDecision:
    """Combina los cuatro criterios en una decisión única y auditable.

    Parameters
    ----------
    asset_symbol:
        Ticker evaluado.
    signal_type:
        Tipo de señal evaluada.
    last_trained_at:
        Entrenamiento del modelo activo.
    last_registration_at:
        Último intento de entrenamiento (promovido o no) de ese activo/señal.
    run_model_metrics, run_baseline_metrics:
        Métricas agregadas por corrida (``evaluation_runs``).
    higher_is_better:
        ``True`` para F1, ``False`` para QLIKE.
    loss_stream:
        Pérdida punto a punto del modelo activo, para ADWIN.
    config:
        Configuración de disparadores.
    now:
        Instante actual; se inyecta para poder probarlo.

    Returns
    -------
    RetrainDecision
        Decisión validada con su razón y disparador.
    """
    evaluated_at = now or datetime.now(UTC)

    def _decision(
        should_retrain: bool, reason: str, triggered_by: str
    ) -> RetrainDecision:
        return RetrainDecision(
            asset_symbol=asset_symbol,
            signal_type=signal_type,
            should_retrain=should_retrain,
            reason=reason,
            triggered_by=triggered_by,
            evaluated_at=evaluated_at,
        )

    if check_cooldown(last_registration_at, evaluated_at, config):
        return _decision(
            False,
            f"En enfriamiento: último intento {last_registration_at}.",
            "cooldown",
        )
    if check_fixed_cadence(last_trained_at, evaluated_at, signal_type, config):
        return _decision(
            True,
            f"Cadencia fija cumplida (entrenado {last_trained_at}).",
            "fixed_cadence",
        )

    degradation = check_metric_degradation(
        run_model_metrics,
        run_baseline_metrics,
        higher_is_better,
        config.degradation_consecutive_windows,
    )
    if degradation.is_degraded:
        return _decision(True, degradation.detail, "degradation")

    if config.drift_detection_enabled and check_drift(loss_stream):
        return _decision(
            True, "ADWIN detectó cambio en la pérdida del modelo.", "drift"
        )

    return _decision(False, "Ningún disparador se cumplió.", "none")

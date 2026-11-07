"""Resultados por activo y etapa para el resumen de una corrida."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from src.exceptions import TradingPipelineError
from src.notifications.notifier import notify_failure
from src.schemas import RetrainDecision, SignalType

logger = logging.getLogger(__name__)


class Stage(StrEnum):
    """Etapa del ciclo del pipeline."""

    PREDICT = "predict"
    EVALUATE = "evaluate"


@dataclass(frozen=True)
class AssetStageResult:
    """Resultado de una etapa para un activo y un tipo de señal.

    Attributes
    ----------
    asset_symbol, signal_type, stage:
        Qué se procesó.
    succeeded:
        ``False`` si la etapa falló para este activo.
    detail:
        Mensaje legible (causa del error si falló).
    prediction_id:
        Id insertado en ``predictions`` (``None`` si ya existía o falló).
    cold_start:
        ``True`` si se tuvo que entrenar el primer modelo del activo.
    outcomes_closed:
        Predicciones pendientes que se evaluaron contra el resultado real.
    retrain_decision:
        Decisión del disparador (``None`` si no había modelo activo).
    retrain_promoted:
        ``True``/``False`` si hubo reentrenamiento, ``None`` si no lo hubo.
    """

    asset_symbol: str
    signal_type: SignalType
    stage: Stage
    succeeded: bool
    detail: str
    prediction_id: int | None = None
    cold_start: bool = False
    outcomes_closed: int = 0
    retrain_decision: RetrainDecision | None = None
    retrain_promoted: bool | None = None


def run_isolated(
    stage_fn: Callable[[], AssetStageResult],
    asset_symbol: str,
    signal_type: SignalType,
    stage: Stage,
) -> AssetStageResult:
    """Ejecuta la etapa de un activo sin que su fallo detenga a los demás.

    Los errores del dominio (``TradingPipelineError``) son los esperados.
    Cualquier otro (un ``KeyError`` de pandas, un ``LinAlgError`` de arch)
    es un bug, pero en un pipeline desatendido no debe dejar sin señal a
    los otros seis activos: se registra con traceback, se notifica y el
    activo queda como fallido, lo que hace terminar la corrida con código
    1 (ver ``run_pipeline.py``).
    """
    context = f"{stage.value}:{signal_type.value}:{asset_symbol}"
    try:
        return stage_fn()
    except TradingPipelineError as exc:
        error: Exception = exc
    except Exception as exc:  # noqa: BLE001 - frontera de aislamiento
        logger.exception("Error no previsto en %s", context)
        error = exc
    notify_failure(context, error)
    return AssetStageResult(
        asset_symbol=asset_symbol,
        signal_type=signal_type,
        stage=stage,
        succeeded=False,
        detail=f"{type(error).__name__}: {error}",
    )

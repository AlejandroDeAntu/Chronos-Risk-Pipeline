"""Resultados por activo y etapa para el resumen de una corrida."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.schemas import RetrainDecision, SignalType


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

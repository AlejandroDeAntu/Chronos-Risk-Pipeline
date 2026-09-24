"""Registro de modelos: capa tipada sobre la tabla ``model_registry``.

Traduce entre los objetos de dominio (``AnomalyDetector``,
``GarchVolatilityModel``) y las filas JSON-serializadas de
``src.persistence.database``, para que el resto del pipeline nunca maneje
JSON crudo directamente.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.exceptions import ModelRegistryError
from src.models.anomaly_model import AnomalyDetector
from src.models.garch_model import GarchVolatilityModel
from src.persistence import database
from src.schemas import SignalType


def register_candidate(
    db_path: Path,
    asset_symbol: str,
    signal_type: SignalType,
    model: AnomalyDetector | GarchVolatilityModel,
    validation_metrics: dict[str, float],
) -> None:
    """Registra un modelo recién entrenado como ``candidate``.

    Un modelo ``candidate`` nunca se usa para predecir en producción hasta
    que ``promote_if_better`` (ver ``src/pipeline/retrain.py``) lo valide
    contra el modelo activo y el baseline, y lo promueva explícitamente.

    Parameters
    ----------
    db_path:
        Ruta a la base de datos SQLite.
    asset_symbol:
        Ticker del activo.
    signal_type:
        Tipo de señal del modelo.
    model:
        Instancia de ``AnomalyDetector`` o ``GarchVolatilityModel``.
    validation_metrics:
        Métricas de validación walk-forward obtenidas antes de registrar.
    """
    database.insert_model_version(
        db_path=db_path,
        asset_symbol=asset_symbol,
        signal_type=signal_type.value,
        version=model.version,
        stage="candidate",
        metadata=model.to_metadata(),
        validation_metrics=validation_metrics,
        created_at=datetime.now(UTC),
    )


def promote_candidate(
    db_path: Path, asset_symbol: str, signal_type: SignalType, version: str
) -> None:
    """Promueve una versión ``candidate`` a ``active`` y retira la previa."""
    database.promote_model(db_path, asset_symbol, signal_type.value, version)


def load_active_anomaly_model(
    db_path: Path, asset_symbol: str
) -> AnomalyDetector | None:
    """Carga el ``AnomalyDetector`` activo de un activo, si existe.

    Parameters
    ----------
    db_path:
        Ruta a la base de datos SQLite.
    asset_symbol:
        Ticker del activo.

    Returns
    -------
    AnomalyDetector | None
        El detector activo, o ``None`` si nunca se ha entrenado uno.

    Raises
    ------
    ModelRegistryError
        Si la metadata almacenada está corrupta o incompleta.
    """
    row = database.get_active_model(
        db_path, asset_symbol, SignalType.ANOMALY.value
    )
    if row is None:
        return None
    try:
        return AnomalyDetector.from_metadata(json.loads(row["metadata_json"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise ModelRegistryError(
            f"Metadata corrupta del detector activo de '{asset_symbol}': {exc}"
        ) from exc


def load_active_garch_model(
    db_path: Path, asset_symbol: str
) -> GarchVolatilityModel | None:
    """Carga el ``GarchVolatilityModel`` activo de un activo, si existe.

    Parameters
    ----------
    db_path:
        Ruta a la base de datos SQLite.
    asset_symbol:
        Ticker del activo.

    Returns
    -------
    GarchVolatilityModel | None
        El modelo activo, o ``None`` si nunca se ha entrenado uno.

    Raises
    ------
    ModelRegistryError
        Si la metadata almacenada está corrupta o incompleta.
    """
    row = database.get_active_model(
        db_path, asset_symbol, SignalType.GARCH_VOLATILITY_FORECAST.value
    )
    if row is None:
        return None
    try:
        return GarchVolatilityModel.from_metadata(
            json.loads(row["metadata_json"])
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ModelRegistryError(
            f"Metadata corrupta del GARCH activo de '{asset_symbol}': {exc}"
        ) from exc

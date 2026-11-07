"""Registro de modelos: capa tipada sobre la tabla ``model_registry``.

Traduce entre los objetos de dominio (``AnomalyDetector``,
modelos de volatilidad) y las filas JSON-serializadas de
``src.persistence.database``, para que el resto del pipeline nunca maneje
JSON crudo directamente.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.exceptions import ModelRegistryError
from src.models.anomaly_model import AnomalyDetector
from src.models.volatility import VolatilityModel, model_from_metadata
from src.persistence import database
from src.schemas import SignalType


def register_candidate(
    db_path: Path,
    asset_symbol: str,
    signal_type: SignalType,
    model: AnomalyDetector | VolatilityModel,
    validation_metrics: dict[str, Any],
    promote: bool = False,
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
        ``AnomalyDetector`` o cualquier familia de volatilidad.
    validation_metrics:
        Métricas de validación walk-forward obtenidas antes de registrar.
    promote:
        Registrar y promover en una sola transacción.
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
        promote=promote,
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


def active_volatility_is_legacy(db_path: Path, asset_symbol: str) -> bool:
    """Indica si el modelo de volatilidad activo es anterior a las familias.

    Los registros sin ``family`` se ajustaron antes de re-alinear las velas
    FX de Yahoo; en FX deben reemplazarse cuanto antes.
    """
    row = database.get_active_model(
        db_path, asset_symbol, SignalType.GARCH_VOLATILITY_FORECAST.value
    )
    return row is not None and "family" not in json.loads(row["metadata_json"])


_EVIDENCE_PRESERVING_TRIGGERS = frozenset({"fixed_cadence"})


def volatility_lineage(db_path: Path, asset_symbol: str) -> tuple[str, ...]:
    """Versiones cuyo historial cuenta como evidencia del modelo activo.

    El reajuste semanal por cadencia solo agrega días de datos a la misma
    familia: si cada versión empezara de cero, la ventana de degradación
    (60 resultados) nunca se llenaría. El linaje se corta, y la evidencia
    se reinicia, cuando cambia la familia, cuando el reentreno lo disparó
    la degradación, el drift o una migración (el modelo viejo era el
    problema), y en registros previos a las familias (velas FX
    desalineadas). Un registro sin ``triggered_by`` es anterior a que se
    guardara el disparador; todos esos reentrenos fueron por cadencia.

    Returns
    -------
    tuple[str, ...]
        Versiones, de la activa hacia atrás; vacía sin modelo activo.
    """
    rows = database.fetch_promoted_versions(
        db_path, asset_symbol, SignalType.GARCH_VOLATILITY_FORECAST.value
    )
    lineage: list[str] = []
    family: str | None = None
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        if "family" not in metadata or (
            family is not None and metadata["family"] != family
        ):
            break
        family = metadata["family"]
        lineage.append(row["version"])
        trigger = json.loads(row["validation_metrics_json"]).get(
            "triggered_by", "fixed_cadence"
        )
        if trigger not in _EVIDENCE_PRESERVING_TRIGGERS:
            break
    return tuple(lineage)


def load_active_volatility_model(
    db_path: Path, asset_symbol: str
) -> VolatilityModel | None:
    """Carga el modelo de volatilidad activo de un activo, si existe.

    La familia se toma de la metadata; los registros previos a la
    selección de familias se cargan como GARCH(1,1).

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
        return model_from_metadata(json.loads(row["metadata_json"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise ModelRegistryError(
            "Metadata corrupta del modelo de volatilidad activo de "
            f"'{asset_symbol}': {exc}"
        ) from exc

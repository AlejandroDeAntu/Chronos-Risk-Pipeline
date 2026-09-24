"""Persistencia SQLite: predicciones, resultados, métricas y modelos.

Se usa ``sqlite3`` de la librería estándar porque el volumen (7 activos, una
fila por corrida) es trivial para un solo operador; el README describe la
migración a Postgres si aparecen escritores concurrentes. Cada función abre
y cierra su propia conexión para que cada operación sea atómica.

Las métricas se agregan con SQL en la base (filtrado por activo, señal y
versión de modelo antes de extraer), siguiendo la jerarquía SQL > Polars >
Pandas: a Python solo llegan las filas de la ventana que se va a evaluar.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from src.exceptions import PersistenceError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts TEXT NOT NULL,
    asset_symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    model_version TEXT NOT NULL,
    reference_timestamp TEXT NOT NULL,
    horizon_bars INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    evaluated INTEGER NOT NULL DEFAULT 0,
    UNIQUE(asset_symbol, signal_type, reference_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_predictions_pending
    ON predictions (asset_symbol, signal_type, evaluated);

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL UNIQUE REFERENCES predictions(id),
    realized_value REAL NOT NULL,
    baseline_predicted_value REAL NOT NULL,
    error_metric REAL NOT NULL,
    baseline_error_metric REAL NOT NULL,
    evaluated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    model_version TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    n_outcomes INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    model_metric REAL,
    baseline_metric REAL,
    sufficient_data INTEGER NOT NULL,
    report_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    version TEXT NOT NULL,
    stage TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    validation_metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(asset_symbol, signal_type, version)
);

CREATE INDEX IF NOT EXISTS idx_registry_active
    ON model_registry (asset_symbol, signal_type, stage);
"""


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Abre una conexión con claves foráneas activas y la cierra al salir.

    Raises
    ------
    PersistenceError
        Ante cualquier ``sqlite3.Error`` (conexión, SQL o integridad); la
        transacción se revierte.
    """
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        raise PersistenceError(f"No se pudo abrir '{db_path}': {exc}") from exc
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise PersistenceError(
            f"Error de SQLite en '{db_path}': {exc}"
        ) from exc
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    """Crea las tablas del pipeline si no existen."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def insert_prediction(
    db_path: Path,
    asset_symbol: str,
    signal_type: str,
    model_version: str,
    reference_timestamp: datetime,
    horizon_bars: int,
    payload: dict[str, Any],
    run_ts: datetime,
) -> int | None:
    """Inserta una predicción si no existe ya una para esa barra.

    ``INSERT OR IGNORE`` sobre ``UNIQUE(asset_symbol, signal_type,
    reference_timestamp)`` hace que reejecutar sobre la misma barra sea
    idempotente.

    Returns
    -------
    int | None
        Id de la fila insertada, o ``None`` si ya existía.
    """
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO predictions
                (run_ts, asset_symbol, signal_type, model_version,
                 reference_timestamp, horizon_bars, payload_json, evaluated)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                run_ts.isoformat(),
                asset_symbol,
                signal_type,
                model_version,
                reference_timestamp.isoformat(),
                horizon_bars,
                json.dumps(payload),
            ),
        )
        return int(cursor.lastrowid) if cursor.rowcount > 0 else None


def fetch_pending_predictions(
    db_path: Path, asset_symbol: str, signal_type: str
) -> list[dict[str, Any]]:
    """Devuelve predicciones no evaluadas, más antiguas primero."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, run_ts, asset_symbol, signal_type, model_version,
                   reference_timestamp, horizon_bars, payload_json
            FROM predictions
            WHERE asset_symbol = ? AND signal_type = ? AND evaluated = 0
            ORDER BY reference_timestamp ASC
            """,
            (asset_symbol, signal_type),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_outcome(
    db_path: Path,
    prediction_id: int,
    realized_value: float,
    baseline_predicted_value: float,
    error_metric: float,
    baseline_error_metric: float,
    evaluated_at: datetime,
) -> None:
    """Inserta el resultado real y marca la predicción como evaluada.

    Se guardan la predicción del baseline y los errores de modelo y baseline
    para el mismo punto, de modo que las métricas agregadas se pueden
    reconstruir después sin volver a generar aleatoriedad.
    """
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO outcomes
                (prediction_id, realized_value, baseline_predicted_value,
                 error_metric, baseline_error_metric, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                realized_value,
                baseline_predicted_value,
                error_metric,
                baseline_error_metric,
                evaluated_at.isoformat(),
            ),
        )
        conn.execute(
            "UPDATE predictions SET evaluated = 1 WHERE id = ?",
            (prediction_id,),
        )


def fetch_recent_outcomes(
    db_path: Path,
    asset_symbol: str,
    signal_type: str,
    model_version: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Últimos resultados de una versión de modelo, en orden cronológico.

    El filtro por ``model_version`` es deliberado: la degradación se mide
    solo sobre el modelo que está sirviendo, no sobre versiones anteriores.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT p.id AS prediction_id, p.reference_timestamp,
                       p.payload_json, o.realized_value,
                       o.baseline_predicted_value, o.error_metric,
                       o.baseline_error_metric
                FROM outcomes o
                JOIN predictions p ON p.id = o.prediction_id
                WHERE p.asset_symbol = ? AND p.signal_type = ?
                  AND p.model_version = ?
                ORDER BY p.reference_timestamp DESC
                LIMIT ?
            ) ORDER BY reference_timestamp ASC
            """,
            (asset_symbol, signal_type, model_version, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_evaluation_run(
    db_path: Path,
    asset_symbol: str,
    signal_type: str,
    model_version: str,
    evaluated_at: datetime,
    n_outcomes: int,
    metric_name: str,
    model_metric: float | None,
    baseline_metric: float | None,
    sufficient_data: bool,
    report: dict[str, Any],
) -> None:
    """Registra la métrica agregada (modelo vs baseline) de una corrida."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO evaluation_runs
                (asset_symbol, signal_type, model_version, evaluated_at,
                 n_outcomes, metric_name, model_metric, baseline_metric,
                 sufficient_data, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_symbol,
                signal_type,
                model_version,
                evaluated_at.isoformat(),
                n_outcomes,
                metric_name,
                model_metric,
                baseline_metric,
                int(sufficient_data),
                json.dumps(report),
            ),
        )


def fetch_recent_evaluation_runs(
    db_path: Path,
    asset_symbol: str,
    signal_type: str,
    model_version: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Últimas corridas con datos suficientes, en orden cronológico."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT evaluated_at, n_outcomes, metric_name, model_metric,
                       baseline_metric
                FROM evaluation_runs
                WHERE asset_symbol = ? AND signal_type = ?
                  AND model_version = ? AND sufficient_data = 1
                ORDER BY evaluated_at DESC
                LIMIT ?
            ) ORDER BY evaluated_at ASC
            """,
            (asset_symbol, signal_type, model_version, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_model_version(
    db_path: Path,
    asset_symbol: str,
    signal_type: str,
    version: str,
    stage: str,
    metadata: dict[str, Any],
    validation_metrics: dict[str, Any],
    created_at: datetime,
) -> int:
    """Registra una nueva versión de modelo y devuelve su id."""
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_registry
                (asset_symbol, signal_type, version, stage, metadata_json,
                 validation_metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_symbol,
                signal_type,
                version,
                stage,
                json.dumps(metadata),
                json.dumps(validation_metrics),
                created_at.isoformat(),
            ),
        )
        return int(cursor.lastrowid)


def get_active_model(
    db_path: Path, asset_symbol: str, signal_type: str
) -> dict[str, Any] | None:
    """Fila del modelo activo, o ``None`` si nunca se entrenó uno."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, asset_symbol, signal_type, version, stage,
                   metadata_json, validation_metrics_json, created_at
            FROM model_registry
            WHERE asset_symbol = ? AND signal_type = ? AND stage = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (asset_symbol, signal_type),
        ).fetchone()
    return dict(row) if row is not None else None


def get_last_registration_time(
    db_path: Path, asset_symbol: str, signal_type: str
) -> datetime | None:
    """Momento del último intento de entrenamiento (cualquier etapa)."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT MAX(created_at) AS last_created_at
            FROM model_registry
            WHERE asset_symbol = ? AND signal_type = ?
            """,
            (asset_symbol, signal_type),
        ).fetchone()
    if row is None or row["last_created_at"] is None:
        return None
    return datetime.fromisoformat(row["last_created_at"])


def promote_model(
    db_path: Path, asset_symbol: str, signal_type: str, new_version: str
) -> None:
    """Activa ``new_version`` y retira el activo previo, en una transacción.

    Raises
    ------
    PersistenceError
        Si ``new_version`` no existe en el registro (la promoción no ocurre).
    """
    with _connect(db_path) as conn:
        exists = conn.execute(
            """
            SELECT 1 FROM model_registry
            WHERE asset_symbol = ? AND signal_type = ? AND version = ?
            """,
            (asset_symbol, signal_type, new_version),
        ).fetchone()
        if exists is None:
            raise PersistenceError(
                f"No existe la versión '{new_version}' de {signal_type} para "
                f"'{asset_symbol}'; no se puede promover."
            )
        conn.execute(
            """
            UPDATE model_registry SET stage = 'retired'
            WHERE asset_symbol = ? AND signal_type = ? AND stage = 'active'
            """,
            (asset_symbol, signal_type),
        )
        conn.execute(
            """
            UPDATE model_registry SET stage = 'active'
            WHERE asset_symbol = ? AND signal_type = ? AND version = ?
            """,
            (asset_symbol, signal_type, new_version),
        )

"""Cierre de resultados y evaluación en vivo del detector de anomalías."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AssetConfig, PipelineSettings
from src.persistence import database
from src.pipeline.evaluate_and_retrain import (
    _close_anomaly_outcomes,
    _record_anomaly_run,
)
from src.schemas import SignalType

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
ANOMALY = SignalType.ANOMALY.value


def _returns(values: list[float]) -> pd.Series:
    index = pd.date_range(
        "2026-09-18 14:00", periods=len(values), freq="5min", tz="UTC"
    )
    return pd.Series(values, index=index, name="log_return")


def _predict(
    settings: PipelineSettings,
    asset: AssetConfig,
    reference: pd.Timestamp,
    is_anomaly: bool = True,
) -> int:
    prediction_id = database.insert_prediction(
        settings.paths.database_path,
        asset.symbol,
        ANOMALY,
        "v1",
        reference.to_pydatetime(),
        settings.anomaly.reversion_horizon_bars,
        {"is_anomaly": is_anomaly},
        NOW,
    )
    assert prediction_id is not None
    return prediction_id


def _evaluated(db_path: Path, prediction_id: int) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(
            conn.execute(
                "SELECT evaluated FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()[0]
        )


def test_tie_with_complete_horizon_is_superseded_not_left_pending(
    test_settings: PipelineSettings, test_asset: AssetConfig
) -> None:
    db_path = test_settings.paths.database_path
    database.init_db(db_path)
    horizon = test_settings.anomaly.reversion_horizon_bars
    # Barra 0 con retorno 0 (cotización rancia): el signo no está definido.
    log_return = _returns([0.0] + [0.001, -0.002] * horizon)
    tie = _predict(test_settings, test_asset, log_return.index[0])
    normal = _predict(test_settings, test_asset, log_return.index[1])
    # Última barra: su horizonte aún no transcurre, debe seguir pendiente.
    open_ = _predict(test_settings, test_asset, log_return.index[-1])
    pending = database.fetch_pending_predictions(
        db_path, test_asset.symbol, ANOMALY
    )

    closed = _close_anomaly_outcomes(test_settings, pending, log_return, NOW)

    assert closed == 1
    assert _evaluated(db_path, normal) == 1
    assert _evaluated(db_path, tie) == 2
    assert _evaluated(db_path, open_) == 0


def test_prediction_older_than_the_data_window_is_superseded(
    test_settings: PipelineSettings, test_asset: AssetConfig
) -> None:
    db_path = test_settings.paths.database_path
    database.init_db(db_path)
    log_return = _returns([0.001, -0.002] * 20)
    lost = _predict(
        test_settings,
        test_asset,
        log_return.index[0] - pd.Timedelta(days=70),
    )
    pending = database.fetch_pending_predictions(
        db_path, test_asset.symbol, ANOMALY
    )

    _close_anomaly_outcomes(test_settings, pending, log_return, NOW)

    assert _evaluated(db_path, lost) == 2


def test_live_anomaly_run_records_f1_against_baseline(
    test_settings: PipelineSettings, test_asset: AssetConfig
) -> None:
    db_path = test_settings.paths.database_path
    database.init_db(db_path)
    rng = np.random.default_rng(7)
    log_return = _returns(list(rng.normal(0.0, 0.001, 200)))
    for position, reference in enumerate(log_return.index):
        _predict(test_settings, test_asset, reference, position % 4 == 0)
    pending = database.fetch_pending_predictions(
        db_path, test_asset.symbol, ANOMALY
    )
    closed = _close_anomaly_outcomes(test_settings, pending, log_return, NOW)

    recent = _record_anomaly_run(test_asset, test_settings, "v1", NOW)

    [run] = database.fetch_recent_evaluation_runs(
        db_path, test_asset.symbol, ANOMALY, "v1", 5
    )
    assert len(recent) == closed == run["n_outcomes"]
    assert run["metric_name"] == "f1"
    assert 0.0 <= run["model_metric"] <= 1.0
    assert 0.0 <= run["baseline_metric"] <= 1.0
    # fetch_recent_evaluation_runs solo devuelve corridas suficientes.

"""Cierre de predicciones de anomalía: ninguna queda pendiente para siempre."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import PipelineSettings
from src.persistence import database
from src.pipeline.evaluate_and_retrain import _close_anomaly_outcomes
from src.schemas import SignalType

NOW = datetime(2026, 10, 26, tzinfo=UTC)


def _status(settings: PipelineSettings) -> dict[str, int]:
    with sqlite3.connect(settings.paths.database_path) as conn:
        rows = conn.execute(
            "SELECT reference_timestamp, evaluated FROM predictions"
        ).fetchall()
    return {pd.Timestamp(ref).isoformat(): flag for ref, flag in rows}


def test_unlabelable_bars_are_superseded_not_left_pending(
    test_settings: PipelineSettings,
) -> None:
    horizon = test_settings.anomaly.reversion_horizon_bars
    n = horizon + 40
    index = pd.date_range("2026-10-20 13:00", periods=n, freq="5min", tz="UTC")
    values = np.random.default_rng(7).normal(0, 1e-3, n)
    values[5] = 0.0  # empate: retorno 0, nunca tendrá etiqueta
    log_return = pd.Series(values, index=index, name="log_return")
    out_of_window = index[0] - pd.Timedelta(days=90)
    refs = {
        "tie": index[5],
        "labeled": index[6],
        "open_horizon": index[-2],
        "out_of_window": out_of_window,
    }
    db = test_settings.paths.database_path
    database.init_db(db)
    for ref in refs.values():
        database.insert_prediction(
            db_path=db,
            asset_symbol="^TEST",
            signal_type=SignalType.ANOMALY.value,
            model_version="v1",
            reference_timestamp=ref.to_pydatetime(),
            horizon_bars=horizon,
            payload={"is_anomaly": True},
            run_ts=NOW,
        )
    pending = database.fetch_pending_predictions(
        db, "^TEST", SignalType.ANOMALY.value
    )

    closed = _close_anomaly_outcomes(test_settings, pending, log_return, NOW)

    status = _status(test_settings)
    assert closed == 1
    assert status[refs["labeled"].isoformat()] == 1
    assert status[refs["tie"].isoformat()] == 2
    assert status[refs["out_of_window"].isoformat()] == 2
    assert status[refs["open_horizon"].isoformat()] == 0

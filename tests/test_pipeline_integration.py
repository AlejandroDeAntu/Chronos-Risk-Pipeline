"""Integración: predicción -> resultado real -> métricas -> reentrenamiento.

Todo con datos sintéticos conocidos (sin red), de modo que el resultado
esperado de cada comparación predicción-vs-real se puede calcular a mano.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import AssetConfig, PipelineSettings, RetrainTriggerConfig
from src.evaluation.metrics import qlike_loss
from src.features.returns import compute_log_returns
from src.persistence import database, registry
from src.pipeline.evaluate_and_retrain import (
    evaluate_anomaly_for_asset,
    evaluate_garch_for_asset,
)
from src.pipeline.predict import run_predictions
from src.schemas import SignalType
from tests.helpers import make_settings

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
pytestmark = pytest.mark.usefixtures("mock_yfinance")


def _count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT COUNT(*) FROM {table}"
        return int(conn.execute(query).fetchone()[0])


def _outcome_rows(db_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT p.reference_timestamp, o.* FROM outcomes o "
            "JOIN predictions p ON p.id = o.prediction_id"
        ).fetchall()
    return [dict(row) for row in rows]


def _bootstrap(settings: PipelineSettings, signal: SignalType) -> None:
    database.init_db(settings.paths.database_path)
    [result] = run_predictions(settings, signal, NOW)
    assert result.succeeded, result.detail


def _seed_degraded_runs(settings: PipelineSettings, symbol: str) -> None:
    detector = registry.load_active_anomaly_model(
        settings.paths.database_path, symbol
    )
    assert detector is not None
    for minutes in range(3):
        database.insert_evaluation_run(
            settings.paths.database_path,
            symbol,
            SignalType.ANOMALY.value,
            detector.version,
            NOW - timedelta(minutes=10 - minutes),
            500,
            "f1",
            0.05,
            0.30,
            True,
            {},
        )


def test_cold_start_predicts_both_signals(
    test_settings: PipelineSettings,
) -> None:
    database.init_db(test_settings.paths.database_path)
    for signal in SignalType:
        [result] = run_predictions(test_settings, signal, NOW)
        assert result.succeeded, result.detail
        assert result.cold_start is True
        assert result.prediction_id is not None

    assert _count(test_settings.paths.database_path, "predictions") == 2
    assert _count(test_settings.paths.database_path, "model_registry") == 2


def test_rerun_on_same_bar_does_not_duplicate(
    test_settings: PipelineSettings,
) -> None:
    _bootstrap(test_settings, SignalType.GARCH_VOLATILITY_FORECAST)
    [again] = run_predictions(
        test_settings, SignalType.GARCH_VOLATILITY_FORECAST, NOW
    )
    assert again.prediction_id is None
    assert again.cold_start is False
    assert _count(test_settings.paths.database_path, "predictions") == 1


def test_anomaly_prediction_is_compared_against_real_outcome(
    test_settings: PipelineSettings,
    test_asset: AssetConfig,
    synthetic_intraday_close: pd.Series,
) -> None:
    _bootstrap(test_settings, SignalType.ANOMALY)
    returns = compute_log_returns(synthetic_intraday_close)
    reference = returns.index[1000]
    horizon = test_settings.anomaly.reversion_horizon_bars
    future_sum = returns.iloc[1001 : 1001 + horizon].sum()
    expected_reverted = bool(
        np.sign(future_sum) != np.sign(returns.iloc[1000])
    )

    database.insert_prediction(
        test_settings.paths.database_path,
        test_asset.symbol,
        SignalType.ANOMALY.value,
        "manual",
        reference.to_pydatetime(),
        horizon,
        {"is_anomaly": True},
        NOW,
    )
    result = evaluate_anomaly_for_asset(test_asset, test_settings, NOW)

    assert result.outcomes_closed == 1
    [row] = [
        r
        for r in _outcome_rows(test_settings.paths.database_path)
        if r["reference_timestamp"] == reference.isoformat()
    ]
    assert row["realized_value"] == float(expected_reverted)
    assert row["error_metric"] == float(not expected_reverted)


def test_garch_forecast_is_scored_with_qlike(
    test_settings: PipelineSettings,
    test_asset: AssetConfig,
    synthetic_daily_prices: pd.Series,
) -> None:
    _bootstrap(test_settings, SignalType.GARCH_VOLATILITY_FORECAST)
    returns_pct = compute_log_returns(synthetic_daily_prices) * 100.0
    origin = returns_pct.index[-10]
    active = registry.load_active_garch_model(
        test_settings.paths.database_path, test_asset.symbol
    )
    assert active is not None
    database.insert_prediction(
        test_settings.paths.database_path,
        test_asset.symbol,
        SignalType.GARCH_VOLATILITY_FORECAST.value,
        active.version,
        origin.to_pydatetime(),
        1,
        {"predicted_volatility_pct": 1.3},
        NOW,
    )

    result = evaluate_garch_for_asset(test_asset, test_settings, NOW)

    assert result.outcomes_closed == 1
    realized = float(returns_pct.iloc[-9])
    expected = qlike_loss(np.array([1.3]), np.array([realized]))[0]
    [row] = _outcome_rows(test_settings.paths.database_path)
    assert row["error_metric"] == pytest.approx(expected)
    assert row["realized_value"] == pytest.approx(realized)
    assert _count(test_settings.paths.database_path, "evaluation_runs") == 1


def test_sustained_degradation_forces_retrain(
    tmp_path: Path, test_asset: AssetConfig
) -> None:
    settings = make_settings(
        tmp_path, test_asset, RetrainTriggerConfig(retrain_cooldown_hours=0)
    )
    _bootstrap(settings, SignalType.ANOMALY)
    _seed_degraded_runs(settings, test_asset.symbol)

    result = evaluate_anomaly_for_asset(test_asset, settings, NOW)

    assert result.retrain_decision is not None
    assert result.retrain_decision.triggered_by == "degradation"
    assert result.retrain_promoted is not None
    assert _count(settings.paths.database_path, "model_registry") == 2


def test_cooldown_blocks_repeated_retrain(
    test_settings: PipelineSettings, test_asset: AssetConfig
) -> None:
    _bootstrap(test_settings, SignalType.ANOMALY)
    _seed_degraded_runs(test_settings, test_asset.symbol)

    result = evaluate_anomaly_for_asset(test_asset, test_settings, NOW)

    assert result.retrain_decision is not None
    assert result.retrain_decision.triggered_by == "cooldown"
    assert result.retrain_promoted is None
    assert _count(test_settings.paths.database_path, "model_registry") == 1


def test_fixed_cadence_forces_garch_retrain(
    tmp_path: Path, test_asset: AssetConfig
) -> None:
    settings = make_settings(
        tmp_path, test_asset, RetrainTriggerConfig(retrain_cooldown_hours=0)
    )
    _bootstrap(settings, SignalType.GARCH_VOLATILITY_FORECAST)
    db_path = settings.paths.database_path
    active = registry.load_active_garch_model(db_path, test_asset.symbol)
    assert active is not None
    stale = dataclasses.replace(
        active, version="stale", trained_at=NOW - timedelta(days=8)
    )
    registry.register_candidate(
        db_path,
        test_asset.symbol,
        SignalType.GARCH_VOLATILITY_FORECAST,
        stale,
        {},
    )
    registry.promote_candidate(
        db_path,
        test_asset.symbol,
        SignalType.GARCH_VOLATILITY_FORECAST,
        "stale",
    )

    result = evaluate_garch_for_asset(test_asset, settings, NOW)

    assert result.retrain_decision is not None
    assert result.retrain_decision.triggered_by == "fixed_cadence"
    assert result.retrain_promoted is not None
    assert _count(db_path, "model_registry") == 3
    with sqlite3.connect(db_path) as conn:
        latest = conn.execute(
            "SELECT validation_metrics_json FROM model_registry "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    metrics = json.loads(latest)
    assert metrics["walk_forward_folds"] >= 1
    assert "baseline_qlike" in metrics["pooled_report"]

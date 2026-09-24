"""Pruebas del disparador: enfriamiento, cadencia, degradación y drift."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.config import RetrainTriggerConfig
from src.evaluation.feedback_loop import (
    check_drift,
    check_metric_degradation,
    decide_retrain,
)
from src.schemas import SignalType

NOW = datetime(2026, 9, 18, 21, 0, tzinfo=UTC)
HEALTHY = np.array([0.40, 0.40, 0.40])
BASELINE = np.array([0.20, 0.20, 0.20])


def _decide(
    config: RetrainTriggerConfig,
    trained_days_ago: float,
    registered_hours_ago: float | None,
    model_runs: np.ndarray = HEALTHY,
    baseline_runs: np.ndarray = BASELINE,
    loss_stream: np.ndarray | None = None,
) -> str:
    decision = decide_retrain(
        asset_symbol="TEST",
        signal_type=SignalType.ANOMALY,
        last_trained_at=NOW - timedelta(days=trained_days_ago),
        last_registration_at=(
            None
            if registered_hours_ago is None
            else NOW - timedelta(hours=registered_hours_ago)
        ),
        run_model_metrics=model_runs,
        run_baseline_metrics=baseline_runs,
        higher_is_better=True,
        loss_stream=np.zeros(5) if loss_stream is None else loss_stream,
        config=config,
        now=NOW,
    )
    return decision.triggered_by


def test_nothing_triggers_for_healthy_recent_model() -> None:
    assert _decide(RetrainTriggerConfig(), 2, 48) == "none"


def test_fixed_cadence_triggers_after_configured_days() -> None:
    assert _decide(RetrainTriggerConfig(), 31, 31 * 24) == "fixed_cadence"


def test_cooldown_suppresses_every_other_trigger() -> None:
    degraded = np.array([0.1, 0.1, 0.1])
    assert (
        _decide(RetrainTriggerConfig(), 31, 2, model_runs=degraded)
        == "cooldown"
    )


def test_sustained_degradation_triggers() -> None:
    degraded = np.array([0.10, 0.12, 0.08])
    assert _decide(RetrainTriggerConfig(), 2, 48, degraded) == "degradation"


def test_single_bad_run_does_not_trigger() -> None:
    mixed = np.array([0.10, 0.40, 0.40])
    result = check_metric_degradation(mixed, BASELINE, True, 3)
    assert result.is_degraded is False


def test_degradation_requires_enough_runs() -> None:
    result = check_metric_degradation(
        np.array([0.0]), np.array([1.0]), True, 3
    )
    assert result.is_degraded is False


def test_degradation_direction_for_losses() -> None:
    worse_loss = np.array([2.0, 2.1, 2.2])
    result = check_metric_degradation(worse_loss, np.ones(3), False, 3)
    assert result.is_degraded is True


def test_degradation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="coincidir"):
        check_metric_degradation(np.ones(2), np.ones(3), True, 1)


def test_drift_detected_on_mean_shift(rng: np.random.Generator) -> None:
    stream = np.concatenate(
        [rng.normal(0.0, 1.0, 300), rng.normal(8.0, 1.0, 300)]
    )
    assert check_drift(stream) is True
    shifted = _decide(RetrainTriggerConfig(), 2, 48, loss_stream=stream)
    assert shifted == "drift"


def test_no_drift_on_stable_stream(rng: np.random.Generator) -> None:
    assert check_drift(rng.normal(0.0, 1.0, 300)) is False

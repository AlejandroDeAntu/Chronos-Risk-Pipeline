"""Pruebas de métricas: nunca solo accuracy, siempre con baseline."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    evaluate_anomaly_classifier,
    evaluate_volatility_forecast,
    qlike_loss,
)


def test_perfect_classifier_beats_anti_correlated_baseline() -> None:
    y_true = np.array([True, False] * 40)
    y_baseline = ~y_true

    report = evaluate_anomaly_classifier(y_true, y_true.copy(), y_baseline)

    assert report.precision == pytest.approx(1.0)
    assert report.recall == pytest.approx(1.0)
    assert report.f1 == pytest.approx(1.0)
    assert report.confusion_matrix == ((40, 0), (0, 40))
    assert report.beats_baseline_f1 is True
    assert report.sample_size_warning is None


def test_classifier_warns_when_few_predicted_positives() -> None:
    y_true = np.array([True, False, True, False])
    y_pred = np.array([True, False, False, False])
    report = evaluate_anomaly_classifier(y_true, y_pred, y_pred.copy())
    assert report.sample_size_warning is not None


def test_classifier_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="misma longitud"):
        evaluate_anomaly_classifier(
            np.array([True]), np.array([True, False]), np.array([True])
        )


def test_qlike_is_minimized_by_true_volatility(
    rng: np.random.Generator,
) -> None:
    returns = rng.normal(0.0, 2.0, 20_000)
    true_sigma = np.full_like(returns, 2.0)
    assert qlike_loss(true_sigma, returns).mean() < (
        qlike_loss(true_sigma * 1.5, returns).mean()
    )
    assert qlike_loss(true_sigma, returns).mean() < (
        qlike_loss(true_sigma * 0.6, returns).mean()
    )


def test_volatility_report_prefers_calibrated_forecast(
    rng: np.random.Generator,
) -> None:
    sigma = np.where(np.arange(500) < 250, 0.5, 2.0)
    returns = rng.normal(0.0, sigma)
    constant = np.full_like(sigma, sigma.mean())

    report = evaluate_volatility_forecast(sigma, returns, constant)

    assert report.beats_baseline_qlike is True
    assert report.mean_squared_standardized_residual == pytest.approx(
        1.0, abs=0.2
    )
    assert report.residual_sq_ljung_box_pvalue is not None


def test_qlike_rejects_non_positive_volatility() -> None:
    with pytest.raises(ValueError, match="> 0"):
        qlike_loss(np.array([0.0, 1.0]), np.array([0.1, 0.2]))

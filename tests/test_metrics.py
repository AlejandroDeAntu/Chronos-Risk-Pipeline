"""Pruebas de métricas: nunca solo accuracy, siempre con baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    diebold_mariano,
    evaluate_anomaly_classifier,
    evaluate_volatility_forecast,
    precision_lift_test,
    qlike_loss,
)
from src.exceptions import InsufficientHistoryError
from src.models.baseline import ewma_volatility_baseline


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


def test_precision_lift_detects_signal_above_prevalence() -> None:
    # Prevalencia 50%; el detector acierta 80 de 100 marcas.
    y_true = np.array([True] * 500 + [False] * 500)
    y_pred = np.zeros(1000, dtype=bool)
    y_pred[:80] = True
    y_pred[500:520] = True
    report = evaluate_anomaly_classifier(y_true, y_pred, y_pred.copy())

    lift = precision_lift_test(report)

    assert lift.prevalence == pytest.approx(0.5)
    assert lift.lift == pytest.approx(1.6)
    assert lift.p_value < 1e-6


def test_precision_lift_is_not_significant_at_chance() -> None:
    y_true = np.array([True, False] * 500)
    y_pred = np.zeros(1000, dtype=bool)
    y_pred[:100] = True  # 50 TP / 50 FP = exactamente la prevalencia
    report = evaluate_anomaly_classifier(y_true, y_pred, y_pred.copy())

    assert precision_lift_test(report).p_value > 0.4


def test_diebold_mariano_prefers_lower_loss_model() -> None:
    rng = np.random.default_rng(1)
    reference = rng.gamma(2.0, 1.0, size=500)
    model = reference - 0.2 + rng.normal(0.0, 0.1, size=500)

    result = diebold_mariano(model, reference)

    assert result.mean_loss_diff < 0.0
    assert result.p_value_one_sided < 0.001


def test_diebold_mariano_rejects_small_samples() -> None:
    with pytest.raises(InsufficientHistoryError, match="requiere"):
        diebold_mariano(np.ones(10), np.ones(10))


def test_ewma_baseline_is_causal_and_matches_recursion() -> None:
    returns = pd.Series([1.0, -2.0, 0.5, 3.0])

    result = ewma_volatility_baseline(returns, decay=0.9)

    s2 = [1.0]  # ewm(adjust=False) arranca en r_0^2
    for r in returns.iloc[1:]:
        s2.append(0.9 * s2[-1] + 0.1 * r**2)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].to_numpy() == pytest.approx(np.sqrt(s2[:-1]))


def test_diebold_mariano_handles_constant_differential() -> None:
    reference = np.linspace(1.0, 2.0, 100)
    same = diebold_mariano(reference, reference)
    better = diebold_mariano(reference - 0.1, reference)

    assert same.t_stat == 0.0
    assert same.p_value_one_sided == pytest.approx(0.5)
    assert better.p_value_one_sided == pytest.approx(0.0)

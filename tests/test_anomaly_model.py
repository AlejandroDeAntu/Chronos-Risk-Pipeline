"""Pruebas del detector: calibración, serialización y etiquetado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exceptions import InsufficientHistoryError, ModelFitError
from src.models.anomaly_model import AnomalyDetector, label_reversion_outcomes


def test_fit_calibrates_threshold_close_to_target_rate(
    rng: np.random.Generator,
) -> None:
    index = pd.date_range("2026-01-01", periods=2000, freq="5min")
    returns = pd.Series(rng.normal(0.0, 0.001, 2000), index=index)

    detector = AnomalyDetector.fit(
        log_return=returns,
        rolling_window=20,
        target_anomaly_rate=0.05,
        reversion_horizon_bars=5,
        min_bars_required=100,
    )
    bands = detector.predict(returns, min_bars_required=100)

    empirical_rate = bands["is_anomaly"].mean()
    assert empirical_rate == pytest.approx(0.05, abs=0.02)


def test_fit_rejects_invalid_target_rate() -> None:
    index = pd.date_range("2026-01-01", periods=200, freq="5min")
    returns = pd.Series(np.zeros(200) + 0.0001, index=index)
    with pytest.raises(ModelFitError):
        AnomalyDetector.fit(
            log_return=returns,
            rolling_window=20,
            target_anomaly_rate=0.9,
            reversion_horizon_bars=5,
            min_bars_required=50,
        )


def test_metadata_roundtrip_preserves_all_fields(
    rng: np.random.Generator,
) -> None:
    index = pd.date_range("2026-01-01", periods=500, freq="5min")
    returns = pd.Series(rng.normal(0.0, 0.001, 500), index=index)
    detector = AnomalyDetector.fit(
        log_return=returns,
        rolling_window=20,
        target_anomaly_rate=0.05,
        reversion_horizon_bars=5,
        min_bars_required=100,
    )

    restored = AnomalyDetector.from_metadata(detector.to_metadata())
    assert restored == detector


def test_label_reversion_outcomes_detects_opposite_sign_move() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="D")
    bands = pd.DataFrame(
        {
            "log_return": [0.0, 0.0, 0.05, -0.01, -0.01, -0.01],
            "rolling_mean": [0.0] * 6,
            "rolling_std": [0.01] * 6,
            "zscore": [0.0] * 6,
            "is_anomaly": [False, False, True, False, False, False],
        },
        index=index,
    )

    labeled = label_reversion_outcomes(bands, horizon_bars=2)

    # La barra 2 (retorno +0.05, anómala) es seguida por dos retornos negativos
    # (-0.01, -0.01): signo opuesto -> se etiqueta como reversión.
    anomaly_row = labeled.loc[index[2]]
    assert bool(anomaly_row["reverted"]) is True


def test_label_reversion_outcomes_raises_without_future_horizon() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    bands = pd.DataFrame(
        {
            "log_return": [0.01, 0.02, 0.03],
            "rolling_mean": [0.0] * 3,
            "rolling_std": [0.01] * 3,
            "zscore": [0.0] * 3,
            "is_anomaly": [False, False, False],
        },
        index=index,
    )
    with pytest.raises(InsufficientHistoryError):
        label_reversion_outcomes(bands, horizon_bars=10)

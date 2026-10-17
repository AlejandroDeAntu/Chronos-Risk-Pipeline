"""Perfil de volatilidad por hora y detector desestacionalizado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.seasonality import deseasonalize, hourly_profile
from src.models.anomaly_model import AnomalyDetector

PEAK_HOUR = 14


def _seasonal_returns(n_days: int = 40) -> pd.Series:
    """Retornos de 5 min, 24 h, con volatilidad x4 a las 14:00 UTC."""
    rng = np.random.default_rng(11)
    index = pd.date_range(
        "2026-01-05", periods=n_days * 288, freq="5min", tz="UTC"
    )
    scale = np.where(index.hour == PEAK_HOUR, 4.0, 1.0) * 1e-4
    return pd.Series(rng.normal(0.0, scale), index=index, name="log_return")


def test_hourly_profile_detects_peak_hour() -> None:
    profile = hourly_profile(_seasonal_returns())

    assert len(profile) == 24
    assert profile.idxmax() == PEAK_HOUR
    assert profile[PEAK_HOUR] == pytest.approx(4.0, rel=0.15)
    assert profile.drop(PEAK_HOUR).between(0.85, 1.15).all()


def test_hours_without_data_fall_back_to_one() -> None:
    returns = _seasonal_returns()
    only_day = returns[returns.index.hour < 12]
    profile = hourly_profile(only_day)
    assert (profile.loc[12:] == 1.0).all()


def test_deseasonalize_removes_hourly_scale() -> None:
    returns = _seasonal_returns()
    adjusted = deseasonalize(returns, hourly_profile(returns))
    by_hour = adjusted.abs().groupby(adjusted.index.hour).median()
    assert by_hour.max() / by_hour.min() < 1.3


def _anomaly_share_at_peak(
    detector: AnomalyDetector, returns: pd.Series
) -> float:
    bands = detector.predict(returns, min_bars_required=60)
    flagged = bands[bands["is_anomaly"]]
    return float((flagged.index.hour == PEAK_HOUR).mean())


def test_seasonal_detector_stops_overflagging_the_peak_hour() -> None:
    returns = _seasonal_returns()
    train, test = returns.iloc[: 30 * 288], returns.iloc[30 * 288 :]
    kwargs = {
        "rolling_window": 20,
        "target_anomaly_rate": 0.0455,
        "reversion_horizon_bars": 10,
        "min_bars_required": 60,
    }
    plain = AnomalyDetector.fit(train, seasonal=False, **kwargs)
    seasonal = AnomalyDetector.fit(train, seasonal=True, **kwargs)

    # 1 de 24 horas: sin sesgo horario, ~4% de las anomalías caería ahí.
    assert _anomaly_share_at_peak(plain, test) > 0.15
    assert _anomaly_share_at_peak(seasonal, test) < 0.10


def test_profile_comes_only_from_training_data() -> None:
    returns = _seasonal_returns()
    train = returns.iloc[: 30 * 288]
    detector = AnomalyDetector.fit(
        train,
        rolling_window=20,
        target_anomaly_rate=0.0455,
        reversion_horizon_bars=10,
        min_bars_required=60,
    )
    np.testing.assert_allclose(
        detector.seasonal_profile, hourly_profile(train).to_numpy()
    )


def test_legacy_metadata_loads_with_flat_profile() -> None:
    legacy = {
        "version": "v",
        "rolling_window": 20,
        "zscore_threshold": 2.1,
        "target_anomaly_rate": 0.0455,
        "reversion_horizon_bars": 10,
        "trained_at": "2026-09-24T21:51:03+00:00",
        "train_rows": 4000,
    }
    detector = AnomalyDetector.from_metadata(legacy)
    assert detector.seasonal_profile == (1.0,) * 24

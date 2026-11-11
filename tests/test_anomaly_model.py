"""Pruebas del detector: calibración, serialización y etiquetado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import AssetConfig, PipelineSettings
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


def test_label_reversion_outcomes_drops_zero_return_ties() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    bands = pd.DataFrame(
        {
            # Barra 1: retorno 0 (empate). Barra 2: futuro acumulado 0.
            "log_return": [0.01, 0.0, 0.02, 0.01, -0.01],
            "rolling_mean": [0.0] * 5,
            "rolling_std": [0.01] * 5,
            "zscore": [0.0] * 5,
            "is_anomaly": [False] * 5,
        },
        index=index,
    )

    labeled = label_reversion_outcomes(bands, horizon_bars=2)

    # Antes: ambas barras se etiquetaban como "reversión" por np.sign(0).
    assert index[1] not in labeled.index
    assert index[2] not in labeled.index
    assert index[0] in labeled.index


def test_walk_forward_skips_fold_when_detector_fit_fails(
    synthetic_intraday_close: pd.Series,
    test_settings: PipelineSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.features.returns import compute_log_returns
    from src.pipeline import retrain

    original = retrain.AnomalyDetector.fit

    def flaky(*args: object, **kwargs: object) -> AnomalyDetector:
        if kwargs.get("version") == "walk_forward_fold_0":
            raise ModelFitError("umbral no finito")
        return original(*args, **kwargs)

    monkeypatch.setattr(retrain.AnomalyDetector, "fit", flaky)
    result = retrain.walk_forward_anomaly(
        compute_log_returns(synthetic_intraday_close), test_settings
    )

    assert result.n_folds == 3


def test_promotion_alpha_is_split_across_assets(
    test_settings: PipelineSettings,
    test_asset: AssetConfig,
    synthetic_intraday_close: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.evaluation.metrics import PrecisionLiftResult
    from src.persistence import database
    from src.pipeline import retrain

    database.init_db(test_settings.paths.database_path)
    log_return = np.log(synthetic_intraday_close).diff().dropna()
    # Un lift con p = 0.02 pasa con un activo (alfa 0.05) pero no cuando la
    # misma prueba se repite en 7 activos (alfa 0.05 / 7 = 0.0071).
    monkeypatch.setattr(
        retrain,
        "independent_precision_lift_test",
        lambda labeled, min_gap: PrecisionLiftResult(0.6, 0.5, 1.2, 0.02, 50),
    )
    others = tuple(
        test_asset.model_copy(update={"symbol": f"^T{i}"}) for i in range(6)
    )
    seven = test_settings.model_copy(update={"assets": (test_asset, *others)})

    alone = retrain.retrain_anomaly_model(
        test_asset, test_settings, log_return
    )
    among_seven = retrain.retrain_anomaly_model(test_asset, seven, log_return)

    assert alone.promoted is True
    assert among_seven.promoted is False
    assert among_seven.validation_metrics["promotion_alpha"] == pytest.approx(
        0.05 / 7
    )

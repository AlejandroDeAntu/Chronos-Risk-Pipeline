"""Modelos de volatilidad: GARCH, GJR, EGARCH (arch) y HAR sobre rango."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exceptions import InsufficientHistoryError, ModelFitError
from src.models.volatility import (
    ConditionalVolModel,
    HarModel,
    fit_volatility_model,
    model_from_metadata,
    volatility_frame,
)
from tests.helpers import synthetic_ohlc

FAMILIES = ("garch", "gjr", "egarch", "har")


@pytest.fixture()
def frame(synthetic_daily_prices: pd.Series) -> pd.DataFrame:
    return volatility_frame(synthetic_ohlc(synthetic_daily_prices))


def test_volatility_frame_has_aligned_columns(frame: pd.DataFrame) -> None:
    assert list(frame.columns) == [
        "return_pct",
        "rv_day",
        "rv_week",
        "rv_month",
    ]
    # Varianzas en %^2, no negativas donde existen.
    assert (frame[["rv_day", "rv_week", "rv_month"]].dropna() >= 0).all().all()
    assert frame["return_pct"].notna().all()


def test_garch_recovers_simulated_persistence(frame: pd.DataFrame) -> None:
    # La fixture simula alpha=0.08, beta=0.90, nu=6.
    model = ConditionalVolModel.fit(frame, "garch", min_history_days=300)
    persistence = model.params["alpha[1]"] + model.params["beta[1]"]
    # Sesgo de muestra finita con 1,500 obs: tolerancia de 0.05.
    assert persistence == pytest.approx(0.98, abs=0.05)
    assert model.params["nu"] > 3.0


@pytest.mark.parametrize("family", FAMILIES)
def test_variance_path_is_causal(frame: pd.DataFrame, family: str) -> None:
    model = fit_volatility_model(frame.iloc[:1000], family, 300)
    shocked = frame.copy()
    shocked.iloc[1100:, :] = shocked.iloc[1100:, :] * 5.0

    base = model.variance_path(frame)
    moved = model.variance_path(shocked)

    # Un shock en t >= 1100 no puede cambiar el pronóstico de t <= 1100.
    np.testing.assert_allclose(
        base.iloc[:1101].dropna(), moved.iloc[:1101].dropna()
    )
    assert not np.allclose(base.iloc[1102:], moved.iloc[1102:])


@pytest.mark.parametrize("family", FAMILIES)
def test_forecast_matches_next_point_of_path(
    frame: pd.DataFrame, family: str
) -> None:
    model = fit_volatility_model(frame.iloc[:1000], family, 300)
    # El pronóstico a 1 paso con datos hasta t es el valor del camino en t+1.
    forecast = model.forecast_variance(frame.iloc[:1200], horizon=1)[0]
    path = model.variance_path(frame.iloc[:1201])
    assert forecast == pytest.approx(path.iloc[-1], rel=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_multi_step_forecast_is_positive(
    frame: pd.DataFrame, family: str
) -> None:
    model = fit_volatility_model(frame.iloc[:1000], family, 300)
    forecast = model.forecast_variance(frame, horizon=5)
    assert forecast.shape == (5,)
    assert np.all(forecast > 0.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_simulation_is_reproducible_and_scaled(
    frame: pd.DataFrame, family: str
) -> None:
    model = fit_volatility_model(frame.iloc[:1000], family, 300)
    first = model.simulate_returns(frame, 5, 4000, seed=7)
    second = model.simulate_returns(frame, 5, 4000, seed=7)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (4000, 5)
    # La dispersión del primer día simulada es coherente con la varianza
    # pronosticada (tolerancia por error de Monte Carlo y colas t).
    expected = np.sqrt(model.forecast_variance(frame, horizon=1)[0])
    assert first[:, 0].std() == pytest.approx(expected, rel=0.15)


@pytest.mark.parametrize("family", FAMILIES)
def test_metadata_roundtrip(frame: pd.DataFrame, family: str) -> None:
    model = fit_volatility_model(frame.iloc[:1000], family, 300)
    assert model_from_metadata(model.to_metadata()) == model


def test_legacy_garch_metadata_is_still_loadable(frame: pd.DataFrame) -> None:
    # Fila real del registro anterior (pipeline.db, 2026-10-17, ^GSPC).
    legacy = {
        "version": "20261017T001106055402Z",
        "omega": 0.0236,
        "alpha": 0.1351,
        "beta": 0.8452,
        "mu": 0.0791,
        "nu": 6.2,
        "aic": 7012.1,
        "bic": 7043.4,
        "trained_at": "2026-10-17T00:11:06.055596+00:00",
        "train_rows": 4205,
        "train_end": "2026-10-16 00:00:00",
    }

    model = model_from_metadata(legacy)

    assert isinstance(model, ConditionalVolModel)
    assert model.family == "garch"
    assert model.params["alpha[1]"] == pytest.approx(0.1351)
    assert model.forecast_variance(frame, horizon=1)[0] > 0.0


def test_fit_rejects_short_history(frame: pd.DataFrame) -> None:
    for family in FAMILIES:
        with pytest.raises(InsufficientHistoryError):
            fit_volatility_model(frame.iloc[:100], family, 300)


def test_fit_rejects_unknown_family(frame: pd.DataFrame) -> None:
    with pytest.raises(ModelFitError, match="(?i)familia"):
        fit_volatility_model(frame, "figarch", 300)


def test_har_is_calibrated_to_squared_returns(frame: pd.DataFrame) -> None:
    model = HarModel.fit(frame.iloc[:1200], min_history_days=300)
    path = model.variance_path(frame.iloc[:1200]).dropna()
    realized = frame["return_pct"].loc[path.index] ** 2
    # Re-escalado en entrenamiento: media del pronóstico = media de r^2.
    assert path.mean() == pytest.approx(realized.mean(), rel=1e-6)


@pytest.mark.parametrize("family", ["garch", "gjr", "egarch"])
def test_recursion_matches_arch_after_burn_in(
    frame: pd.DataFrame, family: str
) -> None:
    from arch import arch_model

    from src.models.volatility import _ARCH_SPECS

    model = ConditionalVolModel.fit(frame, family, min_history_days=300)
    reference = arch_model(
        frame["return_pct"].to_numpy(),
        mean="Constant",
        dist="t",
        **_ARCH_SPECS[family],
    ).fix(pd.Series(model.params))
    ours = model.variance_path(frame).to_numpy()
    theirs = np.asarray(reference.conditional_volatility) ** 2
    # Solo difiere la inicialización, que decae como beta^t.
    np.testing.assert_allclose(ours[300:], theirs[300:], rtol=1e-4)

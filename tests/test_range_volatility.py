"""Estimadores de volatilidad por rango contra valores calculados a mano."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.range_volatility import (
    garman_klass,
    parkinson,
    realized_variance_proxy,
    rogers_satchell,
    yang_zhang,
)
from tests.helpers import synthetic_ohlc

# Vela de referencia: O=100, H=110, L=90, C=105 con cierre previo 100.
# Valores calculados a mano con logaritmos naturales (ver docstrings).
_PARKINSON = 0.014523873553353111
_GARMAN_KLASS = 0.01921479796164129
_ROGERS_SATCHELL = 0.020675226170266067
# Ventana de 2 velas: la de referencia y (106, 108, 102, 103).
_YANG_ZHANG_2 = 0.010389206103593321


def _bars() -> pd.DataFrame:
    index = pd.date_range("2026-01-05", periods=3, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 106.0],
            "high": [100.0, 110.0, 108.0],
            "low": [100.0, 90.0, 102.0],
            "close": [100.0, 105.0, 103.0],
        },
        index=index,
    )


def test_parkinson_matches_hand_calculation() -> None:
    assert parkinson(_bars()).iloc[1] == pytest.approx(_PARKINSON)


def test_garman_klass_matches_hand_calculation() -> None:
    assert garman_klass(_bars()).iloc[1] == pytest.approx(_GARMAN_KLASS)


def test_rogers_satchell_matches_hand_calculation() -> None:
    assert rogers_satchell(_bars()).iloc[1] == pytest.approx(_ROGERS_SATCHELL)


def test_yang_zhang_matches_hand_calculation() -> None:
    result = yang_zhang(_bars(), window=2)
    assert result.iloc[2] == pytest.approx(_YANG_ZHANG_2)
    # Las dos primeras posiciones no tienen ventana completa con cierre
    # previo.
    assert result.iloc[:2].isna().all()


def test_realized_variance_proxy_adds_overnight_gap() -> None:
    bars = _bars()
    proxy = realized_variance_proxy(bars)
    overnight = np.log(106.0 / 105.0) ** 2
    assert proxy.iloc[2] == pytest.approx(
        overnight + garman_klass(bars).iloc[2]
    )
    assert np.isnan(proxy.iloc[0])  # sin cierre previo


def test_flat_bar_has_zero_variance() -> None:
    bars = _bars()
    for estimator in (parkinson, garman_klass, rogers_satchell):
        assert estimator(bars).iloc[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "estimator", [parkinson, garman_klass, rogers_satchell]
)
def test_estimators_are_scale_invariant(estimator: object) -> None:
    bars = _bars()
    scaled = bars * 37.5
    pd.testing.assert_series_equal(estimator(bars), estimator(scaled))


def test_range_estimators_underestimate_slightly_with_discrete_sampling() -> (
    None
):
    # Movimiento browniano con sigma diaria conocida y 390 observaciones por
    # día. Con monitoreo discreto el high/low observado queda por dentro del
    # verdadero, así que los estimadores por rango subestiman algunos puntos
    # porcentuales. Por eso la selección de modelos usa r^2 (insesgado) y el
    # proxy por rango se re-escala antes de usarse en evaluación.
    rng = np.random.default_rng(7)
    sigma, n_days, steps = 0.01, 4000, 390
    paths = np.cumsum(
        rng.normal(0.0, sigma / np.sqrt(steps), (n_days, steps)), axis=1
    )
    log_open = np.zeros(n_days)
    bars = pd.DataFrame(
        {
            "open": np.exp(log_open),
            "high": np.exp(np.maximum(paths.max(axis=1), 0.0)),
            "low": np.exp(np.minimum(paths.min(axis=1), 0.0)),
            "close": np.exp(paths[:, -1]),
        },
        index=pd.date_range("2000-01-03", periods=n_days, freq="B"),
    )
    for estimator in (parkinson, garman_klass, rogers_satchell):
        ratio = estimator(bars).mean() / sigma**2
        assert 0.85 < ratio < 1.0


def test_yang_zhang_rejects_window_below_two(
    synthetic_daily_prices: pd.Series,
) -> None:
    with pytest.raises(ValueError, match="window"):
        yang_zhang(synthetic_ohlc(synthetic_daily_prices), window=1)

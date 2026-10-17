"""Walk-forward multi-modelo y selección champion-challenger."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PipelineSettings
from src.evaluation.volatility_selection import (
    VolatilityWalkForward,
    holm_adjust,
    select_champion,
    walk_forward_volatility,
)
from src.models.volatility import volatility_frame
from tests.helpers import synthetic_ohlc


@pytest.fixture()
def walk_forward(
    synthetic_daily_prices: pd.Series, test_settings: PipelineSettings
) -> VolatilityWalkForward:
    frame = volatility_frame(synthetic_ohlc(synthetic_daily_prices))
    return walk_forward_volatility(
        frame, test_settings, ("garch", "gjr", "har")
    )


def test_walk_forward_aligns_all_series(
    walk_forward: VolatilityWalkForward,
) -> None:
    n = len(walk_forward.dates)
    assert n > 500
    assert walk_forward.realized_pct.shape == (n,)
    for name in ("sd20", "ewma"):
        assert walk_forward.baselines[name].shape == (n,)
    for family in ("garch", "gjr", "har"):
        assert walk_forward.forecasts[family].shape == (n,)
    assert walk_forward.failed == {}
    # Las fechas de prueba son estrictamente crecientes (sin solapamiento).
    assert pd.Index(walk_forward.dates).is_monotonic_increasing


def test_garch_beats_trailing_baseline_on_garch_data(
    walk_forward: VolatilityWalkForward,
) -> None:
    # Los datos sintéticos se simulan con un GARCH(1,1)-t.
    garch = walk_forward.mean_qlike("garch")
    assert garch < walk_forward.mean_qlike("sd20")


def test_holm_adjust_matches_reference() -> None:
    # Ejemplo de Holm (1979) recalculado a mano: m = 3.
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_champion_is_retained_when_challengers_do_not_win(
    walk_forward: VolatilityWalkForward,
) -> None:
    decision = select_champion(walk_forward, "garch", alpha=0.05)
    # GJR y HAR no pueden ganar con significancia a un GARCH en datos GARCH.
    assert decision.champion == "garch"
    assert decision.switched is False
    assert {row.challenger for row in decision.comparisons} == {"gjr", "har"}


def test_challenger_replaces_clearly_worse_champion() -> None:
    rng = np.random.default_rng(3)
    n = 2000
    sigma = np.exp(rng.normal(0.0, 0.4, n))
    realized = sigma * rng.standard_normal(n)
    wf = VolatilityWalkForward(
        dates=tuple(pd.bdate_range("2020-01-01", periods=n)),
        realized_pct=realized,
        rv_proxy_pct2=sigma**2,
        baselines={"sd20": np.full(n, 1.2), "ewma": np.full(n, 1.2)},
        forecasts={
            "garch": np.full(n, 1.2),  # ignora la volatilidad cambiante
            "har": sigma,  # pronóstico perfecto
        },
        var_z={"garch": np.zeros((n, 2)), "har": np.zeros((n, 2))},
        n_folds={"garch": 4, "har": 4},
        failed={},
    )

    decision = select_champion(wf, "garch", alpha=0.05)

    assert decision.champion == "har"
    assert decision.switched is True
    [row] = decision.comparisons
    assert row.p_value_holm < 0.05


def test_failed_family_is_reported_not_hidden(
    synthetic_daily_prices: pd.Series, test_settings: PipelineSettings
) -> None:
    frame = volatility_frame(synthetic_ohlc(synthetic_daily_prices))
    wf = walk_forward_volatility(frame, test_settings, ("garch", "figarch"))
    assert "figarch" in wf.failed
    assert "figarch" not in wf.forecasts


def test_comparisons_use_the_common_sample() -> None:
    n = 100
    realized = np.ones(n)
    partial = np.full(n, 1.0)
    partial[:50] = np.nan  # la familia falló en el primer bloque
    full = np.r_[np.full(50, 5.0), np.full(50, 1.0)]  # mala solo al inicio
    wf = VolatilityWalkForward(
        dates=tuple(pd.bdate_range("2020-01-01", periods=n)),
        realized_pct=realized,
        rv_proxy_pct2=np.ones(n),
        baselines={"sd20": np.ones(n), "ewma": np.ones(n)},
        forecasts={"garch": full, "egarch": partial},
        var_z={"garch": np.zeros((n, 2)), "egarch": np.zeros((n, 2))},
        n_folds={"garch": 2, "egarch": 1},
        failed={},
    )

    assert wf.common_mask.sum() == 50
    # En la muestra común ambas pronostican 1.0: misma pérdida media.
    assert wf.mean_qlike("garch") == pytest.approx(wf.mean_qlike("egarch"))

"""Fixtures compartidas: nunca usan la red, siempre mockean yfinance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    AssetClass,
    AssetConfig,
    PipelineSettings,
    RetrainTriggerConfig,
)
from src.data import fetch
from tests.helpers import as_yfinance_frame, make_settings


@pytest.fixture()
def rng() -> np.random.Generator:
    """Generador con semilla fija para pruebas reproducibles."""
    return np.random.default_rng(seed=42)


@pytest.fixture()
def synthetic_daily_prices(rng: np.random.Generator) -> pd.Series:
    """1500 cierres diarios simulados desde un GARCH(1,1)-t real.

    La recursión GARCH es secuencial, por eso se simula con un bucle sobre
    un arreglo NumPy (no sobre filas de un DataFrame).
    """
    n_days = 1500
    omega, alpha, beta, dof = 0.05, 0.08, 0.90, 6.0
    shocks = rng.standard_t(dof, n_days) / np.sqrt(dof / (dof - 2.0))
    returns_pct = np.empty(n_days)
    variance = omega / (1.0 - alpha - beta)
    for t in range(n_days):
        returns_pct[t] = 0.03 + np.sqrt(variance) * shocks[t]
        variance = (
            omega + alpha * (returns_pct[t] - 0.03) ** 2 + beta * variance
        )
    prices = 100.0 * np.exp(np.cumsum(returns_pct / 100.0))
    index = pd.bdate_range(end="2026-09-18", periods=n_days)
    return pd.Series(prices, index=index, name="close")


@pytest.fixture()
def synthetic_intraday_close(rng: np.random.Generator) -> pd.Series:
    """2000 cierres de 5 minutos consecutivos, índice UTC, ya cerrados."""
    n_bars = 2000
    log_returns = rng.normal(loc=0.0, scale=0.0008, size=n_bars)
    index = pd.date_range(
        end=pd.Timestamp("2026-09-18 20:00", tz="UTC"),
        periods=n_bars,
        freq="5min",
    )
    return pd.Series(
        4000.0 * np.exp(np.cumsum(log_returns)), index=index, name="close"
    )


@pytest.fixture()
def test_asset() -> AssetConfig:
    """Activo ficticio para las pruebas."""
    return AssetConfig(
        symbol="^TEST",
        display_name="Test Index",
        asset_class=AssetClass.EQUITY_INDEX,
    )


@pytest.fixture()
def test_settings(tmp_path: Path, test_asset: AssetConfig) -> PipelineSettings:
    """Configuración por defecto con un solo activo y rutas temporales."""
    return make_settings(tmp_path, test_asset, RetrainTriggerConfig())


@pytest.fixture()
def mock_yfinance(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_daily_prices: pd.Series,
    synthetic_intraday_close: pd.Series,
) -> None:
    """Reemplaza la descarga: diario si ``interval='1d'``, si no intradía."""

    def _fake_download(symbol: str, **kwargs: str) -> pd.DataFrame:
        if kwargs.get("interval") == "1d":
            return as_yfinance_frame(synthetic_daily_prices, symbol)
        return as_yfinance_frame(synthetic_intraday_close, symbol)

    monkeypatch.setattr(fetch, "_download_raw", _fake_download)

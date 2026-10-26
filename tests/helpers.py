"""Utilidades compartidas por las pruebas (no son fixtures)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.config import (
    AssetConfig,
    PathsConfig,
    PipelineSettings,
    RetrainTriggerConfig,
)

YFinanceFactory = Callable[[pd.Series, str], pd.DataFrame]


def synthetic_ohlc(close: pd.Series, spread: float = 0.002) -> pd.DataFrame:
    """OHLC consistente derivado de cierres: open = cierre previo.

    ``high``/``low`` envuelven a open y close con un margen relativo fijo, de
    modo que los estimadores por rango tengan valores conocidos.
    """
    open_ = close.shift(1).fillna(close.iloc[0])
    upper = pd.concat([open_, close], axis=1).max(axis=1)
    lower = pd.concat([open_, close], axis=1).min(axis=1)
    return pd.DataFrame(
        {
            "open": open_,
            "high": upper * (1.0 + spread),
            "low": lower * (1.0 - spread),
            "close": close,
        }
    )


def as_yfinance_frame(close: pd.Series, ticker: str) -> pd.DataFrame:
    """Da a cierres la forma MultiIndex (campo, ticker) OHLC de yfinance."""
    ohlc = synthetic_ohlc(close)
    fields = ["Open", "High", "Low", "Close"]
    columns = pd.MultiIndex.from_product([fields, [ticker]])
    return pd.DataFrame(
        ohlc[["open", "high", "low", "close"]].to_numpy(),
        index=close.index,
        columns=columns,
    )


def make_settings(
    tmp_path: Path, asset: AssetConfig, retrain: RetrainTriggerConfig
) -> PipelineSettings:
    """Configuración con rutas temporales (base y reportes aislados)."""
    paths = PathsConfig(
        database_path=tmp_path / "data" / "pipeline.db",
        reports_dir=tmp_path / "reports",
        weekly_history_path=tmp_path / "data" / "weekly.jsonl",
    )
    return PipelineSettings(assets=(asset,), paths=paths, retrain=retrain)

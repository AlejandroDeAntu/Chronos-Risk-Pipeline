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


def as_yfinance_frame(close: pd.Series, ticker: str) -> pd.DataFrame:
    """Da a una Series la forma MultiIndex (campo, ticker) de yfinance."""
    columns = pd.MultiIndex.from_product([["Close"], [ticker]])
    return pd.DataFrame(
        close.to_numpy().reshape(-1, 1), index=close.index, columns=columns
    )


def make_settings(
    tmp_path: Path, asset: AssetConfig, retrain: RetrainTriggerConfig
) -> PipelineSettings:
    """Configuración con rutas temporales (base y reportes aislados)."""
    paths = PathsConfig(
        database_path=tmp_path / "data" / "pipeline.db",
        reports_dir=tmp_path / "reports",
    )
    return PipelineSettings(assets=(asset,), paths=paths, retrain=retrain)

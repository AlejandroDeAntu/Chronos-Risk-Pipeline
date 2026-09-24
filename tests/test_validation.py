"""Pruebas de contratos Pandera: fallan ruidosamente ante datos malos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    validate_daily_closes,
    validate_intraday_closes,
    validate_log_returns,
)
from src.exceptions import DataContractError


def test_validate_daily_closes_accepts_clean_frame(
    synthetic_daily_prices: pd.Series,
) -> None:
    frame = synthetic_daily_prices.to_frame()
    validated = validate_daily_closes(frame)
    assert len(validated) == len(frame)


def test_validate_daily_closes_rejects_null_values(
    synthetic_daily_prices: pd.Series,
) -> None:
    frame = synthetic_daily_prices.to_frame()
    frame.iloc[5, 0] = np.nan
    with pytest.raises(DataContractError):
        validate_daily_closes(frame)


def test_validate_daily_closes_rejects_non_positive_prices(
    synthetic_daily_prices: pd.Series,
) -> None:
    frame = synthetic_daily_prices.to_frame()
    frame.iloc[10, 0] = -1.0
    with pytest.raises(DataContractError):
        validate_daily_closes(frame)


def test_validate_daily_closes_rejects_duplicate_index(
    synthetic_daily_prices: pd.Series,
) -> None:
    frame = synthetic_daily_prices.to_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(DataContractError):
        validate_daily_closes(duplicated)


def test_validate_daily_closes_rejects_unexpected_column(
    synthetic_daily_prices: pd.Series,
) -> None:
    frame = synthetic_daily_prices.to_frame()
    frame["extra_unvalidated_column"] = 1.0
    with pytest.raises(DataContractError):
        validate_daily_closes(frame)


def test_validate_intraday_closes_accepts_clean_frame(
    synthetic_intraday_close: pd.Series,
) -> None:
    frame = synthetic_intraday_close.to_frame()
    assert len(validate_intraday_closes(frame)) == len(frame)


def test_validate_log_returns_rejects_out_of_range_values() -> None:
    frame = pd.DataFrame({"log_return": [0.01, -0.02, 5.0]})
    with pytest.raises(DataContractError):
        validate_log_returns(frame)


def test_validate_log_returns_rejects_nulls() -> None:
    frame = pd.DataFrame({"log_return": [0.01, np.nan, -0.02]})
    with pytest.raises(DataContractError):
        validate_log_returns(frame)

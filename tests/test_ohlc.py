"""Ingesta y contratos OHLC: reparación explícita, contratos y staleness."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.data import fetch
from src.data.validation import (
    repair_ohlc,
    validate_daily_ohlc,
    warn_if_stale,
    zero_return_share,
)
from src.exceptions import DataContractError, DataUnavailableError
from tests.helpers import as_yfinance_frame, synthetic_ohlc

_FINAL = pd.Timedelta(hours=21, minutes=15)


def _ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.date_range("2026-01-05", periods=len(rows), freq="B")
    return pd.DataFrame(
        rows, index=index, columns=["open", "high", "low", "close"]
    )


def test_fetch_daily_ohlc_returns_four_float_columns(
    synthetic_daily_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = as_yfinance_frame(synthetic_daily_prices, "^TEST")
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)

    result = fetch.fetch_daily_ohlc(
        "^TEST", "2020-01-01", min_rows=10, final_after=_FINAL
    )

    assert list(result.columns) == ["open", "high", "low", "close"]
    assert (result.dtypes == "float64").all()
    assert len(result) == len(synthetic_daily_prices)


def test_fetch_daily_ohlc_rejects_response_without_high(
    synthetic_daily_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = as_yfinance_frame(synthetic_daily_prices, "^TEST")
    raw = raw.drop(columns="High", level=0)
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)

    with pytest.raises(DataUnavailableError, match="High"):
        fetch.fetch_daily_ohlc(
            "^TEST", "2020-01-01", min_rows=10, final_after=_FINAL
        )


def test_repair_ohlc_widens_inconsistent_bars_and_counts_them() -> None:
    frame = _ohlc(
        [
            (100.0, 101.0, 99.0, 100.5),  # consistente
            (100.5, 100.2, 99.5, 100.8),  # high < close (Yahoo FX)
            (100.8, 101.5, 100.9, 100.6),  # low > close
        ]
    )

    repaired, n_repaired = repair_ohlc(frame, max_repair_share=0.9)

    assert n_repaired == 2
    assert repaired.loc[frame.index[1], "high"] == pytest.approx(100.8)
    assert repaired.loc[frame.index[2], "low"] == pytest.approx(100.6)
    assert repaired.loc[frame.index[0]].equals(frame.loc[frame.index[0]])
    # No muta la entrada.
    assert frame.loc[frame.index[1], "high"] == pytest.approx(100.2)


def test_repair_ohlc_fails_when_too_many_bars_are_broken() -> None:
    frame = _ohlc([(100.0, 99.0, 98.0, 100.0)] * 3)

    with pytest.raises(DataContractError, match="reparar"):
        repair_ohlc(frame, max_repair_share=0.05)


def test_validate_daily_ohlc_accepts_consistent_bars(
    synthetic_daily_prices: pd.Series,
) -> None:
    frame = synthetic_ohlc(synthetic_daily_prices)
    assert validate_daily_ohlc(frame).equals(frame)


@pytest.mark.parametrize(
    "row",
    [
        (100.0, 99.0, 98.0, 100.0),  # high < open
        (100.0, 101.0, 100.5, 100.0),  # low > close
        (100.0, 101.0, -1.0, 100.0),  # precio negativo
    ],
)
def test_validate_daily_ohlc_rejects_contract_violations(
    row: tuple[float, float, float, float],
) -> None:
    frame = _ohlc([(100.0, 101.0, 99.0, 100.0), row])
    with pytest.raises(DataContractError):
        validate_daily_ohlc(frame)


def test_zero_return_share_counts_repeated_quotes() -> None:
    close = pd.Series([1.0, 1.0, 1.0, 1.1, 1.1])
    # Retornos: 0, 0, +, 0 -> 3 de 4 son cero.
    assert zero_return_share(close) == pytest.approx(0.75)


def test_warn_if_stale_logs_above_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    close = pd.Series(np.r_[np.ones(50), np.linspace(1.0, 2.0, 50)])

    with caplog.at_level(logging.WARNING):
        share = warn_if_stale(close, "EURUSD=X", max_zero_share=0.2)

    assert share > 0.2
    assert "rancias" in caplog.text


def test_warn_if_stale_is_silent_below_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    close = pd.Series(np.linspace(1.0, 2.0, 100))

    with caplog.at_level(logging.WARNING):
        warn_if_stale(close, "^TEST", max_zero_share=0.2)

    assert caplog.text == ""

"""Pruebas de ingesta: mockean ``_download_raw``, nunca usan la red."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data import fetch
from src.exceptions import DataUnavailableError
from tests.helpers import as_yfinance_frame

_FINAL = pd.Timedelta(hours=21, minutes=15)


def test_fetch_daily_closes_normalizes_multiindex_columns(
    synthetic_daily_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = as_yfinance_frame(synthetic_daily_prices, "^TEST")
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)

    result = fetch.fetch_daily_closes(
        "^TEST", "2020-01-01", min_rows=10, final_after=_FINAL
    )

    assert isinstance(result, pd.Series)
    assert result.name == "close"
    assert result.dtype == "float64"
    assert len(result) == len(synthetic_daily_prices)


def test_fetch_daily_closes_raises_on_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fetch, "_download_raw", lambda symbol, **kw: pd.DataFrame()
    )
    with pytest.raises(DataUnavailableError):
        fetch.fetch_daily_closes(
            "^TEST", "2020-01-01", min_rows=10, final_after=_FINAL
        )


def test_fetch_daily_closes_raises_when_below_min_rows(
    synthetic_daily_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = as_yfinance_frame(synthetic_daily_prices.head(5), "^TEST")
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)
    with pytest.raises(DataUnavailableError):
        fetch.fetch_daily_closes(
            "^TEST", "2020-01-01", min_rows=1000, final_after=_FINAL
        )


def test_fetch_daily_closes_rejects_duplicated_timestamps(
    synthetic_daily_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = as_yfinance_frame(synthetic_daily_prices, "^TEST")
    raw = pd.concat([raw, raw.iloc[[0]]])
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)
    with pytest.raises(DataUnavailableError):
        fetch.fetch_daily_closes(
            "^TEST", "2020-01-01", min_rows=10, final_after=_FINAL
        )


def test_fetch_intraday_bars_drops_bar_still_open(
    synthetic_intraday_close: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = as_yfinance_frame(synthetic_intraday_close, "^TEST")
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)
    last_open = synthetic_intraday_close.index[-1]
    # 2 minutos después de abrir la última vela de 5 min: sigue abierta.
    now = (last_open + pd.Timedelta(minutes=2)).to_pydatetime()

    frame = fetch.fetch_intraday_bars("^TEST", 5, "5d", min_rows=10, now=now)

    assert last_open not in frame.index
    assert frame.index[-1] == synthetic_intraday_close.index[-2]


def test_fetch_intraday_bars_keeps_bar_once_closed(
    synthetic_intraday_close: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = as_yfinance_frame(synthetic_intraday_close, "^TEST")
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)
    now: datetime = (
        synthetic_intraday_close.index[-1] + pd.Timedelta(minutes=5)
    ).to_pydatetime()

    frame = fetch.fetch_intraday_bars("^TEST", 5, "5d", min_rows=10, now=now)

    assert frame.index[-1] == synthetic_intraday_close.index[-1]


def test_fetch_daily_closes_drops_bars_not_yet_final(
    synthetic_daily_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    # La vela del 2026-09-18 (última) aún no cierra a las 15:00 UTC.
    raw = as_yfinance_frame(synthetic_daily_prices, "^TEST")
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)
    now = pd.Timestamp("2026-09-18 15:00", tz="UTC").to_pydatetime()

    result = fetch.fetch_daily_closes(
        "^TEST", "2020-01-01", min_rows=10, final_after=_FINAL, now=now
    )

    assert result.index[-1] == pd.Timestamp("2026-09-17")
    assert len(result) == len(synthetic_daily_prices) - 1


def test_fetch_daily_closes_keeps_bar_after_session_close(
    synthetic_daily_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = as_yfinance_frame(synthetic_daily_prices, "^TEST")
    monkeypatch.setattr(fetch, "_download_raw", lambda symbol, **kw: raw)
    now = pd.Timestamp("2026-09-18 22:00", tz="UTC").to_pydatetime()

    result = fetch.fetch_daily_closes(
        "^TEST", "2020-01-01", min_rows=10, final_after=_FINAL, now=now
    )

    assert result.index[-1] == pd.Timestamp("2026-09-18")

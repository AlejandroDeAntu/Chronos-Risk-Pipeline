"""Ingesta de datos de cierre desde Yahoo Finance con reintentos y validación.

Corrige cuatro defectos del proyecto original:

1. ``yf.download`` se llamaba sin reintentos ni manejo de fallos de red.
2. No había verificación explícita de nulos, tipos ni duplicados.
3. El código dependía de si ``df['Close']`` era Series o DataFrame de una
   columna (varía entre versiones de ``yfinance``); aquí se normaliza.
4. En intradía se analizaba la última vela aunque siguiera abierta: su
   "cierre" es el último precio, no el cierre real. Aquí se descartan las
   velas cuyo cierre aún no ocurre.

Nota sobre encoding: la fuente es una API que devuelve DataFrames ya
tipados, no un archivo de texto, así que no hay encoding que validar aquí;
la verificación equivalente es la de dtypes y esquema.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TypeVar

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.exceptions import DataUnavailableError

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
)
def _download_raw(symbol: str, **kwargs: str) -> pd.DataFrame:
    """Llama a ``yf.download`` con reintento exponencial ante errores de red.

    Parameters
    ----------
    symbol:
        Ticker de Yahoo Finance.
    **kwargs:
        Argumentos reenviados a ``yfinance.download`` (``start``,
        ``interval``, ``period``).

    Returns
    -------
    pandas.DataFrame
        Respuesta cruda, todavía sin validar.
    """
    return yf.download(
        tickers=symbol, progress=False, auto_adjust=True, **kwargs
    )


_OHLC_FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close"}
_SeriesOrFrame = TypeVar("_SeriesOrFrame", pd.Series, pd.DataFrame)


def _extract_field(raw: pd.DataFrame, field: str, symbol: str) -> pd.Series:
    """Extrae un campo de yfinance como Series ``float64``.

    Normaliza la respuesta, que según la versión de ``yfinance`` trae el
    campo como Series o como DataFrame de una columna (MultiIndex).

    Raises
    ------
    DataUnavailableError
        Si la respuesta está vacía, no trae el campo, trae varias columnas
        para él o no es numérico.
    """
    if raw.empty:
        raise DataUnavailableError(
            f"yfinance devolvió un DataFrame vacío para '{symbol}'."
        )
    if field not in raw.columns.get_level_values(0):
        raise DataUnavailableError(
            f"La respuesta para '{symbol}' no incluye la columna '{field}'."
        )

    values = raw[field]
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise DataUnavailableError(
                f"Se esperaba una columna '{field}' para '{symbol}', "
                f"se obtuvieron {values.shape[1]}."
            )
        values = values.iloc[:, 0]

    if not pd.api.types.is_numeric_dtype(values):
        raise DataUnavailableError(
            f"'{field}' de '{symbol}' no es numérico (dtype={values.dtype})."
        )
    return values.astype("float64")


def _extract_close(raw: pd.DataFrame, symbol: str) -> pd.Series:
    """Extrae el cierre como Series ``float64`` llamada ``close``."""
    return _extract_field(raw, "Close", symbol).rename("close")


def _extract_ohlc(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extrae open/high/low/close como DataFrame ``float64``."""
    return pd.DataFrame(
        {
            name: _extract_field(raw, field, symbol)
            for field, name in _OHLC_FIELDS.items()
        }
    )


def _drop_nulls_and_check_duplicates(
    data: _SeriesOrFrame, symbol: str
) -> _SeriesOrFrame:
    """Elimina filas con nulos (registrándolo) y rechaza duplicados."""
    has_null = data.isna()
    if isinstance(has_null, pd.DataFrame):
        has_null = has_null.any(axis=1)
    n_nulls = int(has_null.sum())
    if n_nulls > 0:
        logger.warning(
            "Se descartaron %d filas con nulos de '%s'.", n_nulls, symbol
        )
        data = data[~has_null]

    n_duplicates = int(data.index.duplicated().sum())
    if n_duplicates > 0:
        raise DataUnavailableError(
            f"'{symbol}' tiene {n_duplicates} timestamps duplicados; se "
            "rechaza en vez de deduplicar en silencio."
        )
    return data.sort_index()


def fetch_daily_closes(
    symbol: str,
    start_date: str,
    min_rows: int,
    final_after: pd.Timedelta,
    now: datetime | None = None,
) -> pd.Series:
    """Cierres diarios finalizados (columna ``close`` de ``fetch_daily_ohlc``).

    Raises
    ------
    DataUnavailableError
        Si la descarga falla o no alcanza ``min_rows`` filas.
    """
    return fetch_daily_ohlc(symbol, start_date, min_rows, final_after, now)[
        "close"
    ]


def fetch_daily_ohlc(
    symbol: str,
    start_date: str,
    min_rows: int,
    final_after: pd.Timedelta,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Descarga velas diarias OHLC finalizadas, sin nulos ni duplicados.

    Parameters
    ----------
    symbol:
        Ticker de Yahoo Finance (ej. ``"^GSPC"``).
    start_date:
        Fecha de inicio ``YYYY-MM-DD``.
    min_rows:
        Mínimo de filas utilizables requerido.
    final_after:
        Tiempo, desde la medianoche UTC de la fecha de la vela, a partir del
        cual su cierre es definitivo. Yahoo publica la vela del día en curso
        con el último precio (en FX, incluso con fecha de sábado); usarla
        como origen o como resultado real contamina forecast y evaluación.
    now:
        Instante UTC de referencia; por defecto, ahora. Se inyecta en tests.

    Returns
    -------
    pandas.DataFrame
        Columnas ``open``, ``high``, ``low`` y ``close`` (float64), sin
        nulos ni duplicados, en orden cronológico, solo velas finalizadas.
        La consistencia OHLC se valida aparte (``repair_ohlc`` y
        ``validate_daily_ohlc``).

    Raises
    ------
    DataUnavailableError
        Si la descarga falla o no alcanza ``min_rows`` filas.
    """
    try:
        raw = _download_raw(symbol, start=start_date, interval="1d")
    except _RETRYABLE_EXCEPTIONS as exc:
        raise DataUnavailableError(
            f"No se pudo descargar el diario de '{symbol}': {exc}"
        ) from exc

    ohlc = _drop_nulls_and_check_duplicates(_extract_ohlc(raw, symbol), symbol)
    reference = pd.Timestamp(now or datetime.now(UTC)).tz_convert(None)
    bar_date = ohlc.index.tz_localize(None).normalize()
    is_final = bar_date + final_after <= reference
    n_open = int((~is_final).sum())
    if n_open > 0:
        logger.info(
            "Se descartaron %d velas diarias no finalizadas de '%s'.",
            n_open,
            symbol,
        )
    ohlc = ohlc[is_final]
    if len(ohlc) < min_rows:
        raise DataUnavailableError(
            f"'{symbol}' tiene {len(ohlc)} filas, se requieren {min_rows}."
        )
    return ohlc


def fetch_intraday_bars(
    symbol: str,
    bar_minutes: int,
    period: str,
    min_rows: int,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Descarga velas intradía cerradas y las devuelve validadas.

    Parameters
    ----------
    symbol:
        Ticker de Yahoo Finance.
    bar_minutes:
        Duración de la vela en minutos (ej. ``5``).
    period:
        Ventana histórica solicitada (ej. ``"5d"``).
    min_rows:
        Mínimo de velas cerradas requerido.
    now:
        Instante de referencia UTC para decidir qué velas ya cerraron; si es
        ``None`` se usa la hora actual. Se inyecta para poder probarlo.

    Returns
    -------
    pandas.DataFrame
        Columna ``close`` (float64) con índice UTC único y ordenado, solo
        con velas cuyo cierre ya ocurrió.

    Raises
    ------
    DataUnavailableError
        Si la descarga falla o no hay suficientes velas cerradas.
    """
    try:
        raw = _download_raw(symbol, interval=f"{bar_minutes}m", period=period)
    except _RETRYABLE_EXCEPTIONS as exc:
        raise DataUnavailableError(
            f"No se pudo descargar intradía de '{symbol}': {exc}"
        ) from exc

    close = _extract_close(raw, symbol)
    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    else:
        close.index = close.index.tz_convert("UTC")
    close = _drop_nulls_and_check_duplicates(close, symbol)

    reference = now or datetime.now(UTC)
    bar_close_time = close.index + pd.Timedelta(minutes=bar_minutes)
    n_open = int((bar_close_time > pd.Timestamp(reference)).sum())
    if n_open > 0:
        logger.info(
            "Se descartaron %d velas aún abiertas de '%s'.", n_open, symbol
        )
    close = close[bar_close_time <= pd.Timestamp(reference)]

    if len(close) < min_rows:
        raise DataUnavailableError(
            f"'{symbol}' tiene {len(close)} velas cerradas, "
            f"se requieren {min_rows}."
        )
    return close.to_frame()

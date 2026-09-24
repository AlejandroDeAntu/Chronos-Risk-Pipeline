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


def _extract_close(raw: pd.DataFrame, symbol: str) -> pd.Series:
    """Extrae la columna de cierre como Series ``float64`` llamada ``close``.

    Raises
    ------
    DataUnavailableError
        Si la respuesta está vacía, no trae ``Close`` o trae varias columnas
        de cierre.
    """
    if raw.empty:
        raise DataUnavailableError(
            f"yfinance devolvió un DataFrame vacío para '{symbol}'."
        )
    if "Close" not in raw.columns.get_level_values(0):
        raise DataUnavailableError(
            f"La respuesta para '{symbol}' no incluye la columna 'Close'."
        )

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise DataUnavailableError(
                f"Se esperaba una columna de cierre para '{symbol}', "
                f"se obtuvieron {close.shape[1]}."
            )
        close = close.iloc[:, 0]

    if not pd.api.types.is_numeric_dtype(close):
        raise DataUnavailableError(
            f"El cierre de '{symbol}' no es numérico (dtype={close.dtype})."
        )
    return close.rename("close").astype("float64")


def _drop_nulls_and_check_duplicates(
    close: pd.Series, symbol: str
) -> pd.Series:
    """Elimina nulos (registrándolo) y rechaza timestamps duplicados."""
    n_nulls = int(close.isna().sum())
    if n_nulls > 0:
        logger.warning(
            "Se descartaron %d cierres nulos de '%s'.", n_nulls, symbol
        )
        close = close.dropna()

    n_duplicates = int(close.index.duplicated().sum())
    if n_duplicates > 0:
        raise DataUnavailableError(
            f"'{symbol}' tiene {n_duplicates} timestamps duplicados; se "
            "rechaza en vez de deduplicar en silencio."
        )
    return close.sort_index()


def fetch_daily_closes(
    symbol: str, start_date: str, min_rows: int
) -> pd.Series:
    """Descarga cierres diarios y los devuelve como Series validada.

    Parameters
    ----------
    symbol:
        Ticker de Yahoo Finance (ej. ``"^GSPC"``).
    start_date:
        Fecha de inicio ``YYYY-MM-DD``.
    min_rows:
        Mínimo de filas utilizables requerido.

    Returns
    -------
    pandas.Series
        Cierres diarios sin nulos ni duplicados, en orden cronológico.

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

    close = _drop_nulls_and_check_duplicates(
        _extract_close(raw, symbol), symbol
    )
    if len(close) < min_rows:
        raise DataUnavailableError(
            f"'{symbol}' tiene {len(close)} filas, se requieren {min_rows}."
        )
    return close


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

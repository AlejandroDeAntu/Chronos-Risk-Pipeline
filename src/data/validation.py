"""Esquemas Pandera para validar DataFrames antes de operar sobre ellos.

El proyecto original nunca verificaba tipos, nulos ni duplicados de forma
explícita: dependía de que ``dropna()`` "arreglara" cualquier problema. Aquí
la validación es una puerta dura: si el contrato no se cumple, Pandera
lanza ``pandera.errors.SchemaError`` y el pipeline se detiene con un mensaje
específico en vez de continuar con datos corruptos.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.exceptions import DataContractError

logger = logging.getLogger(__name__)

OHLC_COLUMNS = ["open", "high", "low", "close"]

intraday_close_schema = pa.DataFrameSchema(
    columns={
        "close": pa.Column(
            float,
            checks=[pa.Check.gt(0.0)],
            nullable=False,
        ),
    },
    # No se fija un dtype de índice aquí: la resolución exacta de
    # `datetime64` (ns vs us) varía entre versiones de pandas/yfinance, y
    # fijarla haría el contrato frágil ante actualizaciones de dependencias.
    # La unicidad se valida igual; el tipo fecha/hora y la zona horaria UTC
    # se verifican aparte, explícitamente, en `validate_intraday_closes`.
    index=pa.Index(unique=True),
    strict=True,
    coerce=False,
)

log_return_schema = pa.DataFrameSchema(
    columns={
        "log_return": pa.Column(
            float, nullable=False, checks=pa.Check.in_range(-1.0, 1.0)
        ),
    },
    strict=False,
    coerce=False,
)


def validate_intraday_closes(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida un DataFrame de velas intradía contra ``intraday_close_schema``.

    Parameters
    ----------
    frame:
        DataFrame con columna ``close`` e índice datetime UTC único.

    Returns
    -------
    pandas.DataFrame
        El mismo DataFrame, ya validado.

    Raises
    ------
    DataContractError
        Si el esquema no se cumple, o si el índice no está en zona horaria UTC.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataContractError(
            "El índice intradía debe ser DatetimeIndex, se recibió "
            f"{type(frame.index).__name__}."
        )
    if frame.index.tz is None or str(frame.index.tz) != "UTC":
        raise DataContractError(
            "El índice intradía debe estar en UTC, se recibió "
            f"tz={frame.index.tz!r}."
        )

    try:
        return intraday_close_schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataContractError(
            "Contrato de velas intradía violado: "
            f"{exc.failure_cases.to_dict()}"
        ) from exc


def validate_log_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida que una columna ``log_return`` sea numérica, sin nulos y acotada.

    Un rango de ``[-1.0, 1.0]`` en log-retorno equivale a movimientos de
    hasta ~+-171% en una sola barra, un límite generoso pensado para atrapar
    errores de unidades (ej. mezclar porcentaje con proporción) y no para
    rechazar movimientos extremos legítimos.

    Parameters
    ----------
    frame:
        DataFrame que contiene al menos la columna ``log_return``.

    Returns
    -------
    pandas.DataFrame
        El mismo DataFrame, ya validado.

    Raises
    ------
    DataContractError
        Si ``log_return`` tiene nulos, no es numérica o está fuera de rango.
    """
    try:
        return log_return_schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataContractError(
            f"Contrato de log-retornos violado: {exc.failure_cases.to_dict()}"
        ) from exc


def _positive_price() -> pa.Column:
    return pa.Column(float, checks=pa.Check.gt(0.0), nullable=False)


daily_ohlc_schema = pa.DataFrameSchema(
    columns={column: _positive_price() for column in OHLC_COLUMNS},
    checks=[
        pa.Check(
            lambda df: df["high"] >= df[["open", "close"]].max(axis=1),
            error="high < max(open, close)",
        ),
        pa.Check(
            lambda df: df["low"] <= df[["open", "close"]].min(axis=1),
            error="low > min(open, close)",
        ),
    ],
    index=pa.Index(unique=True),
    strict=True,
    coerce=False,
)


def align_fx_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-alinea las velas diarias FX de Yahoo al día que describen.

    En las velas diarias FX de Yahoo, ``open`` y ``close`` de la fecha D son
    casi idénticos (mediana ``|ln(O/C)|`` ~ 6e-5, 2010-2026) y coinciden
    con el precio al inicio del día D (contrastado con velas de 5 min),
    mientras que ``high``/``low`` cubren el día D. Sin corregir, el retorno
    "cierre a cierre" fechado D mide el día D-1 y el rango de D se solapa
    con el retorno que se quiere pronosticar: fuga de información para
    cualquier modelo que use el rango.

    Corrección: el cierre del día D es el precio al inicio de D+1 (el
    ``close`` de la vela siguiente). La última vela se descarta porque su
    cierre real aún no existe, y ``high``/``low`` se ensanchan para
    contener al nuevo cierre (la ventana del rango de Yahoo no cubre
    exactamente las 24 h).

    Parameters
    ----------
    frame:
        Velas FX crudas (``open``, ``high``, ``low``, ``close``).

    Returns
    -------
    pandas.DataFrame
        Copia re-alineada, una fila menos.
    """
    aligned = frame.assign(close=frame["close"].shift(-1)).iloc[:-1]
    return aligned.assign(
        high=aligned[["high", "open", "close"]].max(axis=1),
        low=aligned[["low", "open", "close"]].min(axis=1),
    )


def repair_ohlc(
    frame: pd.DataFrame, max_repair_share: float
) -> tuple[pd.DataFrame, int]:
    """Ensancha high/low para envolver a open y close, sin mutar la entrada.

    Yahoo publica en FX entre 1% y 3% de velas diarias con ``high`` menor
    que el open o el close (medido 2010-2026). Rechazarlas tumbaría el
    pipeline a diario; repararlas en silencio ocultaría el problema. Aquí se
    reparan, se registran y se falla si la proporción es anómala.

    Parameters
    ----------
    frame:
        Velas con columnas ``open``, ``high``, ``low`` y ``close``.
    max_repair_share:
        Proporción máxima de velas reparables antes de fallar.

    Returns
    -------
    tuple[pandas.DataFrame, int]
        Copia reparada y número de velas modificadas.

    Raises
    ------
    DataContractError
        Si la proporción de velas inconsistentes supera ``max_repair_share``.
    """
    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)
    broken = (frame["high"] < body_high) | (frame["low"] > body_low)
    n_broken = int(broken.sum())
    share = n_broken / len(frame) if len(frame) else 0.0
    if share > max_repair_share:
        raise DataContractError(
            f"{n_broken} de {len(frame)} velas ({share:.1%}) son "
            "inconsistentes; supera el máximo permitido para reparar "
            f"({max_repair_share:.1%})."
        )
    repaired = frame.assign(
        high=np.maximum(frame["high"], body_high),
        low=np.minimum(frame["low"], body_low),
    )
    if n_broken:
        logger.warning(
            "Se repararon %d velas OHLC inconsistentes (%.2f%%).",
            n_broken,
            100.0 * share,
        )
    return repaired, n_broken


def validate_daily_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida velas diarias OHLC contra ``daily_ohlc_schema``.

    Parameters
    ----------
    frame:
        DataFrame con ``open``, ``high``, ``low`` y ``close`` e índice de
        fechas único.

    Returns
    -------
    pandas.DataFrame
        El mismo DataFrame, ya validado.

    Raises
    ------
    DataContractError
        Si el índice no es de fechas o el contrato no se cumple.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataContractError(
            "El índice OHLC diario debe ser DatetimeIndex, se recibió "
            f"{type(frame.index).__name__}."
        )
    try:
        return daily_ohlc_schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataContractError(
            "Contrato OHLC diario violado: "
            f"{exc.failure_cases.head(10).to_dict()}"
        ) from exc


def zero_return_share(close: pd.Series) -> float:
    """Proporción de retornos exactamente cero (cotización repetida)."""
    changes = close.diff().iloc[1:]
    if changes.empty:
        return 0.0
    return float((changes == 0.0).mean())


def warn_if_stale(
    close: pd.Series, symbol: str, max_zero_share: float
) -> float:
    """Advierte si hay demasiadas cotizaciones repetidas; devuelve la tasa.

    Un exceso de retornos cero indica cotizaciones rancias de la fuente
    (EUR/USD en 5 min llega a 34%) y sesga cualquier métrica basada en el
    signo o la magnitud del retorno. No detiene el pipeline: la proporción
    se reporta para que el analista la vea.
    """
    share = zero_return_share(close)
    if share > max_zero_share:
        logger.warning(
            "'%s': %.1f%% de retornos son cero (cotizaciones rancias); "
            "umbral %.1f%%.",
            symbol,
            100.0 * share,
            100.0 * max_zero_share,
        )
    return share

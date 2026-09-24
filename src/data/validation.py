"""Esquemas Pandera para validar DataFrames antes de operar sobre ellos.

El proyecto original nunca verificaba tipos, nulos ni duplicados de forma
explícita: dependía de que ``dropna()`` "arreglara" cualquier problema. Aquí
la validación es una puerta dura: si el contrato no se cumple, Pandera
lanza ``pandera.errors.SchemaError`` y el pipeline se detiene con un mensaje
específico en vez de continuar con datos corruptos.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from src.exceptions import DataContractError

daily_close_schema = pa.DataFrameSchema(
    columns={
        "close": pa.Column(
            float,
            checks=[
                pa.Check.gt(0.0),
                pa.Check(lambda s: s.notna().all(), element_wise=False),
            ],
            nullable=False,
        ),
    },
    # Ver nota en `intraday_close_schema` sobre por qué no se fija el dtype
    # exacto del índice aquí.
    index=pa.Index(unique=True),
    strict=True,
    coerce=False,
)

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


def validate_daily_closes(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida un DataFrame de cierres diarios contra ``daily_close_schema``.

    Parameters
    ----------
    frame:
        DataFrame con una única columna ``close`` e índice de fechas único.

    Returns
    -------
    pandas.DataFrame
        El mismo DataFrame, ya validado.

    Raises
    ------
    DataContractError
        Si el índice no es de tipo fecha, o si el esquema no se cumple
        (nulos, precios <= 0, índice duplicado).
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataContractError(
            "El índice de cierres diarios debe ser DatetimeIndex, se recibió "
            f"{type(frame.index).__name__}."
        )

    try:
        return daily_close_schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataContractError(
            "Contrato de cierres diarios violado: "
            f"{exc.failure_cases.to_dict()}"
        ) from exc


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

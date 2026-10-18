"""Estacionalidad intradía de la volatilidad.

La volatilidad de 5 minutos no es constante en el día: sube en la apertura
de Nueva York, en los cierres y en publicaciones macro a hora fija. Un
z-score con ventana móvil de 20 velas reacciona con retraso a ese patrón y
marca como "anomalía" lo que es simplemente la hora pico. Aquí se estima
un perfil por hora UTC y se divide cada retorno por el factor de su hora.
"""

from __future__ import annotations

import pandas as pd

HOURS: tuple[int, ...] = tuple(range(24))
_MIN_OBS_PER_HOUR = 30
_MIN_FACTOR = 0.1


def hourly_profile(log_return: pd.Series) -> pd.Series:
    """Factor de volatilidad por hora UTC (mediana de ``|r|`` relativa).

    Se usa la mediana de ``|r|`` en vez de la desviación estándar porque es
    robusta a los propios saltos que el detector debe encontrar. El perfil
    se normaliza para que la hora típica valga 1.

    Parameters
    ----------
    log_return:
        Retornos intradía con índice datetime con zona horaria.

    Returns
    -------
    pandas.Series
        24 factores positivos indexados por hora; las horas con menos de
        30 observaciones (o sin variación) valen 1.
    """
    absolute = log_return.abs()
    grouped = absolute.groupby(log_return.index.hour)
    median_by_hour = grouped.median().where(
        grouped.size() >= _MIN_OBS_PER_HOUR
    )
    overall = float(absolute.median())
    if overall <= 0.0:
        return pd.Series(1.0, index=HOURS, name="seasonal_factor")
    profile = (median_by_hour / overall).reindex(HOURS)
    # Mediana cero (horas FX sin cotización nueva): piso, no 1.0, para no
    # inflar el z-score; 1.0 solo para horas sin datos suficientes.
    profile = profile.clip(lower=_MIN_FACTOR).fillna(1.0)
    return profile.rename("seasonal_factor")


def deseasonalize(log_return: pd.Series, profile: pd.Series) -> pd.Series:
    """Divide cada retorno por el factor de su hora UTC."""
    factors = profile.reindex(log_return.index.hour).to_numpy()
    return (log_return / factors).rename(log_return.name)

"""Estimadores de varianza diaria a partir de velas OHLC.

El retorno al cuadrado ``r_t^2`` es un proxy insesgado pero muy ruidoso de
la varianza diaria: usa un solo precio por día. Los estimadores por rango
usan máximo, mínimo, apertura y cierre, y reducen la varianza del proxy
entre 5 y 14 veces bajo un movimiento browniano:

* Parkinson (1980): solo high/low; supone deriva cero.
* Garman-Klass (1980): agrega open/close; supone deriva cero.
* Rogers-Satchell (1991): robusto a deriva distinta de cero.
* Yang-Zhang (2000): combina varianza overnight, open-to-close y
  Rogers-Satchell en una ventana; es el único que incluye el salto entre
  el cierre previo y la apertura.

Todos devuelven varianzas en unidades de log-retorno al cuadrado, con el
mismo índice que la entrada. Las operaciones son vectorizadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TWO_LN2_MINUS_1 = 2.0 * np.log(2.0) - 1.0


def parkinson(ohlc: pd.DataFrame) -> pd.Series:
    """Varianza de Parkinson: ``ln(H/L)^2 / (4 ln 2)``."""
    high_low = np.log(ohlc["high"] / ohlc["low"])
    return (high_low**2 / (4.0 * np.log(2.0))).rename("parkinson")


def garman_klass(ohlc: pd.DataFrame) -> pd.Series:
    """Varianza de Garman-Klass.

    ``0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2``. Nunca es negativa porque
    ``|ln(C/O)| <= ln(H/L)`` en una vela consistente.
    """
    high_low = np.log(ohlc["high"] / ohlc["low"])
    close_open = np.log(ohlc["close"] / ohlc["open"])
    variance = 0.5 * high_low**2 - _TWO_LN2_MINUS_1 * close_open**2
    return variance.rename("garman_klass")


def rogers_satchell(ohlc: pd.DataFrame) -> pd.Series:
    """Varianza de Rogers-Satchell: ``u (u - c) + d (d - c)``.

    Con ``u = ln(H/O)``, ``d = ln(L/O)`` y ``c = ln(C/O)``.
    """
    up = np.log(ohlc["high"] / ohlc["open"])
    down = np.log(ohlc["low"] / ohlc["open"])
    close_open = np.log(ohlc["close"] / ohlc["open"])
    variance = up * (up - close_open) + down * (down - close_open)
    return variance.rename("rogers_satchell")


def overnight_log_return(ohlc: pd.DataFrame) -> pd.Series:
    """Retorno del cierre previo a la apertura: ``ln(O_t / C_{t-1})``."""
    return np.log(ohlc["open"] / ohlc["close"].shift(1)).rename("overnight")


def yang_zhang(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """Varianza de Yang-Zhang sobre una ventana móvil de ``window`` velas.

    ``var(overnight) + k var(open-to-close) + (1 - k) mean(RS)`` con
    ``k = 0.34 / (1.34 + (n + 1) / (n - 1))``, que minimiza la varianza
    del estimador. El valor en ``t`` usa velas ``t-window+1 .. t``.

    Raises
    ------
    ValueError
        Si ``window < 2`` (la varianza muestral no está definida).
    """
    if window < 2:
        raise ValueError(f"window debe ser >= 2, se recibió {window}.")
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    overnight = overnight_log_return(ohlc)
    open_close = np.log(ohlc["close"] / ohlc["open"])
    variance = (
        overnight.rolling(window).var(ddof=1)
        + k * open_close.rolling(window).var(ddof=1)
        + (1.0 - k) * rogers_satchell(ohlc).rolling(window).mean()
    )
    return variance.rename(f"yang_zhang_{window}")


def realized_variance_proxy(ohlc: pd.DataFrame) -> pd.Series:
    """Varianza realizada diaria: ``overnight^2 + Garman-Klass``.

    Es el componente diario de Yang-Zhang sin promediar. Garman-Klass solo
    mide el tramo apertura-cierre; sin el término overnight subestimaría la
    varianza de los índices, que abren con salto respecto al cierre previo.
    La primera vela es ``NaN`` porque no tiene cierre previo.
    """
    return (overnight_log_return(ohlc) ** 2 + garman_klass(ohlc)).rename(
        "realized_variance"
    )

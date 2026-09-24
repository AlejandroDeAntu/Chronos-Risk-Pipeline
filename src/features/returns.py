"""Retornos logarítmicos y bandas de anomalía causales.

Corrige dos defectos del proyecto original:

1. Fuga leve: la media y desviación móviles se calculaban sin desplazar, de
   modo que la barra evaluada contribuía a su propio umbral. Aquí toda
   estadística usada para juzgar la barra ``t`` usa solo ``t-window .. t-1``
   (``shift(1)``).
2. Retornos de salto: en índices bursátiles la primera vela de cada sesión
   se comparaba contra el cierre de la sesión anterior, de modo que el
   movimiento nocturno (o de fin de semana en forex) aparecía como
   "anomalía intradía". Aquí esos retornos se descartan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.exceptions import InsufficientHistoryError


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Calcula retornos logarítmicos ``log(P_t / P_{t-1})`` vectorizados.

    Parameters
    ----------
    close:
        Cierres sin nulos, en orden cronológico.

    Returns
    -------
    pandas.Series
        Retornos logarítmicos llamados ``log_return``, sin la primera fila.

    Raises
    ------
    InsufficientHistoryError
        Si hay menos de 2 precios.
    """
    if len(close) < 2:
        raise InsufficientHistoryError(
            f"Se requieren al menos 2 precios, se recibieron {len(close)}."
        )
    log_return = np.log(close / close.shift(1)).dropna()
    log_return.name = "log_return"
    return log_return


def compute_intraday_log_returns(
    close: pd.Series, bar_minutes: int
) -> pd.Series:
    """Retornos intradía excluyendo los que cruzan un hueco de mercado.

    Solo se conservan los retornos entre velas consecutivas separadas
    exactamente por ``bar_minutes``; los que abarcan un cierre de sesión o
    un fin de semana se descartan.

    Parameters
    ----------
    close:
        Cierres intradía con índice datetime ordenado.
    bar_minutes:
        Duración de la vela en minutos.

    Returns
    -------
    pandas.Series
        Retornos logarítmicos intradía sin retornos de salto.
    """
    log_return = compute_log_returns(close)
    elapsed = close.index.to_series().diff().reindex(log_return.index)
    return log_return[elapsed == pd.Timedelta(minutes=bar_minutes)]


def compute_causal_zscore_bands(
    log_return: pd.Series,
    rolling_window: int,
    min_bars_required: int,
) -> pd.DataFrame:
    """Calcula media/std móviles causales y el z-score de cada barra.

    Parameters
    ----------
    log_return:
        Retornos logarítmicos.
    rolling_window:
        Número de barras pasadas en la ventana móvil.
    min_bars_required:
        Mínimo de barras con estadística causal válida.

    Returns
    -------
    pandas.DataFrame
        Columnas ``log_return``, ``rolling_mean``, ``rolling_std`` y
        ``zscore``. Se eliminan las barras sin historia causal suficiente y
        las de desviación nula (z-score indefinido).

    Raises
    ------
    InsufficientHistoryError
        Si no quedan al menos ``min_bars_required`` barras válidas.
    """
    rolling = log_return.rolling(window=rolling_window)
    frame = pd.DataFrame(
        {
            "log_return": log_return,
            "rolling_mean": rolling.mean().shift(1),
            "rolling_std": rolling.std(ddof=1).shift(1),
        }
    ).dropna()
    frame = frame[frame["rolling_std"] > 0.0]

    if len(frame) < min_bars_required:
        raise InsufficientHistoryError(
            f"Solo hay {len(frame)} barras con estadística causal válida, "
            f"se requieren {min_bars_required}."
        )

    frame["zscore"] = (frame["log_return"] - frame["rolling_mean"]) / frame[
        "rolling_std"
    ]
    return frame

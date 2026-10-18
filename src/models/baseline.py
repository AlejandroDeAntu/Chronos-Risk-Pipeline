"""Baselines ingenuos obligatorios para contextualizar cualquier métrica.

Sin baseline, un F1 de 0.30 o un QLIKE de 1.2 no dicen si el modelo aporta
valor. El proyecto original no tenía ninguno.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_anomaly_baseline(
    n_observations: int, positive_rate: float, rng_seed: int
) -> np.ndarray:
    """Marca anomalías al azar con la misma tasa que el modelo.

    Si el detector no supera a este baseline, no aporta señal: acertaría lo
    mismo marcando barras al azar con la misma frecuencia.

    Raises
    ------
    ValueError
        Si ``positive_rate`` está fuera de ``[0, 1]``.
    """
    if not 0.0 <= positive_rate <= 1.0:
        raise ValueError(
            f"positive_rate debe estar en [0, 1], se recibió {positive_rate}."
        )
    rng = np.random.default_rng(rng_seed)
    return rng.random(n_observations) < positive_rate


def trailing_volatility_baseline(
    returns_pct: pd.Series, window: int
) -> pd.Series:
    """Pronóstico ingenuo: desviación estándar de los ``window`` días previos.

    El valor en la posición ``t`` usa solo retornos hasta ``t-1``
    (``shift(1)``), igual que el pronóstico GARCH a un paso, así que ambos
    se comparan con la misma información disponible.
    """
    return returns_pct.rolling(window=window).std(ddof=1).shift(1)


def ewma_volatility_baseline(
    returns_pct: pd.Series, decay: float = 0.94
) -> pd.Series:
    """Baseline RiskMetrics: ``s2_t = d * s2_{t-1} + (1 - d) * r2_{t-1}``.

    Es el baseline ingenuo estándar en riesgo de mercado y es más difícil
    de batir que la desviación móvil de 20 días. ``ewm(adjust=False)``
    implementa la recursión; ``shift(1)`` garantiza que el valor en ``t``
    solo use retornos hasta ``t-1``.

    Raises
    ------
    ValueError
        Si ``decay`` no está en ``(0, 1)``.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay debe estar en (0, 1), se recibió {decay}.")
    variance = (returns_pct**2).ewm(alpha=1.0 - decay, adjust=False).mean()
    return np.sqrt(variance).shift(1)

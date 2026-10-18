"""Resumen de escenarios simulados: cuantiles de precio y cola de pérdida.

Los escenarios vienen de ``VolatilityModel.simulate_returns`` (retornos log
en %, forma ``(n_paths, horizon)``), sea por Monte Carlo paramétrico o por
simulación histórica filtrada, y alimentan los abanicos de precio. El VaR y
el CVaR publicados usan ``filtered_tail_risk``, el estimador validado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.exceptions import ModelFitError
from src.models.volatility import VolatilityModel
from src.schemas import PriceQuantiles

PERCENTILES: tuple[int, ...] = (5, 25, 50, 75, 95)


def price_quantiles(
    last_close: float, sim_returns_pct: np.ndarray
) -> list[PriceQuantiles]:
    """Cuantiles del precio al cierre de cada día del horizonte.

    Parameters
    ----------
    last_close:
        Último cierre observado (origen del pronóstico).
    sim_returns_pct:
        Retornos log simulados en %, ``(n_paths, horizon)``.

    Returns
    -------
    list[PriceQuantiles]
        Una fila por día con los percentiles 5, 25, 50, 75 y 95.
    """
    cumulative = np.cumsum(sim_returns_pct / 100.0, axis=1)
    prices = last_close * np.exp(cumulative)
    levels = np.percentile(prices, PERCENTILES, axis=0)
    return [
        PriceQuantiles(
            day=day + 1,
            **{
                f"p{p:02d}": float(levels[i, day])
                for i, p in enumerate(PERCENTILES)
            },
        )
        for day in range(prices.shape[1])
    ]


def filtered_tail_risk(
    model: VolatilityModel, frame: pd.DataFrame, level: float
) -> tuple[float, float]:
    """VaR y CVaR filtrados del día siguiente (log-retorno, proporción).

    Es el mismo estimador que se valida con Kupiec y Christoffersen en el
    walk-forward: el cuantil empírico de los residuos estandarizados
    ``r / sigma`` del historial, multiplicado por la sigma pronosticada. No
    supone normalidad ni una t; hereda las colas observadas.

    Parameters
    ----------
    model:
        Modelo de volatilidad (el campeón).
    frame:
        Historial hasta el origen del pronóstico (``volatility_frame``).
    level:
        Probabilidad de cola, p. ej. 0.05.

    Returns
    -------
    tuple[float, float]
        ``(VaR, CVaR)``; el CVaR es la pérdida media más allá del VaR.

    Raises
    ------
    ModelFitError
        Si no hay residuos estandarizados finitos.
    """
    path = model.variance_path(frame).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = frame["return_pct"].to_numpy() / np.sqrt(path)
    standardized = standardized[np.isfinite(standardized)]
    if standardized.size == 0:
        raise ModelFitError("Sin residuos estandarizados para el VaR.")
    z_var = float(np.quantile(standardized, level))
    z_cvar = float(standardized[standardized <= z_var].mean())
    sigma = float(np.sqrt(model.forecast_variance(frame, 1)[0])) / 100.0
    return z_var * sigma, z_cvar * sigma
